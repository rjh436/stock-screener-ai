from __future__ import annotations

import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator

from strategies.generic import GenericStrategy
from strategies.strategy_loader import load_strategies
from execution.parity import (
    apply_strategy_score_multipliers,
    DEFAULT_SCORING_WEIGHTS,
)
from execution.shared_logic import _generic_exit_decision

MIN_BARS = 200
MIN_ENTRY_SCORE = 120.0
SUPER_SIGNAL_NAME = "SUPER SIGNAL (Wealth + Income)"

_ATR_HIGH_THRESH_PCT = 3.0
_ATR_MED_THRESH_PCT = 2.0
_VOL_REL_THRESH = 1.5
_VCP_BB_WIDTH_THRESH = 0.17

# Diagnostics: enable to print every strategy's trail activation on each run.
_DEBUG_TRAIL_ACTIVATION = os.environ.get("APEX_DEBUG_TRAIL_ACTIVATION", "").strip() not in ("", "0", "false", "False")


def get_sector(symbol: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"


def inject_market_rs_rank(
    enriched,
    all_dates,
    lookbacks=(63, 126, 189, 252),
    weights=(0.40, 0.20, 0.20, 0.20),
    min_history=252,
    min_names=100,
):
    """
    Creates rs_rating in [1..99] for each symbol/day using cross-sectional percentile rank
    of weighted multi-horizon returns. NaN-safe; IPO-safe via min_history.
    """
    syms = list(enriched.keys())
    n_days = len(all_dates)
    n_syms = len(syms)

    if n_syms == 0:
        return

    # Build matrix: rows=dates, cols=symbols
    close_mat = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    for j, sym in enumerate(syms):
        sd = enriched[sym]
        # sd.gidx maps local bars -> global all_dates indices
        # Check bounds to be safe
        valid_mask = (sd.gidx >= 0) & (sd.gidx < n_days)
        close_mat[sd.gidx[valid_mask], j] = sd.close[valid_mask].astype(np.float32)

    # Weighted return score
    tmp = np.zeros((n_days, n_syms), dtype=np.float32)

    # Track valid history to avoid ranking IPOs too early
    valid_hist = np.isfinite(close_mat)
    cum = np.cumsum(valid_hist, axis=0)
    enough = cum >= min_history

    # Calculate weighted returns
    # This vectorizes the sliding window return calc across all symbols
    for lb, w in zip(lookbacks, weights):
        # pct_change(lb) equivalent in numpy: (arr / arr shifted) - 1
        shifted = np.roll(close_mat, lb, axis=0)
        # Shift introduces wrap-around at the top; mask those out
        shifted[:lb, :] = np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (close_mat / shifted) - 1.0

        # Where ret is nan, contribute 0? No, if any component is NaN, result is NaN.
        # We rely on 'enough' mask later, but intermediate NaNs propagate.
        tmp += w * ret

    # Only score if we have enough history and valid data
    score = np.where(enough & np.isfinite(tmp), tmp, np.nan)

    # Rank day-by-day in chunks to save memory
    rs_rating = np.zeros((n_days, n_syms), dtype=np.float32)

    chunk = 250
    for start in range(0, n_days, chunk):
        end = min(n_days, start + chunk)
        block = score[start:end, :]

        for i in range(block.shape[0]):
            row = block[i]
            m = np.isfinite(row)
            k = int(m.sum())
            if k < min_names:
                continue

            vals = row[m]
            # argsort gives indices that would sort the array
            order = np.argsort(vals, kind="quicksort")
            # ranks[k] = rank of value at index k
            ranks = np.empty_like(order, dtype=np.int32)
            ranks[order] = np.arange(k, dtype=np.int32)

            # percentile in [0..1] -> mapped to 1..99
            pct = (ranks / (k - 1)) if k > 1 else np.zeros(k, dtype=np.float32)
            rs_rating[start + i, m] = 1.0 + 98.0 * pct

    # Write back to symbol arrays
    for j, sym in enumerate(syms):
        sd = enriched[sym]
        # Map global rating back to local indices
        # sd.gidx contains global indices for each local bar
        valid_indices = (sd.gidx >= 0) & (sd.gidx < n_days)
        if np.any(valid_indices):
            sd.rsrating[valid_indices] = rs_rating[sd.gidx[valid_indices], j]


def simulate_breakout_fill(open_px: float, high_px: float, trigger_px: float, limit_px: float | None = None) -> float | None:
    """
    Conservatively simulates a Buy Stop Limit order on Daily Data.
    Rejects trades if Open > Limit (Gap Up).
    """
    if high_px < trigger_px:
        return None

    if limit_px is not None and open_px > limit_px:
        return None

    fill_px = max(open_px, trigger_px)
    return fill_px * 1.001


def _compute_indicators(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    try:
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()

        if spy_df is not None and not spy_df.empty:
            spy_df = spy_df.copy()
            spy_df.columns = spy_df.columns.str.lower()
        if vix_df is not None and not vix_df.empty:
            vix_df = vix_df.copy()
            vix_df.columns = vix_df.columns.str.lower()

        # Average Daily Range % (20-day)
        hl_range = df["high"] - df["low"]
        df["adr_pct"] = (hl_range / df["close"]).rolling(20).mean() * 100.0

        for p in (10, 20, 50, 200):
            df[f"sma{p}"] = df["close"].rolling(p).mean()
            df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        df["bb_upper"] = df["close"].rolling(20).mean() + (df["close"].rolling(20).std() * 2)
        df["bb_lower"] = df["close"].rolling(20).mean() - (df["close"].rolling(20).std() * 2)
        df["bb_mid"] = df["close"].rolling(20).mean()

        mask = df["bb_mid"] != 0
        df["bb_width"] = 0.0
        df.loc[mask, "bb_width"] = (df.loc[mask, "bb_upper"] - df.loc[mask, "bb_lower"]) / df.loc[mask, "bb_mid"]

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr14_ma20"] = df["atr14"].rolling(20).mean()
        df["natr"] = (df["atr14"] / df["close"]) * 100.0

        df["highest20"] = df["high"].rolling(20).max()
        df["highest20_1"] = df["highest20"].shift(1)
        df["donchian_20"] = df["highest20_1"]
        df["donchian20"] = df["donchian_20"]
        df["highest55"] = df["high"].rolling(55).max()
        df["highest55_1"] = df["highest55"].shift(1)
        df["lowest5"] = df["low"].rolling(5).min()
        df["lowest5_1"] = df["lowest5"].shift(1)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        g2 = (delta.where(delta > 0, 0)).rolling(2).mean()
        l2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = g2 / l2.replace(0, np.nan)
        df["rsi2"] = 100 - (100 / (1 + rs2))

        # Momentum / Velocity Check
        # ROC 60: Percentage change over last 60 trading days (approx 1 quarter)
        df["roc_60"] = df["close"].pct_change(60) * 100.0

        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["plus_di"] = adx.adx_pos()
        df["minus_di"] = adx.adx_neg()

        df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
        df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        if spy_df is not None and not spy_df.empty:
            spy_aligned = spy_df["close"].reindex(df.index).ffill().bfill()
            df["rs_ratio"] = df["close"] / spy_aligned
            df["rs_sma20"] = df["rs_ratio"].rolling(20).mean()
            df["rs_trend"] = df["rs_ratio"] - df["rs_sma20"]
            df["spy_close"] = spy_aligned
            df["spy_sma20"] = spy_aligned.rolling(20).mean()
            df["spy_sma50"] = spy_aligned.rolling(50).mean()
            df["spy_sma200"] = spy_aligned.rolling(200).mean()
            df["rs_mom20"] = (df["rs_ratio"] / df["rs_ratio"].shift(20)) - 1.0
            df["spy_regime"] = (df["spy_close"] > df["spy_sma200"]).astype(int)
        else:
            df["rs_ratio"] = 1.0
            df["rs_trend"] = 0.0
            df["spy_close"] = np.nan
            df["spy_sma20"] = np.nan
            df["spy_sma50"] = np.nan
            df["spy_sma200"] = np.nan
            df["rs_mom20"] = 0.0
            df["spy_regime"] = 0.0

        if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
            vix_aligned = vix_df["close"].reindex(df.index).ffill().bfill()
            df["vix"] = vix_aligned
        elif "vix" in df.columns:
            df["vix"] = df["vix"].ffill().bfill()
        else:
            df["vix"] = 20.0
        df["vix"] = df["vix"].fillna(20.0)

        df["vix_sma20"] = df["vix"].rolling(20).mean()
        df["vix_rel20"] = 0.0
        vix_sma20 = df["vix_sma20"]
        mask = vix_sma20 > 0
        df.loc[mask, "vix_rel20"] = (df.loc[mask, "vix"] / vix_sma20[mask]) - 1.0

        df["sma50"] = df["close"].rolling(50).mean()
        df["sma150"] = df["close"].rolling(150).mean()
        df["sma200"] = df["close"].rolling(200).mean()
        df["high_52w"] = df["high"].rolling(252).max()
        df["low_52w"] = df["low"].rolling(252).min()
        df["sma200_slope"] = df["sma200"].diff(22)
        df["std_20"] = df["close"].rolling(20).std()
        df["bb_width"] = (4 * df["std_20"]) / (df["close"].rolling(20).mean() + 1e-9)
        df["rs_score"] = df["close"].pct_change(126)
        df["donchian20"] = df["highest20_1"]
        df["donchian_20"] = df["highest20_1"]

        if "rs_rating" not in df.columns:
            df["rs_rating"] = 0.0

        return df
    except Exception:
        return df


def _empty_result(name: str, start_cash: float, params: Optional[Dict] = None) -> Dict[str, Any]:
    return {
        "strategy": name,
        "final_value": start_cash,
        "total_trades": 0,
        "hit_rate": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "cagr": 0.0,
        "calmar": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0,
        "avg_days_held": 0.0,
        "exposure_pct": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "max_consecutive_losses": 0,
        "beta": 0.0,
        "avg_signals_per_day": 0.0,
        "Score": 0.0,
        "params": params or {},
        "trades_list": [],
        "equity_curve": [],
    }


@dataclass(frozen=True, slots=True)
class _ScoreWeights:
    rsi_factor: float
    atr_high_bonus: float
    atr_med_bonus: float
    vol_bonus: float
    trend_bonus: float
    trend_penalty: float
    vcp_bonus: float


def _compile_scoring_weights(
    base_override: Optional[Dict[str, float]],
    strategy_override: Optional[Dict[str, float]],
) -> _ScoreWeights:
    merged = dict(DEFAULT_SCORING_WEIGHTS)
    if base_override:
        merged.update(base_override)
    if strategy_override:
        merged.update(strategy_override)

    return _ScoreWeights(
        rsi_factor=float(merged.get("rsi_factor", 0.0) or 0.0),
        atr_high_bonus=float(merged.get("atr_high_bonus", 0.0) or 0.0),
        atr_med_bonus=float(merged.get("atr_med_bonus", 0.0) or 0.0),
        vol_bonus=float(merged.get("vol_bonus", 0.0) or 0.0),
        trend_bonus=float(merged.get("trend_bonus", 0.0) or 0.0),
        trend_penalty=float(merged.get("trend_penalty", 0.0) or 0.0),
        vcp_bonus=float(merged.get("vcp_bonus", 0.0) or 0.0),
    )


def _score_candidate(
    rsi2: float,
    close_px: float,
    atr14: float,
    volume: float,
    vol_ma20: float,
    sma200: float,
    cci: float,
    bb_width: float,
    w: _ScoreWeights,
) -> float:
    np_isfinite = np.isfinite

    if not np_isfinite(rsi2):
        rsi2 = 50.0

    vcp_ok = bool(np_isfinite(cci) and np_isfinite(bb_width) and cci < 0 and bb_width < _VCP_BB_WIDTH_THRESH)

    base_score = (100.0 - float(rsi2)) * w.rsi_factor
    base_score = min(100.0, max(0.0, base_score))

    score = base_score

    if np_isfinite(close_px) and close_px > 0 and np_isfinite(atr14) and atr14 > 0:
        atr_pct = (atr14 / close_px) * 100.0
        if atr_pct >= _ATR_HIGH_THRESH_PCT:
            score += w.atr_high_bonus
        elif atr_pct >= _ATR_MED_THRESH_PCT:
            score += w.atr_med_bonus

    if np_isfinite(volume) and np_isfinite(vol_ma20) and vol_ma20 >= 0:
        vol_rel = volume / (vol_ma20 + 1.0)
        if vol_rel >= _VOL_REL_THRESH:
            score += w.vol_bonus

    if np_isfinite(close_px) and close_px > 0 and np_isfinite(sma200) and sma200 > 0:
        score += w.trend_bonus if close_px > sma200 else w.trend_penalty

    if vcp_ok:
        score += w.vcp_bonus

    return max(0.0, float(score))


def _resolve_scoring_mode(
    strategy_name: str,
    scoring_type: Optional[str] = None,
    type_hint: Optional[str] = None,
) -> str:
    for raw in (scoring_type, type_hint):
        if isinstance(raw, str):
            key = raw.strip().lower()
            if any(token in key for token in ("breakout", "momentum", "vcp", "kinetic")):
                return "breakout"
            if any(token in key for token in ("wealth", "mean", "reversion", "income")):
                return "wealth"

    name = str(strategy_name or "").lower()
    if any(token in name for token in ("breakout", "momentum", "vcp", "kinetic")):
        return "breakout"
    if any(token in name for token in ("wealth", "velocity", "income")):
        return "wealth"
    return "wealth"


def _score_row_dual_core(
    rsi2: float,
    rsi14: float,
    bb_width: float,
    natr: float,
    close_px: float,
    high_52w: float,
    weights: Dict[str, float],
    scoring_mode: str,
) -> float:
    np_isfinite = np.isfinite
    rsi_factor = float(weights.get("rsi_factor", 0.0) or 0.0)
    vcp_bonus = float(weights.get("vcp_bonus", weights.get("sniper_bonus", 0.0)) or 0.0)
    vol_bonus = float(weights.get("vol_bonus", 0.0) or 0.0)
    trend_bonus = float(weights.get("trend_bonus", 0.0) or 0.0)

    if scoring_mode == "breakout":
        if not np_isfinite(rsi14):
            rsi14 = 50.0
        score = (rsi14 * rsi_factor) if rsi14 > 50.0 else -(50.0 - rsi14)
        if np_isfinite(bb_width) and bb_width < _VCP_BB_WIDTH_THRESH:
            score += vcp_bonus
        if np_isfinite(natr) and natr > 2.5:
            score += vol_bonus
        if np_isfinite(close_px) and np_isfinite(high_52w) and high_52w > 0:
            if close_px >= (high_52w * 0.85):
                score += trend_bonus
    else:
        if not np_isfinite(rsi2):
            rsi2 = 50.0
        score = (100.0 - rsi2) * rsi_factor

    return max(0.0, float(score))


def calculate_backtest_quality_score(
    row_or_rsi2: Any,
    strategy_name: str = "",
    weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> float:
    """
    Dual-core ranking engine:
    - Breakout/VCP/Kinetic: high RSI, tight BB width, NATR fuel.
    - Wealth/Mean Reversion: low RSI dip buying.
    """
    merged = dict(DEFAULT_SCORING_WEIGHTS)
    if isinstance(weights, dict):
        merged.update(weights)

    scoring_mode = _resolve_scoring_mode(
        strategy_name,
        scoring_type=kwargs.get("scoring_type"),
        type_hint=kwargs.get("type"),
    )

    if isinstance(row_or_rsi2, (pd.Series, dict)):
        row = row_or_rsi2
        rsi2 = float(row.get("rsi2", 50) or 50)
        rsi14 = float(row.get("rsi14", rsi2) or rsi2)
        bb_width = float(row.get("bb_width", np.nan))
        close_px = float(row.get("close", 0) or 0)
        high_52w = float(row.get("high_52w", np.nan))
        natr = row.get("natr")
        if natr is None or not np.isfinite(natr):
            atr14 = float(row.get("atr14", 0) or 0)
            natr = (atr14 / close_px) * 100.0 if close_px > 0 and atr14 > 0 else 0.0
        else:
            natr = float(natr)
    else:
        rsi2 = float(row_or_rsi2 if row_or_rsi2 is not None else 50.0)
        rsi14 = float(kwargs.get("rsi14", rsi2) or rsi2)
        bb_width = float(kwargs.get("bb_width", np.nan))
        close_px = float(kwargs.get("close", 0.0) or 0.0)
        high_52w = float(kwargs.get("high_52w", np.nan))
        natr = kwargs.get("natr")
        if natr is None or not np.isfinite(natr):
            atr14 = float(kwargs.get("atr14", 0.0) or 0.0)
            natr = (atr14 / close_px) * 100.0 if close_px > 0 and atr14 > 0 else 0.0
        else:
            natr = float(natr)

    return _score_row_dual_core(rsi2, rsi14, bb_width, natr, close_px, high_52w, merged, scoring_mode)


@dataclass(slots=True)
class _SymbolArrays:
    df: pd.DataFrame
    index: np.ndarray
    gidx: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    rsi2: np.ndarray
    rsi14: np.ndarray
    adx: np.ndarray
    stochk: np.ndarray
    atr14: np.ndarray
    natr: np.ndarray
    volma20: np.ndarray
    sma10: np.ndarray
    sma20: np.ndarray
    sma50: np.ndarray
    sma150: np.ndarray
    sma200: np.ndarray
    donchian20: np.ndarray
    sma200slope: np.ndarray
    high52w: np.ndarray
    low52w: np.ndarray
    cci: np.ndarray
    bbwidth: np.ndarray
    bblower: np.ndarray
    bbupper: np.ndarray
    rsratio: np.ndarray
    rstrend: np.ndarray
    rsmom20: np.ndarray
    vix: np.ndarray
    vixsma20: np.ndarray
    vixrel20: np.ndarray
    spyclose: np.ndarray
    spysma20: np.ndarray
    spysma50: np.ndarray
    spysma200: np.ndarray
    spyregime: np.ndarray
    rsrating: np.ndarray
    adr_pct: np.ndarray


@dataclass(slots=True)
class _Candidate:
    sym: str
    entry_px: float
    stop_px: float
    score: float
    strategy_name: str
    strategy_obj: Any
    entry_i: int
    signal_i: int
    is_super_signal: bool = False
    ai_prob: float = 0.0
    size_scalar: float = 1.0


def _strategy_role(params: Dict[str, Any]) -> str:
    raw_type = params.get("type")
    if isinstance(raw_type, str):
        t = raw_type.strip().lower()
        if "wealth" in t:
            return "wealth"
        if "income" in t:
            return "income"

    raw_name = params.get("name")
    if isinstance(raw_name, str):
        n = raw_name.lower()
        if "wealth" in n:
            return "wealth"
        if "income" in n:
            return "income"

    return ""


def _apply_super_signal_overrides(genome: Dict[str, Any]) -> Dict[str, Any]:
    """
    Super Signal execution mode: honor the genome without overrides.
    """
    return genome


@dataclass(slots=True)
class PreparedBacktestData:
    """
    Pre-computed indicator + NumPy array pack for fast repeated backtests.

    This is designed for the optimizer loop where the same market data is reused
    across many genomes (threads).
    """

    enriched: Dict[str, _SymbolArrays]
    all_dates: np.ndarray


def _load_sector_map() -> Dict[str, str]:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "sectors.json"))
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k).upper(): str(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _resolve_gap_protection_ratio(params: Dict[str, Any]) -> float:
    raw_ratio = params.get("gap_protection")
    if raw_ratio is None:
        raw_pct = params.get("gap_protection_pct")
        if raw_pct is None:
            return 0.0
        try:
            pct = float(raw_pct)
        except (TypeError, ValueError):
            return 0.0
        if pct <= 0:
            return 0.0
        return max(0.0, 1.0 - pct)
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        return 0.0
    return ratio if ratio > 0 else 0.0


def _get_param(params: Dict[str, Any], key: str, default: float, min_val: float, max_val: float) -> float:
    try:
        val = float(params.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(min_val, min(val, max_val))


def _get_np_col(df: pd.DataFrame, col: str, fallback: float, *, length: int) -> np.ndarray:
    if col in df.columns:
        return df[col].to_numpy(dtype=np.float64, copy=False)
    return np.full(length, fallback, dtype=np.float64)


def prepare_backtest_data(
    data: Dict[str, pd.DataFrame],
    symbol_universe: Optional[Sequence[str]] = None,
    start_date: Any = None,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> PreparedBacktestData:
    """
    Pre-compute indicators + NumPy arrays once, so repeated `run_backtest` calls
    (e.g., in the GA) don't recompute rolling indicators per genome.
    """
    data_dict: Dict[str, pd.DataFrame] = data or {}

    symbols = list(data_dict.keys())
    if symbol_universe:
        universe_set = set(symbol_universe)
        symbols = [s for s in symbols if s in universe_set]

    vix_df = global_data.get("VIX") if global_data else None
    spy_df = global_data.get("SPY") if global_data else None

    enriched: Dict[str, _SymbolArrays] = {}

    for sym in symbols:
        df_raw = data_dict.get(sym)
        if df_raw is None or df_raw.empty:
            continue

        try:
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df, vix_df=vix_df)

            if "ticker" not in df.columns:
                df["ticker"] = sym

            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]

            if len(df) <= MIN_BARS:
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            n = len(df)
            open_arr = _get_np_col(df, "open", np.nan, length=n)
            high_arr = _get_np_col(df, "high", np.nan, length=n)
            low_arr = _get_np_col(df, "low", np.nan, length=n)
            close_arr = _get_np_col(df, "close", np.nan, length=n)

            if not np.isfinite(low_arr).any():
                low_arr = open_arr
            if not np.isfinite(high_arr).any():
                high_arr = open_arr

            enriched[sym] = _SymbolArrays(
                df=df,
                index=df.index.values.astype("datetime64[ns]"),
                gidx=np.empty(n, dtype=np.int32),
                open=open_arr,
                high=high_arr,
                low=low_arr,
                close=close_arr,
                volume=_get_np_col(df, "volume", 0.0, length=n),
                rsi2=_get_np_col(df, "rsi2", 50.0, length=n),
                rsi14=_get_np_col(df, "rsi14", 50.0, length=n),
                adx=_get_np_col(df, "adx", 0.0, length=n),
                stochk=_get_np_col(df, "stoch_k", 0.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                natr=_get_np_col(df, "natr", 0.0, length=n),
                sma10=_get_np_col(df, "sma10", np.nan, length=n),
                volma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma20=_get_np_col(df, "sma20", np.nan, length=n),
                sma50=_get_np_col(df, "sma50", np.nan, length=n),
                sma150=_get_np_col(df, "sma150", np.nan, length=n),
                sma200=_get_np_col(df, "sma200", np.nan, length=n),
                donchian20=_get_np_col(df, "donchian20", np.nan, length=n),
                sma200slope=_get_np_col(df, "sma200_slope", np.nan, length=n),
                high52w=_get_np_col(df, "high_52w", np.nan, length=n),
                low52w=_get_np_col(df, "low_52w", np.nan, length=n),
                cci=_get_np_col(df, "cci", 0.0, length=n),
                bbwidth=_get_np_col(df, "bb_width", 0.0, length=n),
                bblower=_get_np_col(df, "bb_lower", np.nan, length=n),
                bbupper=_get_np_col(df, "bb_upper", np.nan, length=n),
                rsratio=_get_np_col(df, "rs_ratio", 1.0, length=n),
                rstrend=_get_np_col(df, "rs_trend", 0.0, length=n),
                rsmom20=_get_np_col(df, "rs_mom20", 0.0, length=n),
                vix=_get_np_col(df, "vix", 20.0, length=n),
                vixsma20=_get_np_col(df, "vix_sma20", 20.0, length=n),
                vixrel20=_get_np_col(df, "vix_rel20", 0.0, length=n),
                spyclose=_get_np_col(df, "spy_close", np.nan, length=n),
                spysma20=_get_np_col(df, "spy_sma20", np.nan, length=n),
                spysma50=_get_np_col(df, "spy_sma50", np.nan, length=n),
                spysma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
                spyregime=_get_np_col(df, "spy_regime", 0.0, length=n),
                rsrating=_get_np_col(df, "rs_rating", 0.0, length=n),
                adr_pct=_get_np_col(df, "adr_pct", 0.0, length=n),
            )
        except Exception:
            continue

    if not enriched:
        return PreparedBacktestData(enriched={}, all_dates=np.array([], dtype="datetime64[ns]"))

    all_index = None
    for sym_data in enriched.values():
        all_index = sym_data.df.index if all_index is None else all_index.union(sym_data.df.index)
    all_index = all_index.sort_values()

    min_date = pd.Timestamp.now() - pd.Timedelta(days=365 * 20)
    all_index = all_index[all_index >= min_date]
    if all_index.empty:
        return PreparedBacktestData(enriched={}, all_dates=np.array([], dtype="datetime64[ns]"))

    all_dates = all_index.values.astype("datetime64[ns]")

    for sym_data in enriched.values():
        sym_data.gidx = np.searchsorted(all_dates, sym_data.index).astype(np.int32, copy=False)

    inject_market_rs_rank(enriched, all_dates)
    for sym_data in enriched.values():
        if "rs_rating" in sym_data.df.columns:
            sym_data.df["rs_rating"] = sym_data.rsrating

    return PreparedBacktestData(enriched=enriched, all_dates=all_dates)


def _legacy_run_backtest(
    strategy,
    data,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    global_data=None,
    scoring_weights=None,
    export_ml_data=False,
    ai_model=None,
    ai_threshold=0.60,
    super_signal_only: bool = False,
):
    def _unwrap_genome(params: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict) and isinstance(params.get("genome"), dict):
            return params["genome"]
        return params

    strategies = strategy if isinstance(strategy, (list, tuple)) else [strategy]
    strategies = [s for s in strategies if s is not None]
    if not strategies:
        return _empty_result("NoStrategy", float(start_cash), {})

    data_dict: Dict[str, pd.DataFrame] = data or {}
    strategy_label = strategies[0].name if len(strategies) == 1 else "MultiStrategy"
    if super_signal_only:
        strategy_label = SUPER_SIGNAL_NAME

    sector_map = _load_sector_map()

    def resolve_sector(sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in sector_map:
            return sector_map[sym_up]
        return get_sector(sym_up)

    symbols = list(data_dict.keys())
    if symbol_universe:
        universe_set = set(symbol_universe)
        symbols = [s for s in symbols if s in universe_set]

    vix_df = global_data.get("VIX") if global_data else None
    spy_df = global_data.get("SPY") if global_data else None

    enriched: Dict[str, _SymbolArrays] = {}
    for sym in symbols:
        df_raw = data_dict.get(sym)
        if df_raw is None or df_raw.empty:
            continue

        try:
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df, vix_df=vix_df)

            if "ticker" not in df.columns:
                df["ticker"] = sym

            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]

            if len(df) <= MIN_BARS:
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            n = len(df)
            open_arr = _get_np_col(df, "open", np.nan, length=n)
            high_arr = _get_np_col(df, "high", np.nan, length=n)
            low_arr = _get_np_col(df, "low", np.nan, length=n)
            close_arr = _get_np_col(df, "close", np.nan, length=n)
            if not np.isfinite(low_arr).any():
                low_arr = open_arr
            if not np.isfinite(high_arr).any():
                high_arr = open_arr

            enriched[sym] = _SymbolArrays(
                df=df,
                index=df.index.values.astype("datetime64[ns]"),
                gidx=np.empty(n, dtype=np.int32),
                open=open_arr,
                high=high_arr,
                low=low_arr if low_arr is not None else open_arr,
                close=close_arr,
                volume=_get_np_col(df, "volume", 0.0, length=n),
                rsi2=_get_np_col(df, "rsi2", 50.0, length=n),
                rsi14=_get_np_col(df, "rsi14", 50.0, length=n),
                adx=_get_np_col(df, "adx", 0.0, length=n),
                stochk=_get_np_col(df, "stoch_k", 0.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                natr=_get_np_col(df, "natr", 0.0, length=n),
                sma10=_get_np_col(df, "sma10", np.nan, length=n),
                volma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma20=_get_np_col(df, "sma20", np.nan, length=n),
                sma50=_get_np_col(df, "sma50", np.nan, length=n),
                sma150=_get_np_col(df, "sma150", np.nan, length=n),
                sma200=_get_np_col(df, "sma200", np.nan, length=n),
                donchian20=_get_np_col(df, "donchian20", np.nan, length=n),
                sma200slope=_get_np_col(df, "sma200_slope", np.nan, length=n),
                high52w=_get_np_col(df, "high_52w", np.nan, length=n),
                low52w=_get_np_col(df, "low_52w", np.nan, length=n),
                cci=_get_np_col(df, "cci", 0.0, length=n),
                bbwidth=_get_np_col(df, "bb_width", 0.0, length=n),
                bblower=_get_np_col(df, "bb_lower", np.nan, length=n),
                bbupper=_get_np_col(df, "bb_upper", np.nan, length=n),
                rsratio=_get_np_col(df, "rs_ratio", 1.0, length=n),
                rstrend=_get_np_col(df, "rs_trend", 0.0, length=n),
                rsmom20=_get_np_col(df, "rs_mom20", 0.0, length=n),
                vix=_get_np_col(df, "vix", 20.0, length=n),
                vixsma20=_get_np_col(df, "vix_sma20", 20.0, length=n),
                vixrel20=_get_np_col(df, "vix_rel20", 0.0, length=n),
                spyclose=_get_np_col(df, "spy_close", np.nan, length=n),
                spysma20=_get_np_col(df, "spy_sma20", np.nan, length=n),
                spysma50=_get_np_col(df, "spy_sma50", np.nan, length=n),
                spysma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
                spyregime=_get_np_col(df, "spy_regime", 0.0, length=n),
                rsrating=_get_np_col(df, "rs_rating", 0.0, length=n),
                adr_pct=_get_np_col(df, "adr_pct", 0.0, length=n),
            )
        except Exception:
            continue

    if not enriched:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    all_index = None
    for sym_data in enriched.values():
        all_index = sym_data.df.index if all_index is None else all_index.union(sym_data.df.index)
    all_index = all_index.sort_values()

    min_date = pd.Timestamp.now() - pd.Timedelta(days=365 * 20)
    all_index = all_index[all_index >= min_date]
    if all_index.empty:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    all_dates = all_index.values.astype("datetime64[ns]")
    for sym_data in enriched.values():
        sym_data.gidx = np.searchsorted(all_dates, sym_data.index).astype(np.int32, copy=False)

    inject_market_rs_rank(enriched, all_dates)
    for sym_data in enriched.values():
        if "rs_rating" in sym_data.df.columns:
            sym_data.df["rs_rating"] = sym_data.rsrating

    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]

    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float, float, bool, str]] = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _unwrap_genome(raw_params)
        strat_weights = params.get("scoring_weights") if isinstance(params.get("scoring_weights"), dict) else None
        w = _compile_scoring_weights(scoring_weights, strat_weights)
        gap_ratio = _resolve_gap_protection_ratio(params)
        regime_filter = bool(params.get("regime_filter", False))
        try:
            min_adx = float(params.get("min_adx", 0) or 0)
        except (TypeError, ValueError):
            min_adx = 0.0
        if min_adx <= 0:
            min_adx = 0.0
        vix_limit_scaling = bool(params.get("vix_limit_scaling", False))
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        scoring_mode = _resolve_scoring_mode(
            strat.name,
            scoring_type=params.get("scoring_type"),
            type_hint=params.get("type"),
        )
        compiled_strategies.append(
            (strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling, scoring_mode)
        )

    np_isfinite = np.isfinite

    for sym, sd in enriched.items():
        df = sd.df
        idx = sd.index
        open_arr = sd.open
        low_arr = sd.low
        close_arr = sd.close
        volume_arr = sd.volume
        rsi2_arr = sd.rsi2
        rsi14_arr = sd.rsi14
        adx_arr = sd.adx
        atr14_arr = sd.atr14
        vol_ma20_arr = sd.volma20
        sma20_arr = sd.sma20
        sma50_arr = sd.sma50
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bbwidth
        rs_ratio_arr = sd.rsratio
        rs_trend_arr = sd.rstrend
        rs_mom20_arr = sd.rsmom20
        vix_arr = sd.vix
        vix_rel20_arr = sd.vixrel20
        spy_close_arr = sd.spyclose
        spy_sma200_arr = sd.spysma200
        spy_regime_arr = sd.spyregime

        n = len(idx)
        if n <= MIN_BARS + 1:
            continue

        for i in range(MIN_BARS + 1, n):
            day_idx = date_to_idx.get(idx[i])
            if day_idx is None:
                continue

            prev_i = i - 1

            prev_close = float(close_arr[prev_i])
            if not np_isfinite(prev_close) or prev_close <= 0:
                continue

            open_px = float(open_arr[i])
            if not np_isfinite(open_px) or open_px <= 0:
                continue

            low_px = float(low_arr[i])
            if not np_isfinite(low_px):
                low_px = open_px

            gap_pct = (open_px - prev_close) / prev_close if prev_close > 0 else 0.0
            if gap_pct < -0.08:
                continue

        for strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling, scoring_mode in compiled_strategies:
                entry_signal = strat.entry(df, prev_i)
                if not entry_signal:
                    continue

                market_filter_mode = str(params.get("market_filter_mode") or "").lower()
                if market_filter_mode == "traffic_light":
                    spy_close_prev = float(spy_close_arr[prev_i])
                    spy_sma200_prev = float(spy_sma200_arr[prev_i])
                    spy_sma20_prev = float(spy_sma20_arr[prev_i])
                    rs_rating_prev = float(rs_rating_arr[prev_i]) if np_isfinite(rs_rating_arr[prev_i]) else 0.0
                    if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                        if spy_close_prev < spy_sma200_prev:
                            try:
                                bypass_threshold = float(params.get("red_bypass_rs", 100.0))
                            except (TypeError, ValueError):
                                bypass_threshold = 100.0
                            if rs_rating_prev < bypass_threshold:
                                continue
                        if np_isfinite(spy_sma20_prev) and spy_close_prev < spy_sma20_prev:
                            yellow_floor = float(params.get("yellow_rs_floor", 92.0) or 92.0)
                            if rs_rating_prev < yellow_floor:
                                continue
                elif regime_filter:
                    spy_close_prev = float(spy_close_arr[prev_i])
                    spy_sma200_prev = float(spy_sma200_arr[prev_i])
                    if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                        if spy_close_prev < spy_sma200_prev:
                            continue

                if min_adx > 0:
                    adx_prev = float(adx_arr[prev_i])
                    if not np_isfinite(adx_prev) or adx_prev < min_adx:
                        continue

                if gap_ratio > 0 and prev_close > 0:
                    if open_px < prev_close * gap_ratio:
                        continue

                signal_atr = float(atr14_arr[prev_i])
                if not np_isfinite(signal_atr) or signal_atr <= 0:
                    continue

                entry_px = open_px
                limit_ratio = None
                if isinstance(entry_signal, dict):
                    limit_ratio = entry_signal.get("limit_ratio")
                if limit_ratio is None:
                    limit_ratio = params.get("limit_ratio")

                if limit_ratio is not None and prev_close > 0:
                    try:
                        limit_ratio_val = float(limit_ratio)
                    except (TypeError, ValueError):
                        limit_ratio_val = None
                    if limit_ratio_val is not None:
                        if vix_limit_scaling:
                            vix_prev = float(vix_arr[prev_i])
                            if np_isfinite(vix_prev) and vix_prev > 25.0:
                                limit_ratio_val *= 0.98
                        target_px = prev_close * limit_ratio_val
                        if open_px < target_px:
                            entry_px = open_px
                        elif low_px < target_px:
                            entry_px = target_px
                        else:
                            continue

                stop_mult = base_stop_mult
                if isinstance(entry_signal, dict) and "stop_loss_atr" in entry_signal:
                    try:
                        stop_mult = float(entry_signal.get("stop_loss_atr") or base_stop_mult)
                    except (TypeError, ValueError):
                        stop_mult = base_stop_mult

                atr_pct_for_stop = (signal_atr / entry_px) * 100.0 if entry_px > 0 else 0.0
                adjusted_mult = stop_mult
                if gap_pct > -0.03 and atr_pct_for_stop > 5.0:
                    adjusted_mult *= 1.2

                stop_price = entry_px - (signal_atr * adjusted_mult)
                if not np_isfinite(stop_price):
                    continue

                # --- FIX: Use Dual-Core Ranking Engine ---
                # Was: score = _score_candidate(...) which forced mean-reversion logic

                # Calculate derived metrics for the new engine
                current_natr = 0.0
                if prev_close > 0:
                    current_natr = (signal_atr / prev_close) * 100.0

                # Pass explicit kwargs to support the Momentum/Breakout logic
                score = calculate_backtest_quality_score(
                    row_or_rsi2=float(rsi2_arr[prev_i]), # Legacy support
                    strategy_name=strat.name,
                    weights=params.get("scoring_weights"),
                    scoring_type=scoring_mode,

                    # --- CRITICAL: MOMENTUM SIGNALS ---
                    rsi14=float(rsi14_arr[prev_i]),
                    bb_width=float(bb_width_arr[prev_i]),
                    close=prev_close,
                    natr=current_natr,
                    # We use 'high' as a proxy for 52w high if the array isn't strictly tracked in this scope,
                    # or pass 0.0 if not available. The engine defaults safely.
                    high_52w=float(high_arr[prev_i]),

                    # --- LEGACY SIGNALS ---
                    atr14=signal_atr,
                    volume=float(volume_arr[prev_i]),
                    vol_ma20=float(vol_ma20_arr[prev_i]),
                    sma200=float(sma200_arr[prev_i]),
                    cci=float(cci_arr[prev_i])
                )
                score = apply_strategy_score_multipliers(score, params)
                strat_min_score = float(params.get("min_entry_score", MIN_ENTRY_SCORE))
                if score < strat_min_score:
                    continue

                candidates_by_day[day_idx].append(
                    _Candidate(
                        sym=sym,
                        entry_px=float(entry_px),
                        stop_px=float(stop_price),
                        score=float(score),
                        strategy_name=strat.name,
                        strategy_obj=strat,
                        entry_i=i,
                        signal_i=prev_i,
                        ai_prob=0.0,
                        size_scalar=1.0,
                    )
                )

    for day_list in candidates_by_day:
        if len(day_list) > 1:
            day_list.sort(key=lambda c: c.score, reverse=True)

    cash = float(start_cash)
    positions: Dict[str, Dict[str, Any]] = {}
    equity_curve: List[Dict[str, Any]] = []
    trade_pnls: List[float] = []
    trades_list: List[Dict[str, Any]] = []
    ml_data: Optional[List[Dict[str, Any]]] = [] if export_ml_data else None

    pos_fraction = 0.20
    max_positions = 5
    max_risk_per_trade = 0.02

    for day_idx, current_dt in enumerate(all_dates):
        sector_exposure: Dict[str, float] = {}
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                val = pos["shares"] * pos["entry_price"]
            else:
                loc = int(np.searchsorted(sym_data.index, current_dt))
                if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                    px = float(sym_data.close[loc])
                    if not np_isfinite(px):
                        px = float(pos["entry_price"])
                else:
                    px = float(pos["entry_price"])
                val = float(pos["shares"]) * px
            sec = resolve_sector(sym)
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + float(val)

        daily_candidates = candidates_by_day[day_idx]

        current_sector_equity = float(sum(sector_exposure.values()))
        for cand in daily_candidates:
            strat_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
            if isinstance(strat_params, dict):
                strat_min_score = float(strat_params.get("min_entry_score", MIN_ENTRY_SCORE))
            else:
                strat_min_score = MIN_ENTRY_SCORE
            if cand.score < strat_min_score:
                continue
            if cand.sym in positions:
                continue
            if not export_ml_data and len(positions) >= max_positions:
                break

            current_equity = cash + current_sector_equity
            if current_equity <= 0:
                continue

            risk_amt = current_equity * max_risk_per_trade
            risk_per_share = cand.entry_px - cand.stop_px
            if risk_per_share <= cand.entry_px * 0.01:
                risk_per_share = cand.entry_px * 0.01

            shares_risk = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            val_cap = current_equity * pos_fraction
            shares_val = int(val_cap / cand.entry_px) if cand.entry_px > 0 else 0
            shares = min(shares_risk, shares_val)

            if shares > 0:
                strat_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
                if isinstance(strat_params, dict) and bool(strat_params.get("vix_position_sizing", False)):
                    sym_data = enriched.get(cand.sym)
                    if sym_data is not None:
                        sig_i = int(cand.signal_i)
                        if 0 <= sig_i < sym_data.vix.size:
                            vix_prev = float(sym_data.vix[sig_i])
                            if np_isfinite(vix_prev) and vix_prev > 25.0:
                                shares = int(shares * 0.5)

            cost = shares * cand.entry_px
            if export_ml_data and (shares <= 0 or cash < cost):
                shares = 100
                cost = 0.0
            elif shares <= 0 or cash < cost:
                continue

            cand_sec = resolve_sector(cand.sym)
            projected_exp = (sector_exposure.get(cand_sec, 0.0) + cost) / current_equity if current_equity > 0 else 1.0
            if not export_ml_data and projected_exp > 0.60:
                continue

            cash -= cost
            positions[cand.sym] = {
                "shares": shares,
                "entry_price": cand.entry_px,
                "stop_price": cand.stop_px,
                "entry_i": cand.entry_i,
                "strategy_name": cand.strategy_name,
                "strategy_obj": cand.strategy_obj,
                "signal_i": cand.signal_i,
                "is_super_signal": bool(getattr(cand, "is_super_signal", False)),
            }
            sector_exposure[cand_sec] = sector_exposure.get(cand_sec, 0.0) + cost
            current_sector_equity += cost

        for sym in list(positions.keys()):
            sym_data = enriched.get(sym)
            if sym_data is None:
                continue

            loc = int(np.searchsorted(sym_data.index, current_dt))
            if not (0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt):
                continue

            pos = positions[sym]
            if loc <= int(pos.get("entry_i", 0)):
                continue

            active_strat = pos.get("strategy_obj")
            if not active_strat:
                continue

            try:
                genome = getattr(active_strat, "params", getattr(active_strat, "genome", {})) or {}

                entry_i = int(pos.get("entry_i", 0) or 0)
                entry_price = float(pos["entry_price"])
                initial_stop = float(pos["stop_price"])

                if isinstance(active_strat, GenericStrategy) and active_strat.__class__.exit is GenericStrategy.exit:
                    should_exit, effective_stop, target_px = _generic_exit_decision(
                        genome,
                        sym_data,
                        loc,
                        entry_i,
                        entry_price,
                        initial_stop,
                    )
                else:
                    exit_result = active_strat.exit(sym_data.df, loc, entry_i, entry_price, initial_stop)
                    updated_stop = None
                    target_px = None
                    if isinstance(exit_result, tuple):
                        should_exit = bool(exit_result[0])
                        if len(exit_result) > 1:
                            updated_stop = exit_result[1]
                        if len(exit_result) > 2:
                            target_px = exit_result[2]
                    else:
                        should_exit = bool(exit_result)
                    if updated_stop is not None and np_isfinite(updated_stop):
                        updated_stop = float(updated_stop)
                        pos["stop_price"] = max(float(pos["stop_price"]), updated_stop)
                    effective_stop = float(pos["stop_price"])
            except Exception:
                should_exit = False
                effective_stop = float(pos["stop_price"])
                target_px = None

            if not should_exit:
                continue

            open_px = float(sym_data.open[loc])
            low_px = float(sym_data.low[loc])
            high_px = float(sym_data.high[loc])
            close_px = float(sym_data.close[loc])

            if not np_isfinite(low_px):
                low_px = close_px
            if not np_isfinite(high_px):
                high_px = close_px

            exit_px = close_px
            if np_isfinite(effective_stop) and low_px < effective_stop:
                exit_px = effective_stop if open_px >= effective_stop else open_px
            elif target_px is not None and np_isfinite(target_px) and high_px >= target_px:
                exit_px = float(target_px)
            if exit_px < pos["entry_price"] * 0.5:
                exit_px = pos["entry_price"] * 0.5

            pnl = (exit_px - pos["entry_price"]) * pos["shares"]
            pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100.0 if pos["entry_price"] else 0.0

            cash += pos["shares"] * exit_px
            trade_pnls.append(float(pnl))

            if export_ml_data and ml_data is not None:
                try:
                    entry_feat_i = int(pos.get("signal_i") or pos.get("entry_i") or 0)
                    entry_feat_i = max(0, min(entry_feat_i, len(sym_data.index) - 1))

                    close_entry = float(sym_data.close[entry_feat_i])
                    atr14_entry = float(sym_data.atr14[entry_feat_i])
                    sma50_entry = float(sym_data.sma50[entry_feat_i])
                    sma200_entry = float(sym_data.sma200[entry_feat_i])
                    vol_entry = float(sym_data.volume[entry_feat_i])
                    vol_ma20_entry = float(sym_data.volma20[entry_feat_i])
                    vix_entry = float(sym_data.vix[entry_feat_i])
                    vix_rel20_entry = float(sym_data.vixrel20[entry_feat_i])
                    rs_ratio_entry = float(sym_data.rsratio[entry_feat_i])
                    rs_trend_entry = float(sym_data.rstrend[entry_feat_i])
                    rs_mom20_entry = float(sym_data.rsmom20[entry_feat_i])
                    spy_regime_entry = float(sym_data.spyregime[entry_feat_i])

                    dist50 = (close_entry - sma50_entry) / close_entry if close_entry else 0.0
                    dist200 = (close_entry - sma200_entry) / close_entry if close_entry else 0.0
                    atr_pct = atr14_entry / close_entry if close_entry else 0.0
                    vol_rel = vol_entry / (vol_ma20_entry + 1.0)

                    ml_data.append(
                        {
                            "rsi2": float(sym_data.rsi2[entry_feat_i]),
                            "adx": float(sym_data.adx[entry_feat_i]),
                            "atr14": atr14_entry,
                            "atr_pct": float(atr_pct),
                            "vol_rel": float(vol_rel),
                            "dist_to_sma50": float(dist50),
                            "dist_to_sma200": float(dist200),
                            "dist_sma50": float(dist50),
                            "dist_sma200": float(dist200),
                            "vix": vix_entry,
                            "vix_rel20": vix_rel20_entry,
                            "rs_ratio": rs_ratio_entry,
                            "rs_trend": rs_trend_entry,
                            "rs_mom20": rs_mom20_entry,
                            "spy_regime": spy_regime_entry,
                            "profit_pct": float(pct),
                            "outcome": 1 if pct > 0 else 0,
                        }
                    )
                except Exception:
                    pass

            trades_list.append(
                {
                    "Symbol": sym,
                    "Entry Date": str(sym_data.df.index[pos["entry_i"]].date()),
                    "Exit Date": str(pd.Timestamp(current_dt).date()),
                    "Entry": pos["entry_price"],
                    "Exit": exit_px,
                    "PnL": pnl,
                    "Return%": pct,
                    "Strategy": pos.get("strategy_name", strategy_label),
                    "is_super_signal": bool(pos.get("is_super_signal", False)),
                }
            )
            del positions[sym]

        equity = cash
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                equity += pos["shares"] * pos["entry_price"]
                continue
            loc = int(np.searchsorted(sym_data.index, current_dt))
            if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                px = float(sym_data.close[loc])
                if not np_isfinite(px):
                    px = float(pos["entry_price"])
            else:
                px = float(pos["entry_price"])
            equity += float(pos["shares"]) * px

        equity_curve.append({"Date": pd.Timestamp(current_dt), "Equity": float(equity)})

    trades = len(trade_pnls)
    wins = sum(1 for t in trade_pnls if t > 0)
    hit_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    avg_profit = (sum(t.get("Return%", 0.0) for t in trades_list) / len(trades_list)) if trades_list else 0.0

    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty:
        eq_df = pd.DataFrame({"Equity": [float(start_cash)]}, index=[pd.Timestamp.now()])

    days = int((eq_df.index[-1] - eq_df.index[0]).days)
    years = days / 365.25 if days > 0 else 1.0
    cagr = ((eq_df["Equity"].iloc[-1] / float(start_cash)) ** (1.0 / years)) - 1.0 if start_cash else 0.0

    rolling_max = eq_df["Equity"].cummax()
    drawdowns = (eq_df["Equity"] - rolling_max) / rolling_max
    max_dd = float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

    if export_ml_data and ml_data:
        try:
            out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_training_data.csv"))
            pd.DataFrame(ml_data).to_csv(out_path, index=False)
            print(f"🧠 ML Data Exported: {len(ml_data)} samples saved to {out_path}")
        except Exception as e:
            print(f"⚠️ ML Data Export failed: {e}")

    return {
        "strategy": strategy_label,
        "final_value": float(eq_df["Equity"].iloc[-1]) if not eq_df.empty else float(start_cash),
        "total_trades": trades,
        "hit_rate": hit_rate,
        "cagr": float(cagr),
        "avg_profit_pct": float(avg_profit),
        "max_drawdown_pct": float(max_dd),
        "equity_curve": eq_df,
        "trades_list": trades_list,
        "params": strategies[0].params if strategies else {},
        "Score": float(cagr) * 1000.0,
    }


_VEC_OPS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
}


def _vectorized_entry_indices(sd: _SymbolArrays, entry_rules: Sequence[Dict[str, Any]], warmup: int) -> np.ndarray:
    """
    Return *entry-day* indices (i) where the strategy signals on (i-1).
    """
    n = len(sd.index)
    if n <= 1:
        return np.array([], dtype=np.int32)

    warmup = max(int(warmup or MIN_BARS), MIN_BARS)

    mask = np.ones(n, dtype=bool)
    for rule in entry_rules or []:
        if not isinstance(rule, dict):
            return np.array([], dtype=np.int32)
        col = rule.get("col")
        op = rule.get("op")
        op_fn = _VEC_OPS.get(str(op))
        if not col or op_fn is None:
            return np.array([], dtype=np.int32)

        left = getattr(sd, str(col), None)
        if left is None:
            if str(col) in sd.df.columns:
                left = sd.df[str(col)].to_numpy(dtype=np.float64, copy=False)
            else:
                return np.array([], dtype=np.int32)

        if "val" in rule:
            try:
                rhs = float(rule.get("val"))
            except (TypeError, ValueError):
                return np.array([], dtype=np.int32)
            rule_mask = op_fn(left, rhs) & np.isfinite(left)
        elif "ref" in rule:
            ref = str(rule.get("ref") or "")
            right = getattr(sd, ref, None)
            if right is None:
                if ref in sd.df.columns:
                    right = sd.df[ref].to_numpy(dtype=np.float64, copy=False)
                else:
                    return np.array([], dtype=np.int32)
            if "mult" in rule:
                try:
                    mult = float(rule["mult"])
                    right = right * mult
                except (ValueError, TypeError):
                    pass
            rule_mask = op_fn(left, right) & np.isfinite(left) & np.isfinite(right)
        else:
            return np.array([], dtype=np.int32)

        mask &= rule_mask

    mask[:warmup] = False
    mask[-1] = False  # no next bar for entry

    signal_idx = np.flatnonzero(mask)
    return (signal_idx + 1).astype(np.int32, copy=False)


def _vectorized_signal_indices(sd: _SymbolArrays, entry_rules: Sequence[Dict[str, Any]], warmup: int) -> np.ndarray:
    """
    Return *signal-day* indices (i) where the strategy signals on i.
    """
    n = len(sd.index)
    if n <= 1:
        return np.array([], dtype=np.int32)

    warmup = max(int(warmup or MIN_BARS), MIN_BARS)

    mask = np.ones(n, dtype=bool)
    for rule in entry_rules or []:
        if not isinstance(rule, dict):
            return np.array([], dtype=np.int32)
        col = rule.get("col")
        op = rule.get("op")
        op_fn = _VEC_OPS.get(str(op))
        if not col or op_fn is None:
            return np.array([], dtype=np.int32)

        left = getattr(sd, str(col), None)
        if left is None:
            if str(col) in sd.df.columns:
                left = sd.df[str(col)].to_numpy(dtype=np.float64, copy=False)
            else:
                return np.array([], dtype=np.int32)

        if "val" in rule:
            try:
                rhs = float(rule.get("val"))
            except (TypeError, ValueError):
                return np.array([], dtype=np.int32)
            rule_mask = op_fn(left, rhs) & np.isfinite(left)
        elif "ref" in rule:
            ref = str(rule.get("ref") or "")
            right = getattr(sd, ref, None)
            if right is None:
                if ref in sd.df.columns:
                    right = sd.df[ref].to_numpy(dtype=np.float64, copy=False)
                else:
                    return np.array([], dtype=np.int32)
            if "mult" in rule:
                try:
                    mult = float(rule["mult"])
                    right = right * mult
                except (ValueError, TypeError):
                    pass
            rule_mask = op_fn(left, right) & np.isfinite(left) & np.isfinite(right)
        else:
            return np.array([], dtype=np.int32)

        mask &= rule_mask

    mask[:warmup] = False
    mask[0] = False  # need prior close for gap logic
    mask[-1] = False  # avoid last bar for entries

    return np.flatnonzero(mask).astype(np.int32, copy=False)


def _vectorized_breakout_pivot(
    sd: _SymbolArrays,
    entry_rules: Sequence[Dict[str, Any]],
    idx: np.ndarray,
) -> Optional[np.ndarray]:
    pivots = []
    for rule in entry_rules or []:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("col")) != "close":
            continue
        op = str(rule.get("op"))
        if op not in (">", ">="):
            continue
        ref = rule.get("ref")
        if not ref:
            continue
        right = getattr(sd, str(ref), None)
        if right is None:
            if str(ref) in sd.df.columns:
                right = sd.df[str(ref)].to_numpy(dtype=np.float64, copy=False)
            else:
                continue
        vals = right[idx]
        if "mult" in rule:
            try:
                mult = float(rule["mult"])
                vals = vals * mult
            except (ValueError, TypeError):
                pass
        pivots.append(vals)

    if not pivots:
        return None
    return np.nanmax(np.stack(pivots, axis=0), axis=0)


def _resolve_breakout_pivot(row: pd.Series, entry_rules: Sequence[Dict[str, Any]]) -> Optional[float]:
    pivot = None
    for rule in entry_rules or []:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("col")) != "close":
            continue
        op = str(rule.get("op"))
        if op not in (">", ">="):
            continue
        ref = rule.get("ref")
        if not ref:
            continue
        try:
            val = float(row.get(ref, np.nan))
        except (TypeError, ValueError):
            continue
        if "mult" in rule:
            try:
                val *= float(rule["mult"])
            except (TypeError, ValueError):
                pass
        if not np.isfinite(val):
            continue
        pivot = val if pivot is None else max(pivot, val)
    return pivot


def _score_candidates_vectorized(
    rsi2: np.ndarray,
    rsi14: np.ndarray,
    close_px: np.ndarray,
    atr14: np.ndarray,
    volume: np.ndarray,
    vol_ma20: np.ndarray,
    sma200: np.ndarray,
    cci: np.ndarray,
    bb_width: np.ndarray,
    high_52w: np.ndarray,
    w: _ScoreWeights,
    scoring_mode: str = "wealth",
) -> np.ndarray:
    close_px = np.nan_to_num(close_px, nan=0.0)
    rsi2 = np.nan_to_num(rsi2, nan=50.0)
    rsi14 = np.nan_to_num(rsi14, nan=50.0)
    bb_width = np.nan_to_num(bb_width, nan=1.0)
    high_52w = np.nan_to_num(high_52w, nan=0.0)

    natr = np.zeros_like(close_px)
    natr_mask = (close_px > 0) & np.isfinite(atr14)
    natr[natr_mask] = (atr14[natr_mask] / close_px[natr_mask]) * 100.0

    if scoring_mode == "breakout":
        score = np.where(rsi14 > 50.0, rsi14 * float(w.rsi_factor), -(50.0 - rsi14))
        score += np.where(bb_width < _VCP_BB_WIDTH_THRESH, float(w.vcp_bonus), 0.0)
        score += np.where(natr > 2.5, float(w.vol_bonus), 0.0)
        near_high = (high_52w > 0) & (close_px >= (high_52w * 0.85))
        score += np.where(near_high, float(w.trend_bonus), 0.0)
    else:
        score = (100.0 - rsi2) * float(w.rsi_factor)

    return np.maximum(score, 0.0)


def _can_vectorize_entry(strategy_obj: Any, params: Dict[str, Any], ai_model: Any) -> bool:
    # Vectorized path for base GenericStrategy genomes (optimizer hot path).
    if not isinstance(strategy_obj, GenericStrategy):
        return False
    if strategy_obj.__class__.entry is not GenericStrategy.entry:
        return False
    if params.get("use_fundamentals", False):
        return False
    return isinstance(params.get("entry_rules"), list)


def run_backtest(
    strategy,
    data,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    global_data=None,
    scoring_weights=None,
    export_ml_data=False,
    ai_model=None,
    ai_threshold=0.60,
    super_signal_only: bool = False,
):
    """
    VCP-mode backtest engine:
    - Precomputable data path (PreparedBacktestData) for GA speed
    - MIN_ENTRY_SCORE gating for selective entries
    - Explicit support for trail_activation + time_stop for GenericStrategy genomes
    - Optional super-signal confluence mode (Wealth + Income)
    """
    def _unwrap_genome(params: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict) and isinstance(params.get("genome"), dict):
            return params["genome"]
        return params

    strategies = strategy if isinstance(strategy, (list, tuple)) else [strategy]
    strategies = [s for s in strategies if s is not None]
    if not strategies:
        return _empty_result("NoStrategy", float(start_cash), {})

    strategy_label = strategies[0].name if len(strategies) == 1 else "MultiStrategy"

    sector_map = _load_sector_map()

    def resolve_sector(sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in sector_map:
            return sector_map[sym_up]
        return get_sector(sym_up)

    if isinstance(data, PreparedBacktestData):
        prepared = data
    else:
        prepared = prepare_backtest_data(
            data or {},
            symbol_universe=symbol_universe,
            start_date=start_date,
            global_data=global_data,
        )

    enriched = prepared.enriched
    all_dates = prepared.all_dates
    if not enriched or all_dates.size == 0:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]
    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float, float, bool, str]] = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _unwrap_genome(raw_params)
        strat_weights = params.get("scoring_weights") if isinstance(params.get("scoring_weights"), dict) else None
        w = _compile_scoring_weights(scoring_weights, strat_weights)
        gap_ratio = _resolve_gap_protection_ratio(params)
        regime_filter = bool(params.get("regime_filter", False))
        try:
            min_adx = float(params.get("min_adx", 0) or 0)
        except (TypeError, ValueError):
            min_adx = 0.0
        if min_adx <= 0:
            min_adx = 0.0
        vix_limit_scaling = bool(params.get("vix_limit_scaling", False))
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        scoring_mode = _resolve_scoring_mode(
            strat.name,
            scoring_type=params.get("scoring_type"),
            type_hint=params.get("type"),
        )
        compiled_strategies.append(
            (strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling, scoring_mode)
        )

    # Diagnostic: verify trail_activation is flowing from JSON -> strategy.params -> engine.
    for strat, _w, params, _gap_ratio, _regime_filter, _base_stop_mult, _min_adx, _vix_limit_scaling, _scoring_mode in compiled_strategies:
        if _DEBUG_TRAIL_ACTIVATION or (isinstance(params.get("version_info"), dict) and params["version_info"].get("status") == "Golden State"):
            print(f"DEBUG: {strat.name} using activation: {params.get('trail_activation')}")

    wealth_present = any(_strategy_role(params) == "wealth" for _s, _w, params, _g, _r, _b, _m, _v, _sm in compiled_strategies)
    income_present = any(_strategy_role(params) == "income" for _s, _w, params, _g, _r, _b, _m, _v, _sm in compiled_strategies)
    confluence_possible = wealth_present and income_present
    if super_signal_only and not confluence_possible:
        return _empty_result(SUPER_SIGNAL_NAME, float(start_cash), strategies[0].params if strategies else {})

    np_isfinite = np.isfinite

    # --- Candidate Generation ---
    for sym, sd in enriched.items():
        df = sd.df
        idx = sd.index

        n = len(idx)
        if n <= MIN_BARS + 1:
            continue

        open_arr = sd.open
        high_arr = sd.high
        low_arr = sd.low
        close_arr = sd.close

        volume_arr = sd.volume
        rsi2_arr = sd.rsi2
        rsi14_arr = sd.rsi14
        adx_arr = sd.adx
        atr14_arr = sd.atr14
        vol_ma20_arr = sd.volma20
        sma20_arr = sd.sma20
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bbwidth
        high52w_arr = sd.high52w
        vix_arr = sd.vix
        vix_rel20_arr = sd.vixrel20
        spy_close_arr = sd.spyclose
        spy_sma20_arr = sd.spysma20
        spy_sma50_arr = sd.spysma50
        spy_sma200_arr = sd.spysma200
        rs_rating_arr = sd.rsrating
        gate_atr_arr = atr14_arr
        if not np.any(np.isfinite(gate_atr_arr) & (gate_atr_arr > 0)):
            hl_range = high_arr - low_arr
            if hl_range.size >= 14:
                gate_atr_arr = pd.Series(hl_range, index=idx).rolling(14).mean().to_numpy()
            else:
                gate_atr_arr = hl_range

        for strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling, scoring_mode in compiled_strategies:
            vix_threshold = _get_param(params, "vix_threshold", 25.0, 10.0, 60.0)
            # Vectorized entry for base GenericStrategy genomes (optimizer hot path)
            if _can_vectorize_entry(strat, params, ai_model):
                warmup = max(int(params.get("warmup_bars", MIN_BARS) or MIN_BARS), MIN_BARS)
                same_day_breakout = scoring_mode == "breakout" and bool(params.get("breakout_same_day_entry", False))
                if same_day_breakout:
                    entry_is = _vectorized_signal_indices(sd, params.get("entry_rules") or [], warmup)
                    if entry_is.size == 0:
                        continue
                    prev_is = entry_is
                    prev_close = close_arr[entry_is - 1]
                else:
                    entry_is = _vectorized_entry_indices(sd, params.get("entry_rules") or [], warmup)
                    if entry_is.size == 0:
                        continue
                    prev_is = entry_is - 1
                    prev_close = close_arr[prev_is]

                open_px, low_px = open_arr[entry_is], low_arr[entry_is]
                low_px = np.where(np.isfinite(low_px), low_px, open_px)
                high_px = high_arr[entry_is]
                high_px = np.where(np.isfinite(high_px), high_px, open_px)

                debug_positions = None
                if not super_signal_only:
                    debug_positions = np.flatnonzero(entry_is == (n - 1))

                valid = np.isfinite(prev_close) & (prev_close > 0) & np.isfinite(open_px) & (open_px > 0)

                # --- Dynamic Universe Gatekeeper ---
                close_val = close_arr[entry_is]
                min_price = float(params.get("min_price", 10.0) or 10.0)
                valid &= np.isfinite(close_val) & (close_val >= min_price)

                vol_val = volume_arr[entry_is]
                min_dollar_vol = float(params.get("min_dollar_vol", 25_000_000) or 25_000_000)
                valid &= np.isfinite(vol_val) & ((close_val * vol_val) >= min_dollar_vol)

                require_above_sma200 = bool(params.get("require_above_sma200", False))
                if require_above_sma200:
                    sma200_val = sma200_arr[entry_is]
                    valid &= np.isfinite(sma200_val) & (close_val >= sma200_val)

                min_natr = float(params.get("min_natr", 0) or 0)
                if min_natr > 0:
                    atr_val = gate_atr_arr[entry_is]
                    natr = np.zeros_like(close_val)
                    natr_mask = (
                        np.isfinite(atr_val)
                        & (atr_val > 0)
                        & np.isfinite(close_val)
                        & (close_val > 0)
                    )
                    natr[natr_mask] = (atr_val[natr_mask] / close_val[natr_mask]) * 100.0
                    valid &= natr_mask & (natr >= min_natr)

                gap_pct_arr = (open_px - prev_close) / prev_close
                valid &= gap_pct_arr >= -0.15

                market_filter_mode = str(params.get("market_filter_mode") or "").lower()
                if market_filter_mode == "traffic_light":
                    spy_close_prev = spy_close_arr[prev_is]
                    spy_sma200_prev = spy_sma200_arr[prev_is]
                    spy_sma20_prev = spy_sma20_arr[prev_is]
                    rs_rating_prev = rs_rating_arr[prev_is]

                    red_mask = (
                        np.isfinite(spy_close_prev)
                        & np.isfinite(spy_sma200_prev)
                        & (spy_close_prev < spy_sma200_prev)
                    )
                    try:
                        bypass_threshold = float(params.get("red_bypass_rs", 100.0))
                    except (TypeError, ValueError):
                        bypass_threshold = 100.0
                    valid &= ~(red_mask & (rs_rating_prev < bypass_threshold))

                    yellow_mask = (
                        np.isfinite(spy_close_prev)
                        & np.isfinite(spy_sma200_prev)
                        & np.isfinite(spy_sma20_prev)
                        & (spy_close_prev >= spy_sma200_prev)
                        & (spy_close_prev < spy_sma20_prev)
                    )
                    yellow_floor = float(params.get("yellow_rs_floor", 92.0) or 92.0)
                    valid &= ~(yellow_mask & (rs_rating_prev < yellow_floor))
                elif regime_filter:
                    valid &= ~(
                        np.isfinite(spy_close_arr[prev_is])
                        & np.isfinite(spy_sma200_arr[prev_is])
                        & (spy_close_arr[prev_is] < spy_sma200_arr[prev_is])
                    )

                if min_adx > 0:
                    adx_prev = adx_arr[prev_is]
                    adx_ok = np.isfinite(adx_prev) & (adx_prev >= min_adx)
                    if debug_positions is not None and debug_positions.size > 0:
                        fail_mask = valid[debug_positions] & ~adx_ok[debug_positions]
                        for pos in debug_positions[fail_mask]:
                            adx_val = adx_prev[pos]
                            adx_str = f"{adx_val:.1f}" if np.isfinite(adx_val) else "nan"
                            print(f"DEBUG: {sym} REJECTED: ADX {adx_str} < {min_adx:.1f}")
                    valid &= adx_ok

                signal_atr = atr14_arr[prev_is]
                valid &= np.isfinite(signal_atr) & (signal_atr > 0)

                # Batched Scoring
                score = _score_candidates_vectorized(
                    rsi2_arr[prev_is],
                    rsi14_arr[prev_is],
                    prev_close,
                    signal_atr,
                    volume_arr[prev_is],
                    vol_ma20_arr[prev_is],
                    sma200_arr[prev_is],
                    cci_arr[prev_is],
                    bb_width_arr[prev_is],
                    high52w_arr[prev_is],
                    w,
                    scoring_mode=scoring_mode,
                )

                # FIX 1: Apply Parity Strategy Multipliers
                score = apply_strategy_score_multipliers(score, params)

                # FIX 2: Vectorized Limit/Stop Entry Logic
                entry_px_arr = open_px.copy()
                limit_ratio = params.get("limit_ratio")
                breakout_max_gap_pct = float(params.get("breakout_max_gap_pct", params.get("max_gap_pct", 0.0)) or 0.0)

                if scoring_mode == "breakout":
                    prev_high = high_arr[entry_is - 1]
                    trigger_px = prev_high * 1.0005
                    filled_mask = np.isfinite(trigger_px) & (high_px > trigger_px)
                    if breakout_max_gap_pct > 0:
                        filled_mask &= open_px <= (trigger_px * (1.0 + breakout_max_gap_pct))
                    limit_ratio_val = None
                    if limit_ratio is not None:
                        try:
                            limit_ratio_val = float(limit_ratio)
                        except (ValueError, TypeError):
                            limit_ratio_val = None
                    if limit_ratio_val is not None:
                        limit_px = trigger_px * limit_ratio_val
                        filled_mask &= open_px <= limit_px
                    entry_px_arr = np.where(open_px > trigger_px, open_px, trigger_px)
                    valid &= filled_mask
                elif limit_ratio is not None:
                    try:
                        limit_ratio_val = float(limit_ratio)
                        target_px = prev_close * limit_ratio_val
                        if vix_limit_scaling:
                            vix_prev = vix_arr[prev_is]
                            scale = np.where(
                                np.isfinite(vix_prev) & (vix_prev > vix_threshold),
                                0.98,
                                1.0,
                            )
                            target_px = target_px * scale
                        # Mean-reversion: buy on pullback to the target.
                        filled_mask = (open_px < target_px) | (low_px < target_px)
                        entry_px_arr = np.where(open_px < target_px, open_px, target_px)

                        # Zero out scores for unfilled trades.
                        score[~filled_mask] = 0.0
                    except (ValueError, TypeError):
                        pass

                # FIX 3: Vectorized ATR Stop Logic
                stop_mult = float(params.get("stop_loss_atr", 3.0))

                # Recalculate ATR% based on actual fill price
                atr_pct = np.zeros_like(entry_px_arr)
                valid_px = entry_px_arr > 0
                atr_pct[valid_px] = (signal_atr[valid_px] / entry_px_arr[valid_px]) * 100.0

                # Gap protection
                gap_pct = (open_px - prev_close) / prev_close
                adj_mult = np.where((gap_pct > -0.03) & (atr_pct > 5.0), stop_mult * 1.2, stop_mult)

                stop_px_arr = entry_px_arr - (signal_atr * adj_mult)

                spy_close_prev = spy_close_arr[prev_is]
                spy_sma200_prev = spy_sma200_arr[prev_is]
                is_bull_regime = (
                    np_isfinite(spy_close_prev)
                    & np_isfinite(spy_sma200_prev)
                    & (spy_close_prev > spy_sma200_prev)
                )
                # --- DECOUPLING FIX: Binary Override ---
                market_filter_mode = str(params.get("market_filter_mode") or "").lower()
                bypass_regime = bool(params.get("bypass_regime_scoring", False))
                strat_min = float(params.get("min_entry_score", MIN_ENTRY_SCORE))

                if market_filter_mode == "traffic_light":
                    dynamic_min_score = np.full_like(score, strat_min)
                elif bypass_regime:
                    dynamic_min_score = np.full_like(score, strat_min)
                else:
                    bull_min = min(strat_min, 100.0)
                    dynamic_min_score = np.where(is_bull_regime, bull_min, strat_min)

                # Final Validity Check
                score_ok = score >= dynamic_min_score
                if debug_positions is not None and debug_positions.size > 0:
                    fail_mask = valid[debug_positions] & ~score_ok[debug_positions]
                    for pos in debug_positions[fail_mask]:
                        score_val = score[pos]
                        min_score_val = dynamic_min_score[pos]
                        score_str = f"{score_val:.1f}" if np.isfinite(score_val) else "nan"
                        min_score_str = f"{min_score_val:.1f}" if np.isfinite(min_score_val) else "nan"
                        print(f"DEBUG: {sym} REJECTED: SCORE {score_str} < {min_score_str}")
                valid &= score_ok
                if not np.any(valid):
                    continue

                cand_k = np.flatnonzero(valid)
                for k in cand_k:
                    day_idx = int(sd.gidx[entry_is[k]]) if sd.gidx.size else date_to_idx.get(idx[entry_is[k]])
                    if day_idx is None:
                        continue
                    candidates_by_day[day_idx].append(
                        _Candidate(
                            sym=sym,
                            entry_px=float(entry_px_arr[k]),
                            stop_px=float(stop_px_arr[k]),
                            score=float(score[k]),
                            strategy_name=strat.name,
                            strategy_obj=strat,
                            entry_i=int(entry_is[k]),
                            signal_i=int(prev_is[k]),
                            ai_prob=0.0,
                            size_scalar=1.0,
                        )
                    )
                continue

            # Fallback path (supports custom strategy.entry)
            debug_last_bar = (not super_signal_only) and (n > 0)
            for i in range(MIN_BARS + 1, n):
                day_idx = date_to_idx.get(idx[i])
                if day_idx is None:
                    continue

                same_day_breakout = scoring_mode == "breakout" and bool(params.get("breakout_same_day_entry", False))
                signal_i = i if same_day_breakout else (i - 1)
                prev_i = signal_i
                debug_reject = debug_last_bar and i == (n - 1)

                prev_close = float(close_arr[i - 1]) if same_day_breakout else float(close_arr[prev_i])
                if not np_isfinite(prev_close) or prev_close <= 0:
                    continue

                open_px = float(open_arr[i])
                if not np_isfinite(open_px) or open_px <= 0:
                    continue

                low_px = float(low_arr[i])
                if not np_isfinite(low_px):
                    low_px = open_px

                high_px = float(high_arr[i])
                if not np_isfinite(high_px):
                    high_px = open_px

                # --- Dynamic Universe Gatekeeper ---
                curr_i = i
                close_val = float(close_arr[curr_i])
                min_price = float(params.get("min_price", 10.0) or 10.0)
                if not np_isfinite(close_val) or close_val < min_price:
                    continue

                sma200_val = float(sma200_arr[curr_i])
                if not np_isfinite(sma200_val) or close_val < sma200_val:
                    continue

                vol_val = float(volume_arr[curr_i])
                min_dollar_vol = float(params.get("min_dollar_vol", 25_000_000) or 25_000_000)
                if not np_isfinite(vol_val) or (close_val * vol_val) < min_dollar_vol:
                    continue

                atr_val = float(gate_atr_arr[curr_i])
                if not np_isfinite(atr_val) or atr_val <= 0:
                    continue
                natr = (atr_val / close_val) * 100.0
                if natr < 2.0:
                    continue

                gap_pct = (open_px - prev_close) / prev_close if prev_close > 0 else 0.0
                if gap_pct < -0.15:
                    continue

                entry_signal = strat.entry(df, prev_i)
                if not entry_signal:
                    continue

                if regime_filter:
                    spy_close_prev = float(spy_close_arr[prev_i])
                    spy_sma200_prev = float(spy_sma200_arr[prev_i])
                    if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                        if spy_close_prev < spy_sma200_prev:
                            continue

                if min_adx > 0:
                    adx_prev = float(adx_arr[prev_i])
                    if not np_isfinite(adx_prev) or adx_prev < min_adx:
                        if debug_reject:
                            adx_str = f"{adx_prev:.1f}" if np_isfinite(adx_prev) else "nan"
                            print(f"DEBUG: {sym} REJECTED: ADX {adx_str} < {min_adx:.1f}")
                        continue

                if gap_ratio > 0 and prev_close > 0:
                    if open_px < prev_close * gap_ratio:
                        continue

                signal_atr = float(atr14_arr[prev_i])
                if not np_isfinite(signal_atr) or signal_atr <= 0:
                    continue

                entry_px = open_px
                limit_ratio = None
                if isinstance(entry_signal, dict):
                    limit_ratio = entry_signal.get("limit_ratio")
                if limit_ratio is None:
                    limit_ratio = params.get("limit_ratio")

                breakout_max_gap_pct = float(params.get("breakout_max_gap_pct", params.get("max_gap_pct", 0.0)) or 0.0)

                if scoring_mode == "breakout":
                    prev_high = float(high_arr[i - 1])
                    if not np_isfinite(prev_high) or prev_high <= 0:
                        continue
                    trigger_px = prev_high * 1.0005
                    if high_px <= trigger_px:
                        continue
                    if breakout_max_gap_pct > 0 and open_px > trigger_px * (1.0 + breakout_max_gap_pct):
                        continue
                    limit_ratio_val = None
                    if limit_ratio is not None:
                        try:
                            limit_ratio_val = float(limit_ratio)
                        except (TypeError, ValueError):
                            limit_ratio_val = None
                    if limit_ratio_val is not None:
                        limit_px = trigger_px * limit_ratio_val
                        if open_px > limit_px:
                            continue
                    entry_px = open_px if open_px > trigger_px else trigger_px
                elif limit_ratio is not None and prev_close > 0:
                    try:
                        limit_ratio_val = float(limit_ratio)
                    except (TypeError, ValueError):
                        limit_ratio_val = None
                    if limit_ratio_val is not None:
                        if vix_limit_scaling:
                            vix_prev = float(vix_arr[prev_i])
                            if np_isfinite(vix_prev) and vix_prev > vix_threshold:
                                limit_ratio_val *= 0.98
                        target_px = prev_close * limit_ratio_val
                        # Mean-reversion: buy on pullback to the target.
                        if open_px < target_px:
                            entry_px = open_px
                        elif low_px < target_px:
                            entry_px = target_px
                        else:
                            continue

                stop_mult = base_stop_mult
                if isinstance(entry_signal, dict) and "stop_loss_atr" in entry_signal:
                    try:
                        stop_mult = float(entry_signal.get("stop_loss_atr") or base_stop_mult)
                    except (TypeError, ValueError):
                        stop_mult = base_stop_mult

                atr_pct_for_stop = (signal_atr / entry_px) * 100.0 if entry_px > 0 else 0.0
                adjusted_mult = stop_mult
                if gap_pct > -0.03 and atr_pct_for_stop > 5.0:
                    adjusted_mult *= 1.2

                stop_price = entry_px - (signal_atr * adjusted_mult)
                if not np_isfinite(stop_price):
                    continue

                # --- FIX: Use Dual-Core Ranking Engine ---
                # Was: score = _score_candidate(...) which forced mean-reversion logic

                # Calculate derived metrics for the new engine
                current_natr = 0.0
                if prev_close > 0:
                    current_natr = (signal_atr / prev_close) * 100.0

                # Pass explicit kwargs to support the Momentum/Breakout logic
                score = calculate_backtest_quality_score(
                    row_or_rsi2=float(rsi2_arr[prev_i]), # Legacy support
                    strategy_name=strat.name,
                    weights=params.get("scoring_weights"),
                    scoring_type=scoring_mode,

                    # --- CRITICAL: MOMENTUM SIGNALS ---
                    rsi14=float(rsi14_arr[prev_i]),
                    bb_width=float(bb_width_arr[prev_i]),
                    close=prev_close,
                    natr=current_natr,
                    # We use 'high' as a proxy for 52w high if the array isn't strictly tracked in this scope,
                    # or pass 0.0 if not available. The engine defaults safely.
                    high_52w=float(high_arr[prev_i]),

                    # --- LEGACY SIGNALS ---
                    atr14=signal_atr,
                    volume=float(volume_arr[prev_i]),
                    vol_ma20=float(vol_ma20_arr[prev_i]),
                    sma200=float(sma200_arr[prev_i]),
                    cci=float(cci_arr[prev_i])
                )
                spy_close_prev = float(spy_close_arr[prev_i])
                spy_sma200_prev = float(spy_sma200_arr[prev_i])
                is_bull_regime = (
                    np_isfinite(spy_close_prev)
                    and np_isfinite(spy_sma200_prev)
                    and spy_close_prev > spy_sma200_prev
                )
                # --- DECOUPLING FIX: Binary Override ---
                market_filter_mode = str(params.get("market_filter_mode") or "").lower()
                bypass_regime = bool(params.get("bypass_regime_scoring", False))
                strat_min = float(params.get("min_entry_score", MIN_ENTRY_SCORE))

                if market_filter_mode == "traffic_light":
                    dynamic_min_score = strat_min
                elif bypass_regime:
                    dynamic_min_score = strat_min
                else:
                    bull_min = min(strat_min, 100.0)
                    dynamic_min_score = bull_min if is_bull_regime else strat_min

                if score < dynamic_min_score:
                    if debug_reject:
                        print(f"DEBUG: {sym} REJECTED: SCORE {score:.1f} < {dynamic_min_score:.1f}")
                    continue

                candidates_by_day[day_idx].append(
                    _Candidate(
                        sym=sym,
                        entry_px=float(entry_px),
                        stop_px=float(stop_price),
                        score=float(score),
                        strategy_name=strat.name,
                        strategy_obj=strat,
                        entry_i=i,
                        signal_i=prev_i,
                        ai_prob=0.0,
                        size_scalar=1.0,
                    )
                )

    # V8 WEALTH-ONLY: VIX scaling disabled to prevent momentum drag.
    # Crash protection is already handled by the SPY SMA200 Regime Filter.
    for day_idx, day_list in enumerate(candidates_by_day):
        if not day_list:
            continue

        filtered: List[_Candidate] = []
        for cand in day_list:
            if cand.is_super_signal:
                filtered.append(cand)
                continue

            sym_data = enriched.get(cand.sym)
            if sym_data is None:
                filtered.append(cand)
                continue

            sig_i = cand.signal_i
            vix_prev = float(sym_data.vix[sig_i]) if 0 <= sig_i < sym_data.vix.size else float("nan")
            if not np_isfinite(vix_prev):
                filtered.append(cand)
                continue

            if vix_prev > 0:
                # V8 WEALTH-ONLY: VIX scaling disabled to prevent momentum drag.
                # crash protection is already handled by the SPY SMA200 Regime Filter.
                pass

            filtered.append(cand)

        candidates_by_day[day_idx] = filtered

    for day_list in candidates_by_day:
        if len(day_list) > 1:
            day_list.sort(key=lambda c: c.score, reverse=True)

    # --- Portfolio Simulation ---
    cash = float(start_cash)
    positions: Dict[str, Dict[str, Any]] = {}
    equity_curve: List[Dict[str, Any]] = []
    trade_pnls: List[float] = []
    trades_list: List[Dict[str, Any]] = []
    ml_data: Optional[List[Dict[str, Any]]] = [] if export_ml_data else None

    main_params = strategies[0].params if strategies else {}
    if not isinstance(main_params, dict):
        main_params = {}
    dyn_max_pos = int(_get_param(main_params, "max_positions", 5, 1, 20))
    dyn_pos_fraction = 1.0 / dyn_max_pos

    MAX_POSITIONS = dyn_max_pos
    REBALANCE_TO_SLOTS = True

    spy_close_by_day = None
    spy_sma200_by_day = None
    spy_data = enriched.get("SPY")
    if spy_data is not None:
        spy_close_by_day = pd.Series(spy_data.close, index=pd.Index(spy_data.index))
        spy_sma200_by_day = pd.Series(spy_data.sma200, index=pd.Index(spy_data.index))
    elif global_data and "SPY" in global_data and not global_data["SPY"].empty:
        spy_df = global_data["SPY"].copy()
        spy_df = spy_df.sort_index()
        spy_df.columns = spy_df.columns.str.lower()
        if "close" in spy_df.columns:
            if "sma200" not in spy_df.columns:
                spy_df["sma200"] = spy_df["close"].rolling(200).mean()
            spy_close_by_day = spy_df["close"]
            spy_sma200_by_day = spy_df["sma200"]

    if spy_close_by_day is not None:
        all_dates_index = pd.Index(all_dates)
        spy_close_by_day = spy_close_by_day.reindex(all_dates_index).ffill().bfill().to_numpy()
        spy_sma200_by_day = spy_sma200_by_day.reindex(all_dates_index).ffill().bfill().to_numpy()

    for day_idx, current_dt in enumerate(all_dates):
        prev_day_idx = day_idx - 1
        is_bull = False
        if spy_close_by_day is not None and prev_day_idx >= 0:
            spy_close_prev = float(spy_close_by_day[prev_day_idx])
            spy_sma200_prev = float(spy_sma200_by_day[prev_day_idx])
            is_bull = np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev) and spy_close_prev > spy_sma200_prev

        if is_bull:
            current_max_pos = 4
            current_pos_frac = 0.25
        else:
            current_max_pos = 7
            current_pos_frac = 0.14

        sector_exposure: Dict[str, float] = {}
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                val = pos["shares"] * pos["entry_price"]
            else:
                loc = int(np.searchsorted(sym_data.index, current_dt))
                if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                    px = float(sym_data.close[loc])
                    if not np_isfinite(px):
                        px = float(pos["entry_price"])
                else:
                    px = float(pos["entry_price"])
                val = float(pos["shares"]) * px
            sec = resolve_sector(sym)
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + float(val)

        daily_candidates = candidates_by_day[day_idx]
        if len(daily_candidates) > 1:
            daily_candidates.sort(key=lambda c: c.score, reverse=True)

        current_sector_equity = float(sum(sector_exposure.values()))
        current_equity = cash + current_sector_equity
        if current_equity <= 0:
            continue

        slot_value = current_equity * current_pos_frac
        open_slots = current_max_pos - len(positions)
        if open_slots < 0:
            open_slots = 0

        is_bull_regime = False
        for cand in daily_candidates:
            sym_data = enriched.get(cand.sym)
            if sym_data is None:
                continue
            sig_i = int(cand.signal_i)
            if 0 <= sig_i < sym_data.spyclose.size and 0 <= sig_i < sym_data.spysma200.size:
                spy_close_prev = float(sym_data.spyclose[sig_i])
                spy_sma200_prev = float(sym_data.spysma200[sig_i])
                if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                    is_bull_regime = spy_close_prev > spy_sma200_prev
                break

        min_score_floor = None
        for cand in daily_candidates:
            strat_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
            if isinstance(strat_params, dict):
                strat_min_score = float(strat_params.get("min_entry_score", MIN_ENTRY_SCORE))
            else:
                strat_min_score = MIN_ENTRY_SCORE
            if is_bull_regime:
                strat_min_score = min(strat_min_score, 100.0)
            if min_score_floor is None or strat_min_score < min_score_floor:
                min_score_floor = strat_min_score

        for cand in daily_candidates:
            if open_slots <= 0:
                break
            if min_score_floor is not None and cand.score < min_score_floor:
                break
            if cand.sym in positions:
                continue
            if cand.entry_px <= 0 or slot_value <= 0:
                continue

            sizing_mode = "slot"
            strat_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
            if isinstance(strat_params, dict):
                strat_min_score = float(strat_params.get("min_entry_score", MIN_ENTRY_SCORE))
                sizing_mode = str(strat_params.get("sizing_mode", "slot")).lower()
            else:
                strat_min_score = MIN_ENTRY_SCORE
            if is_bull_regime:
                strat_min_score = min(strat_min_score, 100.0)
            if cand.score < strat_min_score:
                continue

            if sizing_mode == "risk":
                risk_per_trade = current_equity * 0.02
                risk_per_share = cand.entry_px - cand.stop_px
                if risk_per_share > 0:
                    shares = int(risk_per_trade / risk_per_share)
                else:
                    shares = int((current_equity * current_pos_frac) / cand.entry_px) if cand.entry_px > 0 else 0

                max_shares_by_value = int((current_equity * (current_pos_frac * 1.25)) / cand.entry_px) if cand.entry_px > 0 else 0
                shares = min(shares, max_shares_by_value)
            else:
                target_entry_value = slot_value * cand.size_scalar
                if target_entry_value <= 0:
                    continue
                shares = int(target_entry_value / cand.entry_px)
            if shares <= 0:
                continue

            if isinstance(strat_params, dict) and bool(strat_params.get("vix_position_sizing", False)):
                sym_data = enriched.get(cand.sym)
                if sym_data is not None:
                    sig_i = int(cand.signal_i)
                    if 0 <= sig_i < sym_data.vix.size:
                        vix_prev = float(sym_data.vix[sig_i])
                        vix_threshold = _get_param(strat_params, "vix_threshold", 25.0, 10.0, 60.0)
                        if np.isfinite(vix_prev) and vix_prev > vix_threshold:
                            shares = int(shares * 0.5)
            if shares <= 0:
                continue

            cost = shares * cand.entry_px
            if cost > cash and REBALANCE_TO_SLOTS:
                needed_cash = cost - cash
                if needed_cash > 0:
                    trim_list = []
                    for sym, pos in positions.items():
                        sym_data = enriched.get(sym)
                        if sym_data is None:
                            px = float(pos["entry_price"])
                        else:
                            loc = int(np.searchsorted(sym_data.index, current_dt))
                            if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                                px = float(sym_data.close[loc])
                                if not np_isfinite(px):
                                    px = float(pos["entry_price"])
                            else:
                                px = float(pos["entry_price"])
                        if not np_isfinite(px) or px <= 0:
                            continue
                        pos_val = float(pos.get("shares", 0)) * px
                        excess = pos_val - slot_value
                        if excess > 0:
                            trim_list.append((excess, sym, px))

                    trim_list.sort(key=lambda t: t[0], reverse=True)
                    for excess, sym, px in trim_list:
                        if needed_cash <= 0:
                            break
                        pos = positions.get(sym)
                        if pos is None:
                            continue
                        shares_avail = int(pos.get("shares", 0))
                        if shares_avail <= 0:
                            continue
                        trim_value = min(excess, needed_cash)
                        shares_to_sell = int(trim_value / px)
                        if shares_to_sell <= 0:
                            continue
                        if shares_to_sell > shares_avail:
                            shares_to_sell = shares_avail
                        sale_proceeds = shares_to_sell * px
                        pos["shares"] = shares_avail - shares_to_sell
                        cash += sale_proceeds
                        needed_cash -= sale_proceeds

                        cand_sec = resolve_sector(sym)
                        sector_exposure[cand_sec] = max(0.0, sector_exposure.get(cand_sec, 0.0) - sale_proceeds)
                        current_sector_equity = max(0.0, current_sector_equity - sale_proceeds)
                        if pos["shares"] <= 0:
                            del positions[sym]

                current_equity = cash + current_sector_equity
                slot_value = current_equity * current_pos_frac
                open_slots = current_max_pos - len(positions)
                if open_slots < 0:
                    open_slots = 0

            if cost > cash:
                affordable_shares = int(cash / cand.entry_px) if cand.entry_px > 0 else 0
                if export_ml_data and (affordable_shares <= 0 or cash < cost):
                    shares = 100
                    cost = 0.0
                else:
                    if affordable_shares <= 0:
                        continue
                    shares = min(shares, affordable_shares)
                    cost = shares * cand.entry_px

            cand_sec = resolve_sector(cand.sym)
            projected_exp = (sector_exposure.get(cand_sec, 0.0) + cost) / current_equity if current_equity > 0 else 1.0
            if not export_ml_data and projected_exp > 0.60:
                continue

            cash -= cost
            positions[cand.sym] = {
                "shares": shares,
                "entry_price": cand.entry_px,
                "stop_price": cand.stop_px,
                "entry_i": cand.entry_i,
                "strategy_name": cand.strategy_name,
                "strategy_obj": cand.strategy_obj,
                "signal_i": cand.signal_i,
                "is_super_signal": bool(getattr(cand, "is_super_signal", False)),
            }
            sector_exposure[cand_sec] = sector_exposure.get(cand_sec, 0.0) + cost
            current_sector_equity += cost
            open_slots -= 1

        for sym in list(positions.keys()):
            sym_data = enriched.get(sym)
            if sym_data is None:
                continue

            loc = int(np.searchsorted(sym_data.index, current_dt))
            if not (0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt):
                continue

            pos = positions[sym]
            entry_i = int(pos.get("entry_i", 0) or 0)
            if loc <= entry_i:
                continue

            active_strat = pos.get("strategy_obj")
            if not active_strat:
                continue

            genome = getattr(active_strat, "params", getattr(active_strat, "genome", {})) or {}

            # --- PARTIAL PROFIT LOGIC ---
            # Qullamaggie Rule: Sell 1/3 to 1/2 into strength (3-5 days or 15-20% move)
            # and move stop to Breakeven.
            strat_params = getattr(active_strat, "params", {}) or {}
            partial_target = float(strat_params.get("partial_profit_target", 0.0))
            partial_scale = float(strat_params.get("partial_exit_scale", 0.0))

            if partial_target > 1.0 and partial_scale > 0 and not pos.get("partial_taken", False):
                # Check if High hit the target
                high_px = float(sym_data.high[loc])
                target_price = float(pos["entry_price"]) * partial_target

                if high_px >= target_price:
                    # Execute Partial Sale
                    sell_shares = int(pos["shares"] * partial_scale)
                    if sell_shares > 0:
                        exit_px = target_price
                        pnl = (exit_px - pos["entry_price"]) * sell_shares
                        pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100.0

                        cash += sell_shares * exit_px
                        trade_pnls.append(float(pnl))

                        trades_list.append(
                            {
                                "Symbol": sym,
                                "Entry Date": str(sym_data.df.index[entry_i].date()),
                                "Exit Date": str(pd.Timestamp(current_dt).date()),
                                "Entry": pos["entry_price"],
                                "Exit": exit_px,
                                "PnL": pnl,
                                "Return%": pct,
                                "Strategy": pos.get("strategy_name", strategy_label) + " (Partial)",
                                "is_super_signal": bool(pos.get("is_super_signal", False)),
                            }
                        )

                        # Update Position
                        pos["shares"] -= sell_shares
                        pos["partial_taken"] = True

                        # Move Stop to Breakeven (Risk Free Ride)
                        pos["stop_price"] = max(float(pos["stop_price"]), float(pos["entry_price"]))

            try:
                used_generic_exit = False
                if bool(pos.get("is_super_signal", False)):
                    should_exit, effective_stop, target_px = _generic_exit_decision(
                        _apply_super_signal_overrides(genome),
                        sym_data,
                        loc,
                        entry_i,
                        float(pos["entry_price"]),
                        float(pos["stop_price"]),
                    )
                    used_generic_exit = True
                elif isinstance(active_strat, GenericStrategy) and active_strat.__class__.exit is GenericStrategy.exit:
                    should_exit, effective_stop, target_px = _generic_exit_decision(
                        genome,
                        sym_data,
                        loc,
                        entry_i,
                        float(pos["entry_price"]),
                        float(pos["stop_price"]),
                    )
                    used_generic_exit = True
                else:
                    exit_result = active_strat.exit(sym_data.df, loc, entry_i, pos["entry_price"], pos["stop_price"])
                    updated_stop = None
                    target_px = None
                    if isinstance(exit_result, tuple):
                        should_exit = bool(exit_result[0])
                        if len(exit_result) > 1:
                            updated_stop = exit_result[1]
                        if len(exit_result) > 2:
                            target_px = exit_result[2]
                    else:
                        should_exit = bool(exit_result)
                    if updated_stop is not None and np_isfinite(updated_stop):
                        updated_stop = float(updated_stop)
                        pos["stop_price"] = max(float(pos["stop_price"]), updated_stop)
                    effective_stop = float(pos["stop_price"])

                # --- FIX: Explicitly check JSON Exit Rules (e.g. SMA10) ---
                if used_generic_exit and not should_exit:
                    exit_rules = genome.get("exit_rules", [])
                    for rule in exit_rules:
                        if rule.get("type") != "rule":
                            continue
                        col = rule.get("col")
                        ref = rule.get("ref")
                        op = rule.get("op")
                        if not col or not ref or not op:
                            continue

                        val_col = getattr(sym_data, col, None)
                        val_ref = getattr(sym_data, ref, None)
                        if val_col is None or val_ref is None:
                            continue

                        curr_val = float(val_col[loc])
                        ref_val = float(val_ref[loc])
                        if not np_isfinite(curr_val) or not np_isfinite(ref_val):
                            continue

                        if op == "<" and curr_val < ref_val:
                            should_exit = True
                        elif op == ">" and curr_val > ref_val:
                            should_exit = True
                        elif op == "<=" and curr_val <= ref_val:
                            should_exit = True
                        elif op == ">=" and curr_val >= ref_val:
                            should_exit = True

                        if should_exit:
                            open_px = float(sym_data.open[loc])
                            if not np_isfinite(open_px):
                                open_px = float(sym_data.close[loc])
                            effective_stop = open_px
                            break
            except Exception:
                should_exit = False
                effective_stop = float(pos["stop_price"])
                target_px = None

            if not should_exit:
                continue

            open_px = float(sym_data.open[loc])
            low_px = float(sym_data.low[loc])
            high_px = float(sym_data.high[loc])
            close_px = float(sym_data.close[loc])

            if not np_isfinite(low_px):
                low_px = close_px
            if not np_isfinite(high_px):
                high_px = close_px

            exit_px = close_px
            if np_isfinite(effective_stop) and low_px < effective_stop:
                exit_px = effective_stop if open_px >= effective_stop else open_px
            elif target_px is not None and np_isfinite(target_px) and high_px >= target_px:
                exit_px = float(target_px)

            if exit_px < pos["entry_price"] * 0.5:
                exit_px = pos["entry_price"] * 0.5

            pnl = (exit_px - pos["entry_price"]) * pos["shares"]
            pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100.0 if pos["entry_price"] else 0.0

            cash += pos["shares"] * exit_px
            trade_pnls.append(float(pnl))

            if export_ml_data and ml_data is not None:
                try:
                    entry_feat_i = int(pos.get("signal_i") or pos.get("entry_i") or 0)
                    entry_feat_i = max(0, min(entry_feat_i, len(sym_data.index) - 1))

                    close_entry = float(sym_data.close[entry_feat_i])
                    atr14_entry = float(sym_data.atr14[entry_feat_i])
                    sma50_entry = float(sym_data.sma50[entry_feat_i])
                    sma200_entry = float(sym_data.sma200[entry_feat_i])
                    vol_entry = float(sym_data.volume[entry_feat_i])
                    vol_ma20_entry = float(sym_data.volma20[entry_feat_i])
                    vix_entry = float(sym_data.vix[entry_feat_i])
                    vix_rel20_entry = float(sym_data.vixrel20[entry_feat_i])
                    rs_ratio_entry = float(sym_data.rsratio[entry_feat_i])
                    rs_trend_entry = float(sym_data.rstrend[entry_feat_i])
                    rs_mom20_entry = float(sym_data.rsmom20[entry_feat_i])
                    spy_regime_entry = float(sym_data.spyregime[entry_feat_i])

                    dist50 = (close_entry - sma50_entry) / close_entry if close_entry else 0.0
                    dist200 = (close_entry - sma200_entry) / close_entry if close_entry else 0.0
                    atr_pct = atr14_entry / close_entry if close_entry else 0.0
                    vol_rel = vol_entry / (vol_ma20_entry + 1.0)

                    ml_data.append(
                        {
                            "rsi2": float(sym_data.rsi2[entry_feat_i]),
                            "adx": float(sym_data.adx[entry_feat_i]),
                            "atr14": atr14_entry,
                            "atr_pct": float(atr_pct),
                            "vol_rel": float(vol_rel),
                            "dist_to_sma50": float(dist50),
                            "dist_to_sma200": float(dist200),
                            "dist_sma50": float(dist50),
                            "dist_sma200": float(dist200),
                            "vix": vix_entry,
                            "vix_rel20": vix_rel20_entry,
                            "rs_ratio": rs_ratio_entry,
                            "rs_trend": rs_trend_entry,
                            "rs_mom20": rs_mom20_entry,
                            "spy_regime": spy_regime_entry,
                            "profit_pct": float(pct),
                            "outcome": 1 if pct > 0 else 0,
                        }
                    )
                except Exception:
                    pass

            trades_list.append(
                {
                    "Symbol": sym,
                    "Entry Date": str(sym_data.df.index[entry_i].date()),
                    "Exit Date": str(pd.Timestamp(current_dt).date()),
                    "Entry": pos["entry_price"],
                    "Exit": exit_px,
                    "PnL": pnl,
                    "Return%": pct,
                    "Strategy": pos.get("strategy_name", strategy_label),
                    "is_super_signal": bool(pos.get("is_super_signal", False)),
                }
            )
            del positions[sym]

        equity = cash
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                equity += pos["shares"] * pos["entry_price"]
                continue
            loc = int(np.searchsorted(sym_data.index, current_dt))
            if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                px = float(sym_data.close[loc])
                if not np_isfinite(px):
                    px = float(pos["entry_price"])
            else:
                px = float(pos["entry_price"])
            equity += float(pos["shares"]) * px

        equity_curve.append({"Date": pd.Timestamp(current_dt), "Equity": float(equity)})

    trades = len(trade_pnls)
    wins = sum(1 for t in trade_pnls if t > 0)
    hit_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    avg_profit = (sum(t.get("Return%", 0.0) for t in trades_list) / len(trades_list)) if trades_list else 0.0

    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty:
        eq_df = pd.DataFrame({"Equity": [float(start_cash)]}, index=[pd.Timestamp.now()])

    days = int((eq_df.index[-1] - eq_df.index[0]).days)
    years = days / 365.25 if days > 0 else 1.0
    cagr = ((eq_df["Equity"].iloc[-1] / float(start_cash)) ** (1.0 / years)) - 1.0 if start_cash else 0.0

    rolling_max = eq_df["Equity"].cummax()
    drawdowns = (eq_df["Equity"] - rolling_max) / rolling_max
    max_dd = float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

    if export_ml_data and ml_data:
        try:
            out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_training_data.csv"))
            pd.DataFrame(ml_data).to_csv(out_path, index=False)
            print(f"🧠 ML Data Exported: {len(ml_data)} samples saved to {out_path}")
        except Exception as e:
            print(f"⚠️ ML Data Export failed: {e}")

    return {
        "strategy": strategy_label,
        "final_value": float(eq_df["Equity"].iloc[-1]) if not eq_df.empty else float(start_cash),
        "total_trades": trades,
        "hit_rate": hit_rate,
        "cagr": float(cagr),
        "avg_profit_pct": float(avg_profit),
        "max_drawdown_pct": float(max_dd),
        "equity_curve": eq_df,
        "trades_list": trades_list,
        "params": strategies[0].params if strategies else {},
        "Score": float(cagr) * 1000.0,
    }


def run_compare(
    strategy_names,
    data_dict,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    use_parallel=True,
    max_workers=8,
    global_data=None,
):
    gen_strategies = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "generated_strategies.json")), "r") as f:
            for g in json.load(f):
                gen_strategies[g["name"]] = g
    except Exception:
        pass

    strategies = load_strategies([gen_strategies[name] for name in strategy_names if name in gen_strategies])
    if not strategies:
        return pd.DataFrame()

    results = []
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat
                for strat in strategies
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
    else:
        for strat in strategies:
            try:
                results.append(run_backtest(strat, data_dict, symbol_universe, start_cash, start_date, global_data))
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["equity_curve", "trades_list"]} for r in results])
    return df.sort_values("Score", ascending=False)
