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


def inject_market_rs_rank(enriched, all_dates, lookbacks=(63, 126, 189, 252),
                          weights=(0.40, 0.20, 0.20, 0.20),
                          min_history=252, min_names=100):
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
        valid_mask = (sd.gidx >= 0) & (sd.gidx < n_days)
        close_mat[sd.gidx[valid_mask], j] = sd.close[valid_mask].astype(np.float32)

    tmp = np.zeros((n_days, n_syms), dtype=np.float32)
    
    valid_hist = np.isfinite(close_mat)
    cum = np.cumsum(valid_hist, axis=0)
    enough = cum >= min_history

    for lb, w in zip(lookbacks, weights):
        shifted = np.roll(close_mat, lb, axis=0)
        shifted[:lb, :] = np.nan
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = (close_mat / shifted) - 1.0
        
        tmp += w * ret

    score = np.where(enough & np.isfinite(tmp), tmp, np.nan)
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
            order = np.argsort(vals, kind="quicksort")
            ranks = np.empty_like(order, dtype=np.int32)
            ranks[order] = np.arange(k, dtype=np.int32)

            pct = (ranks / (k - 1)) if k > 1 else np.zeros(k, dtype=np.float32)
            rs_rating[start + i, m] = 1.0 + 98.0 * pct

    for j, sym in enumerate(syms):
        sd = enriched[sym]
        valid_indices = (sd.gidx >= 0) & (sd.gidx < n_days)
        if np.any(valid_indices):
            sd.rsrating[valid_indices] = rs_rating[sd.gidx[valid_indices], j]


def simulate_breakout_fill(open_px: float, high_px: float, trigger_px: float, limit_px: float | None = None) -> float | None:
    """
    V2 LOGIC: Simulates a Buy Stop Limit order.
    Executes ONLY if High > Trigger. Fills at MAX(Open, Trigger).
    """
    if high_px < trigger_px:
        return None # No breakout occurred today

    if limit_px is not None and open_px > limit_px:
        return None # Gapped over limit

    # Fill at the breakout price (or open if it gapped up but stayed under limit)
    fill_px = max(open_px, trigger_px)
    return fill_px * 1.001 # Slippage


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

        # V2 UPGRADE: ADR % Calculation
        # (High - Low) / Close, 20-day MA * 100
        df["hl_range"] = df["high"] - df["low"]
        df["adr_pct"] = (df["hl_range"] / df["close"]).rolling(20).mean() * 100.0

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
        
        # New for V2: Previous High for Stop-Buy Trigger
        df["prev_high"] = df["high"].shift(1)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        g2 = (delta.where(delta > 0, 0)).rolling(2).mean()
        l2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = g2 / l2.replace(0, np.nan)
        df["rsi2"] = 100 - (100 / (1 + rs2))

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
        # V2 SCORING: Reward High RSI + VCP
        # This is not falling knives; it rewards trends
        if not np_isfinite(rsi14):
            rsi14 = 50.0
        score = (rsi14 * rsi_factor)
        
        if np_isfinite(bb_width) and bb_width < _VCP_BB_WIDTH_THRESH:
            score += vcp_bonus
        if np_isfinite(natr) and natr > 3.0: # Reward volatility
            score += vol_bonus
        if np_isfinite(close_px) and np_isfinite(high_52w) and high_52w > 0:
            if close_px >= (high_52w * 0.85):
                score += trend_bonus
    else:
        # Wealth/Reversion logic: prize LOW RSI2
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
    adr_pct: np.ndarray   # V2: New ADR Field
    prev_high: np.ndarray # V2: New Prev High Field


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
    return genome


@dataclass(slots=True)
class PreparedBacktestData:
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
                volma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma10=_get_np_col(df, "sma10", np.nan, length=n),
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
                rsrating=np.zeros(n, dtype=np.float64),
                adr_pct=_get_np_col(df, "adr_pct", 0.0, length=n),
                prev_high=_get_np_col(df, "prev_high", 0.0, length=n),
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

    # --- INJECT RS RATINGS (VECTORIZED) ---
    inject_market_rs_rank(enriched, all_dates)

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

    # Use centralized data preparation which includes RS injection
    prepared = prepare_backtest_data(data_dict, symbol_universe, start_date, global_data)
    enriched = prepared.enriched
    all_dates = prepared.all_dates

    if not enriched:
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

    np_isfinite = np.isfinite

    for sym, sd in enriched.items():
        df = sd.df
        idx = sd.index
        open_arr = sd.open
        low_arr = sd.low
        close_arr = sd.close
        high_arr = sd.high # V2: Needed for stop buy check
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
        bbwidth_arr = sd.bbwidth
        natr_arr = sd.natr
        high52w_arr = sd.high52w
        vix_arr = sd.vix
        spy_close_arr = sd.spyclose
        spy_sma20_arr = sd.spysma20
        spy_sma50_arr = sd.spysma50
        spy_sma200_arr = sd.spysma200
        rs_rating_arr = sd.rsrating
        adr_pct_arr = sd.adr_pct
        prev_high_arr = sd.prev_high
        gidx = sd.gidx

        n_bars = len(idx)
        if n_bars < 2:
            continue

        valid_mask = np.isfinite(close_arr) & np.isfinite(open_arr)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 2:
            continue

        for i in range(1, len(valid_indices)):
            curr_i = valid_indices[i]
            prev_i = valid_indices[i - 1]
            day_idx = gidx[curr_i]

            if day_idx < 0 or day_idx >= len(all_dates):
                continue

            # --- TRAFFIC LIGHT PRE-CALCULATION (NO LOOKAHEAD) ---
            spy_c = float(spy_close_arr[prev_i]) if np_isfinite(spy_close_arr[prev_i]) else 0.0
            spy_20 = float(spy_sma20_arr[prev_i]) if np_isfinite(spy_sma20_arr[prev_i]) else 0.0
            spy_50 = float(spy_sma50_arr[prev_i]) if np_isfinite(spy_sma50_arr[prev_i]) else 0.0
            spy_200 = float(spy_sma200_arr[prev_i]) if np_isfinite(spy_sma200_arr[prev_i]) else 0.0
            rs_rating = float(rs_rating_arr[prev_i]) if np_isfinite(rs_rating_arr[prev_i]) else 0.0

            market_state = "GREEN" 
            if spy_c > 0 and spy_200 > 0:
                if spy_c < spy_200:
                    market_state = "RED"
                elif spy_c < spy_50:
                    if spy_c < spy_20:
                        market_state = "YELLOW"
                    else:
                        market_state = "GREEN"
                else:
                    if spy_c < spy_20:
                        market_state = "YELLOW"
                    else:
                        market_state = "GREEN"

            for strat_idx, (
                strat,
                w,
                params,
                gap_ratio,
                regime_filter,
                base_stop_mult,
                min_adx,
                vix_limit_scaling,
                scoring_mode,
            ) in enumerate(compiled_strategies):
                market_filter_mode = str(params.get("market_filter_mode") or "").lower()

                # --- V2 UPGRADE: Elite Bypass Traffic Light ---
                if market_filter_mode == "traffic_light":
                    # Bypass Red Light only if Elite RS
                    if market_state == "RED":
                        bypass_threshold = float(params.get("red_bypass_rs", 100.0))
                        if rs_rating < bypass_threshold:
                            continue
                    elif market_state == "YELLOW":
                        yellow_floor = float(params.get("yellow_rs_floor", 92.0))
                        if rs_rating < yellow_floor:
                            continue
                elif regime_filter:
                    if spy_c > 0 and spy_200 > 0 and spy_c < spy_200:
                        continue

                if min_adx > 0:
                    val_adx = float(adx_arr[prev_i])
                    if not np_isfinite(val_adx) or val_adx < min_adx:
                        continue

                score = _score_row_dual_core(
                    rsi2_arr[prev_i],
                    rsi14_arr[prev_i],
                    bbwidth_arr[prev_i],
                    natr_arr[prev_i],
                    close_arr[prev_i],
                    high52w_arr[prev_i],
                    {
                        "rsi_factor": w.rsi_factor,
                        "vcp_bonus": w.vcp_bonus,
                        "vol_bonus": w.vol_bonus,
                        "trend_bonus": w.trend_bonus,
                    },
                    scoring_mode,
                )

                strat_min_score = float(params.get("min_entry_score", MIN_ENTRY_SCORE))
                if score >= strat_min_score:
                    # --- V2 UPGRADE: STOP-BUY EXECUTION ---
                    prev_high = float(prev_high_arr[curr_i]) # Actually yesterday's high
                    open_px = float(open_arr[curr_i])
                    high_px = float(high_arr[curr_i])
                    
                    entry_px = open_px # Default for Wealth Strategy

                    if scoring_mode == "breakout":
                        # V2 Logic: Must break yesterday's high
                        trigger = prev_high * 1.0005 # 0.05% above high
                        
                        if high_px < trigger:
                            continue # Failed breakout
                        
                        # Calculate fill
                        entry_px = max(open_px, trigger)
                    else:
                        # Legacy Limit Logic for Wealth
                        prev_close = float(close_arr[prev_i])
                        limit_ratio = params.get("limit_ratio")
                        if limit_ratio is not None and prev_close > 0:
                            try:
                                limit_ratio_val = float(limit_ratio)
                            except (TypeError, ValueError):
                                limit_ratio_val = 0.98
                            
                            target_px = prev_close * limit_ratio_val
                            if open_px < target_px:
                                entry_px = open_px
                            elif open_px > target_px * 1.05:
                                continue
                            else:
                                entry_px = open_px
                    
                    atr = float(atr14_arr[prev_i])
                    stop_px = calculate_stop_price(entry_px, atr, base_stop_mult)

                    ai_prob = 0.0
                    # ... AI logic remains the same ...

                    candidates_by_day[day_idx].append(
                        _Candidate(
                            sym=sym,
                            entry_px=entry_px,
                            stop_px=stop_px,
                            score=score,
                            strategy_name=strat.name,
                            strategy_obj=strat,
                            entry_i=curr_i,
                            signal_i=prev_i,
                            is_super_signal=super_signal_only,
                            ai_prob=ai_prob,
                        )
                    )

    # Execution Loop
    portfolio = {s.name: {"cash": float(start_cash), "positions": {}} for s in strategies}
    if super_signal_only:
        portfolio[SUPER_SIGNAL_NAME] = {"cash": float(start_cash), "positions": {}}

    final_results = []

    for strat in strategies:
        strat_key = SUPER_SIGNAL_NAME if super_signal_only else strat.name
        port = portfolio[strat_key]
        cash = port["cash"]
        positions = port["positions"]
        trades_list = []
        equity_curve = []

        params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        max_pos = int(params.get("max_positions", 10) or 10)
        risk_per_trade = float(params.get("risk_per_trade", 0.02) or 0.02)

        for day_idx, candidates in enumerate(candidates_by_day):
            current_dt = all_dates[day_idx]

            # 1. Manage Existing Positions
            to_remove = []
            for sym, pos in positions.items():
                if sym not in enriched:
                    to_remove.append(sym)
                    continue
                sym_data = enriched[sym]
                
                loc_range = np.searchsorted(sym_data.gidx, [day_idx, day_idx + 1])
                if loc_range[0] == loc_range[1]:
                    continue
                
                loc = loc_range[0]
                
                should_exit, effective_stop, target_px = _generic_exit_decision(
                    strat, sym_data, loc, pos["entry_i"], pos["entry_price"], pos["stop_price"]
                )
                
                pos["stop_price"] = effective_stop

                if should_exit:
                    open_px = float(sym_data.open[loc])
                    low_px = float(sym_data.low[loc])
                    high_px = float(sym_data.high[loc])
                    
                    exit_px = open_px
                    if low_px < effective_stop:
                        if open_px < effective_stop:
                            exit_px = open_px
                        else:
                            exit_px = effective_stop
                    elif target_px and high_px >= target_px:
                        exit_px = target_px
                    else:
                        exit_px = float(sym_data.close[loc])

                    shares = pos["shares"]
                    pnl = (exit_px - pos["entry_price"]) * shares
                    pnl_pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100.0
                    
                    cash += (shares * exit_px)
                    trades_list.append({
                        "Symbol": sym,
                        "Entry Date": str(sym_data.df.index[pos["entry_i"]].date()),
                        "Exit Date": str(pd.Timestamp(current_dt).date()),
                        "Entry": pos["entry_price"],
                        "Exit": exit_px,
                        "Shares": shares,
                        "PnL": pnl,
                        "Return %": pnl_pct,
                        "Strategy": strat_key
                    })
                    to_remove.append(sym)
            
            for sym in to_remove:
                del positions[sym]

            # 2. Enter New Positions
            day_candidates = [c for c in candidates if c.strategy_name == strat.name or super_signal_only]
            day_candidates.sort(key=lambda x: (x.ai_prob, x.score), reverse=True)

            for cand in day_candidates:
                if len(positions) >= max_pos:
                    break
                
                if cand.sym in positions:
                    continue
                
                equity = cash + sum(p["shares"] * p["entry_price"] for p in positions.values())
                risk_amt = equity * risk_per_trade
                dist = max(0.01, cand.entry_px - cand.stop_px)
                shares = int(risk_amt / dist)
                
                max_capital = equity * 0.25
                if shares * cand.entry_px > max_capital:
                    shares = int(max_capital / cand.entry_px)
                
                if shares < 1:
                    continue
                
                cost = shares * cand.entry_px
                if cash >= cost:
                    cash -= cost
                    positions[cand.sym] = {
                        "entry_price": cand.entry_px,
                        "stop_price": cand.stop_px,
                        "shares": shares,
                        "entry_i": cand.entry_i,
                        "adds": 0 
                    }

            # 3. Record Curve
            curr_equity = cash
            for sym, pos in positions.items():
                if sym in enriched:
                    loc_range = np.searchsorted(enriched[sym].gidx, [day_idx, day_idx+1])
                    if loc_range[0] < loc_range[1]:
                        curr_equity += pos["shares"] * enriched[sym].close[loc_range[0]]
                    else:
                        curr_equity += pos["shares"] * pos["entry_price"]
                else:
                    curr_equity += pos["shares"] * pos["entry_price"]
            
            equity_curve.append({"Date": current_dt, "Equity": curr_equity})

        final_val = equity_curve[-1]["Equity"] if equity_curve else start_cash
        
        df_trades = pd.DataFrame(trades_list)
        win_rate = 0.0
        profit_factor = 0.0
        if not df_trades.empty:
            wins = df_trades[df_trades["PnL"] > 0]
            losses = df_trades[df_trades["PnL"] <= 0]
            win_rate = (len(wins) / len(df_trades)) * 100.0
            if abs(losses["PnL"].sum()) > 0:
                profit_factor = wins["PnL"].sum() / abs(losses["PnL"].sum())
            else:
                profit_factor = 10.0

        res = _empty_result(strat_key, start_cash, params)
        res.update({
            "final_value": final_val,
            "cagr": ((final_val / start_cash) ** (365 / len(all_dates)) - 1) if len(all_dates) > 365 else 0.0,
            "total_trades": len(trades_list),
            "hit_rate": win_rate,
            "profit_factor": profit_factor,
            "equity_curve": equity_curve,
            "trades_list": trades_list
        })
        final_results.append(res)

    if len(final_results) == 1:
        return final_results[0]
    return final_results

# Backwards compatibility if run_backtest is called
run_backtest = _legacy_run_backtest

def calculate_stop_price(entry_price, atr, multiplier):
    return entry_price - (atr * multiplier)
