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
_SNIPER_BB_WIDTH_THRESH = 0.17

# Diagnostics: enable to print every strategy's trail activation on each run.
_DEBUG_TRAIL_ACTIVATION = os.environ.get("APEX_DEBUG_TRAIL_ACTIVATION", "").strip() not in ("", "0", "false", "False")


def get_sector(symbol: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"


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

        df["highest20"] = df["high"].rolling(20).max()
        df["highest20_1"] = df["highest20"].shift(1)
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
            df["spy_sma200"] = spy_aligned.rolling(200).mean()
            df["rs_mom20"] = (df["rs_ratio"] / df["rs_ratio"].shift(20)) - 1.0
            df["spy_regime"] = (df["spy_close"] > df["spy_sma200"]).astype(int)
        else:
            df["rs_ratio"] = 1.0
            df["rs_trend"] = 0.0
            df["spy_close"] = np.nan
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
    sniper_bonus: float


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
        sniper_bonus=float(merged.get("sniper_bonus", 0.0) or 0.0),
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

    sniper_ok = bool(np_isfinite(cci) and np_isfinite(bb_width) and cci < 0 and bb_width > _SNIPER_BB_WIDTH_THRESH)

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

    if sniper_ok:
        score += w.sniper_bonus

    return max(0.0, float(score))


def calculate_backtest_quality_score(
    row_or_rsi2: Any,
    strategy_name: str = "",
    weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> float:
    """
    Compatibility wrapper.

    - Legacy call sites pass a pandas Series/dict row as the first arg.
    - Performance call sites can pass scalars: `calculate_backtest_quality_score(rsi2, close=..., atr14=..., ...)`.
    """
    _ = strategy_name
    w = _compile_scoring_weights(weights, None)

    if isinstance(row_or_rsi2, (pd.Series, dict)):
        row = row_or_rsi2
        rsi2 = float(row.get("rsi2", 50) or 50)
        close_px = float(row.get("close", 0) or 0)
        atr14 = float(row.get("atr14", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        vol_ma20 = float(row.get("vol_ma20", 1) or 1)
        sma200 = float(row.get("sma200", np.nan))
        cci = float(row.get("cci", 0) or 0)
        bb_width = float(row.get("bb_width", 0) or 0)
        return _score_candidate(rsi2, close_px, atr14, volume, vol_ma20, sma200, cci, bb_width, w)

    rsi2 = float(row_or_rsi2)
    close_px = float(kwargs.get("close", 0.0) or 0.0)
    atr14 = float(kwargs.get("atr14", 0.0) or 0.0)
    volume = float(kwargs.get("volume", 0.0) or 0.0)
    vol_ma20 = float(kwargs.get("vol_ma20", 1.0) or 1.0)
    sma200 = float(kwargs.get("sma200", np.nan))
    cci = float(kwargs.get("cci", 0.0) or 0.0)
    bb_width = float(kwargs.get("bb_width", 0.0) or 0.0)
    return _score_candidate(rsi2, close_px, atr14, volume, vol_ma20, sma200, cci, bb_width, w)


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
    stoch_k: np.ndarray
    atr14: np.ndarray
    vol_ma20: np.ndarray
    sma20: np.ndarray
    sma50: np.ndarray
    sma200: np.ndarray
    cci: np.ndarray
    bb_width: np.ndarray
    bb_lower: np.ndarray
    bb_upper: np.ndarray
    rs_ratio: np.ndarray
    rs_trend: np.ndarray
    rs_mom20: np.ndarray
    vix: np.ndarray
    vix_sma20: np.ndarray
    vix_rel20: np.ndarray
    spy_close: np.ndarray
    spy_sma200: np.ndarray
    spy_regime: np.ndarray


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
                stoch_k=_get_np_col(df, "stoch_k", 0.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                vol_ma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma20=_get_np_col(df, "sma20", np.nan, length=n),
                sma50=_get_np_col(df, "sma50", np.nan, length=n),
                sma200=_get_np_col(df, "sma200", np.nan, length=n),
                cci=_get_np_col(df, "cci", 0.0, length=n),
                bb_width=_get_np_col(df, "bb_width", 0.0, length=n),
                bb_lower=_get_np_col(df, "bb_lower", np.nan, length=n),
                bb_upper=_get_np_col(df, "bb_upper", np.nan, length=n),
                rs_ratio=_get_np_col(df, "rs_ratio", 1.0, length=n),
                rs_trend=_get_np_col(df, "rs_trend", 0.0, length=n),
                rs_mom20=_get_np_col(df, "rs_mom20", 0.0, length=n),
                vix=_get_np_col(df, "vix", 20.0, length=n),
                vix_sma20=_get_np_col(df, "vix_sma20", 20.0, length=n),
                vix_rel20=_get_np_col(df, "vix_rel20", 0.0, length=n),
                spy_close=_get_np_col(df, "spy_close", np.nan, length=n),
                spy_sma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
                spy_regime=_get_np_col(df, "spy_regime", 0.0, length=n),
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
                stoch_k=_get_np_col(df, "stoch_k", 0.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                vol_ma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma20=_get_np_col(df, "sma20", np.nan, length=n),
                sma50=_get_np_col(df, "sma50", np.nan, length=n),
                sma200=_get_np_col(df, "sma200", np.nan, length=n),
                cci=_get_np_col(df, "cci", 0.0, length=n),
                bb_width=_get_np_col(df, "bb_width", 0.0, length=n),
                bb_lower=_get_np_col(df, "bb_lower", np.nan, length=n),
                bb_upper=_get_np_col(df, "bb_upper", np.nan, length=n),
                rs_ratio=_get_np_col(df, "rs_ratio", 1.0, length=n),
                rs_trend=_get_np_col(df, "rs_trend", 0.0, length=n),
                rs_mom20=_get_np_col(df, "rs_mom20", 0.0, length=n),
                vix=_get_np_col(df, "vix", 20.0, length=n),
                vix_sma20=_get_np_col(df, "vix_sma20", 20.0, length=n),
                vix_rel20=_get_np_col(df, "vix_rel20", 0.0, length=n),
                spy_close=_get_np_col(df, "spy_close", np.nan, length=n),
                spy_sma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
                spy_regime=_get_np_col(df, "spy_regime", 0.0, length=n),
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

    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]

    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float, float, bool]] = []
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
        compiled_strategies.append((strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling))

    np_isfinite = np.isfinite

    for sym, sd in enriched.items():
        df = sd.df
        idx = sd.index
        open_arr = sd.open
        low_arr = sd.low
        close_arr = sd.close
        volume_arr = sd.volume
        rsi2_arr = sd.rsi2
        adx_arr = sd.adx
        atr14_arr = sd.atr14
        vol_ma20_arr = sd.vol_ma20
        sma20_arr = sd.sma20
        sma50_arr = sd.sma50
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bb_width
        rs_ratio_arr = sd.rs_ratio
        rs_trend_arr = sd.rs_trend
        rs_mom20_arr = sd.rs_mom20
        vix_arr = sd.vix
        vix_rel20_arr = sd.vix_rel20
        spy_close_arr = sd.spy_close
        spy_sma200_arr = sd.spy_sma200
        spy_regime_arr = sd.spy_regime

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

            for strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling in compiled_strategies:
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

                score = _score_candidate(
                    float(rsi2_arr[prev_i]),
                    prev_close,
                    signal_atr,
                    float(volume_arr[prev_i]),
                    float(vol_ma20_arr[prev_i]),
                    float(sma200_arr[prev_i]),
                    float(cci_arr[prev_i]),
                    float(bb_width_arr[prev_i]),
                    w,
                )
                score = apply_strategy_score_multipliers(score, params)
                if score < MIN_ENTRY_SCORE:
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
            if cand.score < MIN_ENTRY_SCORE:
                break
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
                    vol_ma20_entry = float(sym_data.vol_ma20[entry_feat_i])
                    vix_entry = float(sym_data.vix[entry_feat_i])
                    vix_rel20_entry = float(sym_data.vix_rel20[entry_feat_i])
                    rs_ratio_entry = float(sym_data.rs_ratio[entry_feat_i])
                    rs_trend_entry = float(sym_data.rs_trend[entry_feat_i])
                    rs_mom20_entry = float(sym_data.rs_mom20[entry_feat_i])
                    spy_regime_entry = float(sym_data.spy_regime[entry_feat_i])

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
            rule_mask = op_fn(left, right) & np.isfinite(left) & np.isfinite(right)
        else:
            return np.array([], dtype=np.int32)

        mask &= rule_mask

    mask[:warmup] = False
    mask[-1] = False  # no next bar for entry

    signal_idx = np.flatnonzero(mask)
    return (signal_idx + 1).astype(np.int32, copy=False)


def _score_candidates_vectorized(
    rsi2: np.ndarray,
    close_px: np.ndarray,
    atr14: np.ndarray,
    volume: np.ndarray,
    vol_ma20: np.ndarray,
    sma200: np.ndarray,
    cci: np.ndarray,
    bb_width: np.ndarray,
    w: _ScoreWeights,
) -> np.ndarray:
    # SAFEGUARD: Clean data before matrix math
    rsi2 = np.nan_to_num(rsi2, nan=50.0, posinf=50.0, neginf=50.0)
    close_px = np.nan_to_num(close_px, nan=0.0, posinf=0.0, neginf=0.0)
    atr14 = np.nan_to_num(atr14, nan=0.0, posinf=0.0, neginf=0.0)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    vol_ma20 = np.nan_to_num(vol_ma20, nan=0.0, posinf=0.0, neginf=0.0)
    sma200 = np.nan_to_num(sma200, nan=0.0, posinf=0.0, neginf=0.0)
    cci = np.nan_to_num(cci, nan=0.0, posinf=0.0, neginf=0.0)
    bb_width = np.nan_to_num(bb_width, nan=0.0, posinf=0.0, neginf=0.0)

    sniper_ok = (cci < 0) & (bb_width > _SNIPER_BB_WIDTH_THRESH)
    base = (100.0 - rsi2) * float(w.rsi_factor)
    base = np.clip(base, 0.0, 100.0)
    score = base.astype(np.float64, copy=True)

    atr_pct = np.zeros_like(score)
    atr_ok = (close_px > 0) & (atr14 > 0)
    atr_pct[atr_ok] = (atr14[atr_ok] / close_px[atr_ok]) * 100.0

    score += np.where(atr_pct >= _ATR_HIGH_THRESH_PCT, float(w.atr_high_bonus),
             np.where(atr_pct >= _ATR_MED_THRESH_PCT, float(w.atr_med_bonus), 0.0))
    vol_rel = volume / (vol_ma20 + 1.0)
    score += np.where(vol_rel >= _VOL_REL_THRESH, float(w.vol_bonus), 0.0)
    trend_ok = (close_px > 0) & (sma200 > 0)
    score += np.where(trend_ok & (close_px > sma200), float(w.trend_bonus),
             np.where(trend_ok, float(w.trend_penalty), 0.0))
    score += np.where(sniper_ok, float(w.sniper_bonus), 0.0)
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
    Sniper-mode backtest engine:
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
    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float, float, bool]] = []
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
        compiled_strategies.append((strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling))

    # Diagnostic: verify trail_activation is flowing from JSON -> strategy.params -> engine.
    for strat, _w, params, _gap_ratio, _regime_filter, _base_stop_mult, _min_adx, _vix_limit_scaling in compiled_strategies:
        if _DEBUG_TRAIL_ACTIVATION or (isinstance(params.get("version_info"), dict) and params["version_info"].get("status") == "Golden State"):
            print(f"DEBUG: {strat.name} using activation: {params.get('trail_activation')}")

    wealth_present = any(_strategy_role(params) == "wealth" for _s, _w, params, _g, _r, _b, _m, _v in compiled_strategies)
    income_present = any(_strategy_role(params) == "income" for _s, _w, params, _g, _r, _b, _m, _v in compiled_strategies)
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
        adx_arr = sd.adx
        atr14_arr = sd.atr14
        vol_ma20_arr = sd.vol_ma20
        sma20_arr = sd.sma20
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bb_width
        vix_arr = sd.vix
        vix_rel20_arr = sd.vix_rel20
        spy_close_arr = sd.spy_close
        spy_sma200_arr = sd.spy_sma200

        for strat, w, params, gap_ratio, regime_filter, base_stop_mult, min_adx, vix_limit_scaling in compiled_strategies:
            vix_threshold = _get_param(params, "vix_threshold", 25.0, 10.0, 60.0)
            # Vectorized entry for base GenericStrategy genomes (optimizer hot path)
            if _can_vectorize_entry(strat, params, ai_model):
                warmup = max(int(params.get("warmup_bars", MIN_BARS) or MIN_BARS), MIN_BARS)
                entry_is = _vectorized_entry_indices(sd, params.get("entry_rules") or [], warmup)
                if entry_is.size == 0:
                    continue

                prev_is = entry_is - 1
                prev_close, open_px, low_px = close_arr[prev_is], open_arr[entry_is], low_arr[entry_is]
                low_px = np.where(np.isfinite(low_px), low_px, open_px)

                valid = np.isfinite(prev_close) & (prev_close > 0) & np.isfinite(open_px) & (open_px > 0)
                gap_pct_arr = (open_px - prev_close) / prev_close
                valid &= gap_pct_arr >= -0.08

                if regime_filter:
                    valid &= ~(np.isfinite(spy_close_arr[prev_is]) & np.isfinite(spy_sma200_arr[prev_is]) &
                               (spy_close_arr[prev_is] < spy_sma200_arr[prev_is]))

                if min_adx > 0:
                    adx_prev = adx_arr[prev_is]
                    valid &= np.isfinite(adx_prev) & (adx_prev >= min_adx)

                signal_atr = atr14_arr[prev_is]
                valid &= np.isfinite(signal_atr) & (signal_atr > 0)

                # Batched Scoring
                score = _score_candidates_vectorized(
                    rsi2_arr[prev_is],
                    prev_close,
                    signal_atr,
                    volume_arr[prev_is],
                    vol_ma20_arr[prev_is],
                    sma200_arr[prev_is],
                    cci_arr[prev_is],
                    bb_width_arr[prev_is],
                    w,
                )

                # FIX 1: Apply Parity Strategy Multipliers
                score = apply_strategy_score_multipliers(score, params)

                # FIX 2: Vectorized Limit Entry Logic
                entry_px_arr = open_px.copy()
                limit_ratio = params.get("limit_ratio")

                if limit_ratio is not None:
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
                        # Fill if Open < Target OR Low < Target
                        filled_mask = (open_px < target_px) | (low_px < target_px)

                        # Zero out scores for unfilled trades
                        score[~filled_mask] = 0.0

                        # Set entry price
                        entry_px_arr = np.where(open_px < target_px, open_px, target_px)
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

                # Final Validity Check
                valid &= score >= MIN_ENTRY_SCORE
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
                            if np_isfinite(vix_prev) and vix_prev > vix_threshold:
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

                score = _score_candidate(
                    float(rsi2_arr[prev_i]),
                    prev_close,
                    signal_atr,
                    float(volume_arr[prev_i]),
                    float(vol_ma20_arr[prev_i]),
                    float(sma200_arr[prev_i]),
                    float(cci_arr[prev_i]),
                    float(bb_width_arr[prev_i]),
                    w,
                )
                if score < MIN_ENTRY_SCORE:
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
        if len(daily_candidates) > 1:
            daily_candidates.sort(key=lambda c: c.score, reverse=True)

        current_sector_equity = float(sum(sector_exposure.values()))
        current_equity = cash + current_sector_equity
        if current_equity <= 0:
            continue

        slot_value = current_equity / MAX_POSITIONS if MAX_POSITIONS else 0.0
        open_slots = MAX_POSITIONS - len(positions)

        for cand in daily_candidates:
            if open_slots <= 0:
                break
            if cand.score < MIN_ENTRY_SCORE:
                break
            if cand.sym in positions:
                continue
            if cand.entry_px <= 0 or slot_value <= 0:
                continue

            sizing_mode = "slot"
            strat_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
            if isinstance(strat_params, dict):
                sizing_mode = str(strat_params.get("sizing_mode", "slot")).lower()

            if sizing_mode == "risk":
                risk_per_trade = current_equity * 0.02
                risk_per_share = cand.entry_px - cand.stop_px
                if risk_per_share > 0:
                    shares = int(risk_per_trade / risk_per_share)
                else:
                    shares = int((current_equity * dyn_pos_fraction) / cand.entry_px) if cand.entry_px > 0 else 0

                max_shares_by_value = int((current_equity * (dyn_pos_fraction * 1.25)) / cand.entry_px) if cand.entry_px > 0 else 0
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
                slot_value = current_equity / MAX_POSITIONS if MAX_POSITIONS else 0.0
                open_slots = MAX_POSITIONS - len(positions)

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

            try:
                if bool(pos.get("is_super_signal", False)):
                    should_exit, effective_stop, target_px = _generic_exit_decision(
                        _apply_super_signal_overrides(genome),
                        sym_data,
                        loc,
                        entry_i,
                        float(pos["entry_price"]),
                        float(pos["stop_price"]),
                    )
                elif isinstance(active_strat, GenericStrategy) and active_strat.__class__.exit is GenericStrategy.exit:
                    should_exit, effective_stop, target_px = _generic_exit_decision(
                        genome,
                        sym_data,
                        loc,
                        entry_i,
                        float(pos["entry_price"]),
                        float(pos["stop_price"]),
                    )
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
                    vol_ma20_entry = float(sym_data.vol_ma20[entry_feat_i])
                    vix_entry = float(sym_data.vix[entry_feat_i])
                    vix_rel20_entry = float(sym_data.vix_rel20[entry_feat_i])
                    rs_ratio_entry = float(sym_data.rs_ratio[entry_feat_i])
                    rs_trend_entry = float(sym_data.rs_trend[entry_feat_i])
                    rs_mom20_entry = float(sym_data.rs_mom20[entry_feat_i])
                    spy_regime_entry = float(sym_data.spy_regime[entry_feat_i])

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
