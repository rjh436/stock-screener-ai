from __future__ import annotations

import concurrent.futures
import json
import operator
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator
from threadpoolctl import threadpool_limits

from strategies.generic import GenericStrategy
from strategies.strategy_loader import load_strategies

DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    "rsi_factor": 3.0,
    "atr_high_bonus": 12.0,
    "atr_med_bonus": 5.0,
    "vol_bonus": 11.0,
    "sniper_bonus": 50.0,
    "trend_bonus": 40.0,
    "trend_penalty": -15.0,
}

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


def _compute_indicators(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()

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
        else:
            df["rs_ratio"] = 1.0
            df["rs_trend"] = 0.0
            df["spy_close"] = np.nan
            df["spy_sma200"] = np.nan

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
    vix: np.ndarray
    spy_close: np.ndarray
    spy_sma200: np.ndarray


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


def _prob_to_size_scalar(prob: float) -> float:
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(p):
        return 0.0
    if p < 0.50:
        return 0.0
    if p < 0.60:
        t = (p - 0.50) / 0.10
        return 0.25 + (0.75 * t)
    t = (p - 0.60) / 0.40
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return 1.0 + (0.25 * t)


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
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df)

            if "ticker" not in df.columns:
                df["ticker"] = sym

            if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
                df["vix"] = vix_df["close"].reindex(df.index).ffill().fillna(20.0)
            else:
                df["vix"] = 20.0

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
                vix=_get_np_col(df, "vix", 20.0, length=n),
                spy_close=_get_np_col(df, "spy_close", np.nan, length=n),
                spy_sma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
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

    ml_feature_cols = ["rsi2", "adx", "atr_pct", "dist_sma50", "dist_sma200", "vol_rel"]
    if ai_model is not None and hasattr(ai_model, "feature_names_in_"):
        try:
            ml_feature_cols = [str(c) for c in ai_model.feature_names_in_]
        except Exception:
            pass

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
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df)

            if "ticker" not in df.columns:
                df["ticker"] = sym

            if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
                df["vix"] = vix_df["close"].reindex(df.index).ffill().fillna(20.0)
            else:
                df["vix"] = 20.0

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
                vix=_get_np_col(df, "vix", 20.0, length=n),
                spy_close=_get_np_col(df, "spy_close", np.nan, length=n),
                spy_sma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
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

    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float]] = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _unwrap_genome(raw_params)
        strat_weights = params.get("scoring_weights") if isinstance(params.get("scoring_weights"), dict) else None
        w = _compile_scoring_weights(scoring_weights, strat_weights)
        gap_ratio = _resolve_gap_protection_ratio(params)
        regime_filter = bool(params.get("regime_filter", False))
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        compiled_strategies.append((strat, w, params, gap_ratio, regime_filter, base_stop_mult))

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
        sma50_arr = sd.sma50
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bb_width
        spy_close_arr = sd.spy_close
        spy_sma200_arr = sd.spy_sma200

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

            for strat, w, params, gap_ratio, regime_filter, base_stop_mult in compiled_strategies:
                entry_signal = strat.entry(df, prev_i)
                if not entry_signal:
                    continue

                if regime_filter:
                    spy_close_prev = float(spy_close_arr[prev_i])
                    spy_sma200_prev = float(spy_sma200_arr[prev_i])
                    if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                        if spy_close_prev < spy_sma200_prev:
                            continue

                if gap_ratio > 0 and prev_close > 0:
                    if open_px < prev_close * gap_ratio:
                        continue

                if ai_model is not None:
                    try:
                        rsi2 = float(rsi2_arr[prev_i])
                        adx = float(adx_arr[prev_i])
                        atr14 = float(atr14_arr[prev_i])
                        sma50 = float(sma50_arr[prev_i])
                        sma200 = float(sma200_arr[prev_i])
                        vol = float(volume_arr[prev_i])
                        vol_ma20 = float(vol_ma20_arr[prev_i])

                        atr_pct = (atr14 / prev_close) if prev_close > 0 else 0.0
                        dist_sma50 = (prev_close - sma50) / prev_close if prev_close > 0 else 0.0
                        dist_sma200 = (prev_close - sma200) / prev_close if prev_close > 0 else 0.0
                        vol_rel = vol / (vol_ma20 + 1.0)

                        features = np.array(
                            [[rsi2, adx, atr_pct, dist_sma50, dist_sma200, vol_rel]],
                            dtype=float,
                        )
                        if np.all(np_isfinite(features)):
                            features_df = pd.DataFrame(features, columns=ml_feature_cols)
                            with threadpool_limits(limits=1):
                                prob = ai_model.predict_proba(features_df)[0][1]
                            if prob < ai_threshold:
                                continue
                    except Exception:
                        pass

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
                    should_exit = bool(active_strat.exit(sym_data.df, loc, entry_i, entry_price, initial_stop))
                    effective_stop = initial_stop
                    target_px = None
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

_SCALAR_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _best_profit_target_multiple(exit_rules: Any) -> Optional[float]:
    best: Optional[float] = None
    for rule in exit_rules or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "profit_target":
            continue
        try:
            mult = float(rule.get("val", 1.0) or 1.0)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(mult) or mult <= 1.0:
            continue
        best = mult if best is None else min(best, mult)
    return best


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


def _generic_exit_decision(
    genome: Dict[str, Any],
    sd: _SymbolArrays,
    loc: int,
    entry_i: int,
    entry_price: float,
    initial_stop: float,
) -> Tuple[bool, float, Optional[float]]:
    """
    Engine-native exit logic for rule-based (GenericStrategy) genomes.

    Returns:
      - should_exit
      - effective_stop (including trail_activation)
      - target_px (when profit_target triggers), else None
    """
    if isinstance(genome, dict) and isinstance(genome.get("genome"), dict):
        genome = genome["genome"]

    np_isfinite = np.isfinite

    close_px = float(sd.close[loc])
    if not np_isfinite(close_px) or close_px <= 0:
        close_px = float(entry_price)

    high_px = float(sd.high[loc])
    if not np_isfinite(high_px):
        high_px = close_px

    low_px = float(sd.low[loc])
    if not np_isfinite(low_px):
        low_px = close_px

    days_held = int(loc) - int(entry_i)
    pnl_pct = ((close_px - entry_price) / entry_price) * 100.0 if entry_price else 0.0

    use_bb_exit = bool(genome.get("use_bb_exit", False))

    atr = float(sd.atr14[loc])
    if not np_isfinite(atr) or atr <= 0:
        atr = close_px * 0.02

    trail_mult_raw = genome.get("trail_atr", genome.get("stop_loss_atr", 3.0))
    try:
        trail_mult = float(trail_mult_raw or 3.0)
    except (TypeError, ValueError):
        trail_mult = 3.0

    try:
        act_raw = genome.get("trail_activation", 1.0) if isinstance(genome, dict) else 1.0
        trail_activation = float(act_raw or 1.0)
    except (TypeError, ValueError, AttributeError):
        trail_activation = 1.0
    if not np_isfinite(trail_activation) or trail_activation < 1.0:
        trail_activation = 1.0

    peak_high = high_px
    if 0 <= entry_i <= loc:
        try:
            peak_high_val = float(np.nanmax(sd.high[entry_i : loc + 1]))
        except Exception:
            peak_high_val = float("nan")
        if np_isfinite(peak_high_val):
            peak_high = peak_high_val

    trailing_active = (trail_activation <= 1.0) or (peak_high >= entry_price * trail_activation)
    trailing_stop = float("-inf")
    if trailing_active and atr > 0:
        trailing_stop = peak_high - (atr * trail_mult)

    effective_stop = float(initial_stop)
    if np_isfinite(trailing_stop):
        effective_stop = max(effective_stop, float(trailing_stop))

    # 1. STOP LOSS CHECK
    if low_px < effective_stop:
        return True, effective_stop, None

    # 1b. Bollinger profit release (optional)
    if use_bb_exit:
        bb_upper = float(sd.bb_upper[loc])
        if np_isfinite(bb_upper) and high_px >= bb_upper:
            return True, effective_stop, None

    # 2. TIME-BASED EXITS
    try:
        time_limit = int(genome.get("time_stop", 45) or 45)
    except Exception:
        time_limit = 45
    time_limit = max(1, time_limit)

    # Progressive tightening in final 20% of hold period
    if days_held >= int(time_limit * 0.8):
        if pnl_pct < -5.0:
            return True, effective_stop, None
        if pnl_pct < 3.0:
            sma20 = float(sd.sma20[loc])
            if not np_isfinite(sma20):
                sma20 = close_px
            if close_px < sma20:
                return True, effective_stop, None

    # Final time limit
    if days_held >= time_limit:
        if pnl_pct < 0:
            return True, effective_stop, None
        if pnl_pct < 8.0:
            return True, effective_stop, None
        sma50 = float(sd.sma50[loc])
        if np_isfinite(sma50) and close_px < sma50:
            return True, effective_stop, None

    # 3. TREND BREAK PROTECTION
    sma50 = float(sd.sma50[loc])
    if np_isfinite(sma50) and close_px < sma50:
        if pnl_pct > 0:
            return True, effective_stop, None
        if days_held > 10:
            return True, effective_stop, None

    # 4. PROFIT TARGETS
    target_px: Optional[float] = None
    if not use_bb_exit:
        mult = _best_profit_target_multiple(genome.get("exit_rules"))
        if mult is not None:
            target_px = entry_price * mult
            if high_px >= target_px:
                return True, effective_stop, target_px

    # 5. OTHER RULE-BASED EXITS
    row = None
    for rule in genome.get("exit_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") == "profit_target":
            continue

        col = rule.get("col")
        op = rule.get("op")
        if not col or not op:
            continue

        op_fn = _SCALAR_OPS.get(str(op))
        if op_fn is None:
            continue

        if row is None:
            try:
                row = sd.df.iloc[loc]
            except Exception:
                break

        val_a = row.get(col, np.nan)
        if "val" in rule:
            val_b = rule.get("val", np.nan)
        elif "ref" in rule:
            val_b = row.get(rule.get("ref"), np.nan)
        else:
            continue

        try:
            a = float(val_a)
            b = float(val_b)
        except Exception:
            continue
        if not np_isfinite(a) or not np_isfinite(b):
            continue
        if op_fn(a, b):
            return True, effective_stop, None

    return False, effective_stop, None


def _can_vectorize_entry(strategy_obj: Any, params: Dict[str, Any], ai_model: Any) -> bool:
    # UNBLOCK: Allow vectorized path even with AI, provided it supports batched prediction
    if ai_model is not None and not hasattr(ai_model, "predict_proba"):
        return False
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

    ml_feature_cols = ["rsi2", "adx", "atr_pct", "dist_sma50", "dist_sma200", "vol_rel"]
    if ai_model is not None and hasattr(ai_model, "feature_names_in_"):
        try:
            ml_feature_cols = [str(c) for c in ai_model.feature_names_in_]
        except Exception:
            pass

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
    use_global_ai = ai_model is not None and hasattr(ai_model, "predict_proba")
    all_pre_ai_candidates: List[Dict[str, Any]] = []

    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float]] = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _unwrap_genome(raw_params)
        strat_weights = params.get("scoring_weights") if isinstance(params.get("scoring_weights"), dict) else None
        w = _compile_scoring_weights(scoring_weights, strat_weights)
        gap_ratio = _resolve_gap_protection_ratio(params)
        regime_filter = bool(params.get("regime_filter", False))
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        compiled_strategies.append((strat, w, params, gap_ratio, regime_filter, base_stop_mult))

    # Diagnostic: verify trail_activation is flowing from JSON -> strategy.params -> engine.
    for strat, _w, params, _gap_ratio, _regime_filter, _base_stop_mult in compiled_strategies:
        if _DEBUG_TRAIL_ACTIVATION or (isinstance(params.get("version_info"), dict) and params["version_info"].get("status") == "Golden State"):
            print(f"DEBUG: {strat.name} using activation: {params.get('trail_activation')}")

    wealth_present = any(_strategy_role(params) == "wealth" for _s, _w, params, _g, _r, _b in compiled_strategies)
    income_present = any(_strategy_role(params) == "income" for _s, _w, params, _g, _r, _b in compiled_strategies)
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
        sma50_arr = sd.sma50
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bb_width
        spy_close_arr = sd.spy_close
        spy_sma200_arr = sd.spy_sma200

        for strat, w, params, gap_ratio, regime_filter, base_stop_mult in compiled_strategies:
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
                valid &= ((open_px - prev_close) / prev_close) >= -0.08

                if regime_filter:
                    valid &= ~(np.isfinite(spy_close_arr[prev_is]) & np.isfinite(spy_sma200_arr[prev_is]) &
                               (spy_close_arr[prev_is] < spy_sma200_arr[prev_is]))

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
                valid &= score >= MIN_ENTRY_SCORE
                if not np.any(valid):
                    continue

                cand_k = np.flatnonzero(valid)
                if use_global_ai:
                    for k in cand_k:
                        day_idx = int(sd.gidx[entry_is[k]]) if sd.gidx.size else date_to_idx.get(idx[entry_is[k]])
                        if day_idx is None:
                            continue
                        prev_close_k = float(prev_close[k])
                        if prev_close_k <= 0:
                            continue

                        rsi2_k = float(rsi2_arr[prev_is[k]])
                        adx_k = float(adx_arr[prev_is[k]])
                        atr_k = float(signal_atr[k])
                        sma50_k = float(sma50_arr[prev_is[k]])
                        sma200_k = float(sma200_arr[prev_is[k]])
                        vol_k = float(volume_arr[prev_is[k]])
                        vol_ma20_k = float(vol_ma20_arr[prev_is[k]])

                        atr_pct = atr_k / prev_close_k if prev_close_k > 0 else 0.0
                        dist_sma50 = (prev_close_k - sma50_k) / prev_close_k if prev_close_k > 0 else 0.0
                        dist_sma200 = (prev_close_k - sma200_k) / prev_close_k if prev_close_k > 0 else 0.0
                        vol_rel = vol_k / (vol_ma20_k + 1.0)

                        all_pre_ai_candidates.append(
                            {
                                "day_idx": day_idx,
                                "sym": sym,
                                "entry_px": float(open_px[k]),
                                "stop_px": float(open_px[k] * 0.97),
                                "score": float(score[k]),
                                "strategy_name": strat.name,
                                "strategy_obj": strat,
                                "entry_i": int(entry_is[k]),
                                "signal_i": int(prev_is[k]),
                                "rsi2": rsi2_k,
                                "adx": adx_k,
                                "atr_pct": atr_pct,
                                "dist_sma50": dist_sma50,
                                "dist_sma200": dist_sma200,
                                "vol_rel": vol_rel,
                            }
                        )
                else:
                    for k in cand_k:
                        day_idx = int(sd.gidx[entry_is[k]]) if sd.gidx.size else date_to_idx.get(idx[entry_is[k]])
                        if day_idx is None:
                            continue
                        candidates_by_day[day_idx].append(
                            _Candidate(
                                sym=sym,
                                entry_px=float(open_px[k]),
                                stop_px=float(open_px[k] * 0.97),
                                score=float(score[k]),
                                strategy_name=strat.name,
                                strategy_obj=strat,
                                entry_i=int(entry_is[k]),
                                signal_i=int(prev_is[k]),
                            )
                        )
                continue

            # Fallback path (supports custom strategy.entry / AI filtering)
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

                if use_global_ai:
                    rsi2 = float(rsi2_arr[prev_i])
                    adx = float(adx_arr[prev_i])
                    atr14 = float(atr14_arr[prev_i])
                    sma50 = float(sma50_arr[prev_i])
                    sma200 = float(sma200_arr[prev_i])
                    vol = float(volume_arr[prev_i])
                    vol_ma20 = float(vol_ma20_arr[prev_i])

                    atr_pct = (atr14 / prev_close) if prev_close > 0 else 0.0
                    dist_sma50 = (prev_close - sma50) / prev_close if prev_close > 0 else 0.0
                    dist_sma200 = (prev_close - sma200) / prev_close if prev_close > 0 else 0.0
                    vol_rel = vol / (vol_ma20 + 1.0)

                    all_pre_ai_candidates.append(
                        {
                            "day_idx": day_idx,
                            "sym": sym,
                            "entry_px": float(entry_px),
                            "stop_px": float(stop_price),
                            "score": float(score),
                            "strategy_name": strat.name,
                            "strategy_obj": strat,
                            "entry_i": i,
                            "signal_i": prev_i,
                            "rsi2": rsi2,
                            "adx": adx,
                            "atr_pct": atr_pct,
                            "dist_sma50": dist_sma50,
                            "dist_sma200": dist_sma200,
                            "vol_rel": vol_rel,
                        }
                    )
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
                    )
                )

    if use_global_ai and all_pre_ai_candidates:
        probs = None
        try:
            features = np.array(
                [
                    [
                        cand["rsi2"],
                        cand["adx"],
                        cand["atr_pct"],
                        cand["dist_sma50"],
                        cand["dist_sma200"],
                        cand["vol_rel"],
                    ]
                    for cand in all_pre_ai_candidates
                ],
                dtype=float,
            )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features_df = pd.DataFrame(features, columns=ml_feature_cols)
            with threadpool_limits(limits=1):
                probs = ai_model.predict_proba(features_df)[:, 1]
        except Exception:
            probs = None

        if probs is None or len(probs) != len(all_pre_ai_candidates):
            for cand in all_pre_ai_candidates:
                candidates_by_day[cand["day_idx"]].append(
                    _Candidate(
                        sym=cand["sym"],
                        entry_px=cand["entry_px"],
                        stop_px=cand["stop_px"],
                        score=cand["score"],
                        strategy_name=cand["strategy_name"],
                        strategy_obj=cand["strategy_obj"],
                        entry_i=cand["entry_i"],
                        signal_i=cand["signal_i"],
                    )
                )
        else:
            for cand, prob in zip(all_pre_ai_candidates, probs):
                scalar = _prob_to_size_scalar(prob)
                if scalar <= 0.0:
                    continue
                candidates_by_day[cand["day_idx"]].append(
                    _Candidate(
                        sym=cand["sym"],
                        entry_px=cand["entry_px"],
                        stop_px=cand["stop_px"],
                        score=cand["score"],
                        strategy_name=cand["strategy_name"],
                        strategy_obj=cand["strategy_obj"],
                        entry_i=cand["entry_i"],
                        signal_i=cand["signal_i"],
                        ai_prob=float(prob),
                        size_scalar=float(scalar),
                    )
                )

    # --- Super Signal Confluence (Wealth + Income on same symbol/day) ---
    if confluence_possible:
        for day_idx, day_list in enumerate(candidates_by_day):
            if not day_list:
                continue

            wealth_by_sym: Dict[str, _Candidate] = {}
            income_by_sym: Dict[str, _Candidate] = {}

            for cand in day_list:
                c_params = getattr(cand.strategy_obj, "params", getattr(cand.strategy_obj, "genome", {})) or {}
                role = _strategy_role(c_params or {})
                if role == "wealth":
                    best = wealth_by_sym.get(cand.sym)
                    if best is None or cand.score > best.score:
                        wealth_by_sym[cand.sym] = cand
                elif role == "income":
                    best = income_by_sym.get(cand.sym)
                    if best is None or cand.score > best.score:
                        income_by_sym[cand.sym] = cand

            super_syms = set(wealth_by_sym).intersection(income_by_sym)
            if not super_syms:
                if super_signal_only:
                    candidates_by_day[day_idx] = []
                continue

            super_candidates: List[_Candidate] = []
            for sym in super_syms:
                w_cand = wealth_by_sym[sym]
                i_cand = income_by_sym[sym]
                super_candidates.append(
                    _Candidate(
                        sym=sym,
                        entry_px=w_cand.entry_px,
                        stop_px=w_cand.stop_px,
                        score=max(w_cand.score, i_cand.score),
                        strategy_name=SUPER_SIGNAL_NAME,
                        strategy_obj=w_cand.strategy_obj,
                        entry_i=w_cand.entry_i,
                        signal_i=w_cand.signal_i,
                        is_super_signal=True,
                    )
                )

            if super_signal_only:
                candidates_by_day[day_idx] = super_candidates
            else:
                kept = [c for c in day_list if c.sym not in super_syms]
                kept.extend(super_candidates)
                candidates_by_day[day_idx] = kept

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
            shares = int(shares * cand.size_scalar)
            if shares <= 0:
                continue

            cost = shares * cand.entry_px
            if export_ml_data and cash < cost:
                shares = 100
                cost = 0.0
            elif cash < cost:
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
                    should_exit = bool(active_strat.exit(sym_data.df, loc, entry_i, pos["entry_price"], pos["stop_price"]))
                    effective_stop = float(pos["stop_price"])
                    target_px = None
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
