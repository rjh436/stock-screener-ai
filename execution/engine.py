from __future__ import annotations

import sys
import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, SMAIndicator

from strategies.generic import GenericStrategy
from strategies.strategy_loader import load_strategies
from execution.parity import (
    apply_strategy_score_multipliers,
    DEFAULT_SCORING_WEIGHTS,
)
from execution.shared_logic import _generic_exit_decision

# --- CONFIGURATION ---
MIN_BARS = 200
MIN_ENTRY_SCORE = 120.0
SUPER_SIGNAL_NAME = "SUPER SIGNAL (Wealth + Income)"

_ATR_HIGH_THRESH_PCT = 3.0
_ATR_MED_THRESH_PCT = 2.0
_VOL_REL_THRESH = 1.5
# AUDIT UPDATE: Tighter VCP threshold for Minervini compliance
_VCP_BB_WIDTH_THRESH = 0.15

_DEBUG_TRAIL_ACTIVATION = os.environ.get("APEX_DEBUG_TRAIL_ACTIVATION", "").strip() not in ("", "0", "false", "False")


def get_sector(symbol: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"


def inject_market_rs_rank(enriched, all_dates, lookbacks=(63, 126, 189, 252),
                          weights=(0.40, 0.20, 0.20, 0.20),
                          min_history=252, min_names=None):
    """
    Creates rs_rating in [1..99] for each symbol/day using cross-sectional percentile rank.
    """
    syms = list(enriched.keys())
    n_days = len(all_dates)
    n_syms = len(syms)

    if n_syms == 0:
        return

    # AUDIT FIX: Dynamic min_names to handle small universes (Diagnostic Mode)
    if min_names is None:
        min_names = min(50, max(1, int(n_syms * 0.5)))
    
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

        # V2 UPGRADE: Robust ADR % Calculation
        df["hl_range"] = df["high"] - df["low"]
        close_safe = df["close"].replace(0, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            df["adr_pct"] = (df["hl_range"] / close_safe).rolling(20).mean() * 100.0

        df["adr_pct"] = df["adr_pct"].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        for p in (10, 20, 50, 200):
            df[f"sma{p}"] = df["close"].rolling(p).mean()
            df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        # AUDIT UPGRADE: Add SMA150 for Minervini Trend Template
        df["sma150"] = df["close"].rolling(150).mean()

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
        df["natr"] = (df["atr14"] / df["close"]) * 100.0

        df["highest20"] = df["high"].rolling(20).max()
        df["highest20_1"] = df["highest20"].shift(1)
        df["donchian_20"] = df["highest20_1"]
        df["donchian20"] = df["donchian_20"]
        df["highest55"] = df["high"].rolling(55).max()
        df["highest55_1"] = df["highest55"].shift(1)
        
        df["prev_high"] = df["high"].shift(1)
        df["prev_close"] = df["close"].shift(1)
        
        with np.errstate(divide="ignore", invalid="ignore"):
            df["gap_pct"] = ((df["open"] - df["prev_close"]) / df["prev_close"]) * 100.0
        df["gap_pct"] = df["gap_pct"].fillna(0.0)

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
        df["roc_40"] = df["close"].pct_change(40) * 100.0
        df["adr_pct_ma10"] = df["adr_pct"].rolling(10).mean()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            df["clv"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["clv"] = df["clv"].fillna(0.5)

        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        
        df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
        df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ma50"] = df["volume"].rolling(50).mean()

        df["highest10"] = df["high"].rolling(10).max()
        df["highest10_1"] = df["highest10"].shift(1) 

        if spy_df is not None and not spy_df.empty:
            spy_aligned = spy_df["close"].reindex(df.index).ffill().bfill()
            df["rs_ratio"] = df["close"] / spy_aligned
            df["spy_close"] = spy_aligned
            df["spy_sma20"] = spy_aligned.rolling(20).mean()
            df["spy_sma50"] = spy_aligned.rolling(50).mean()
            if "sma200" in spy_df.columns:
                df["spy_sma200"] = spy_df["sma200"].reindex(df.index).ffill().bfill()
            else:
                df["spy_sma200"] = spy_aligned.rolling(200).mean()
        else:
            df["rs_ratio"] = 1.0
            df["spy_close"] = np.nan
            df["spy_sma20"] = np.nan
            df["spy_sma50"] = np.nan
            df["spy_sma200"] = np.nan

        if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
            vix_aligned = vix_df["close"].reindex(df.index).ffill().bfill()
            df["vix"] = vix_aligned
        elif "vix" in df.columns:
            df["vix"] = df["vix"].ffill().bfill()
        else:
            df["vix"] = 20.0
        df["vix"] = df["vix"].fillna(20.0)

        df["sma50"] = df["close"].rolling(50).mean()
        df["sma200"] = df["close"].rolling(200).mean()
        df["high_52w"] = df["high"].rolling(252).max()
        df["low_52w"] = df["low"].rolling(252).min()
        
        # AUDIT FIX: Fill NaN slope with 0.0 to prevent hard gates from rejecting all trades
        df["sma200_slope"] = df["sma200"].diff(22).fillna(0.0)
        
        df["std_20"] = df["close"].rolling(20).std()
        df["bb_width"] = (4 * df["std_20"]) / (df["close"].rolling(20).mean() + 1e-9)
        
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
        "max_drawdown_pct": 0.0,
        "cagr": 0.0,
        "params": params or {},
        "equity_curve": [],
    }


@dataclass(frozen=True, slots=True)
class _ScoreWeights:
    rsi_factor: float
    vcp_bonus: float
    vol_bonus: float
    trend_bonus: float


def _score_row_dual_core(
    rsi14: float,
    bb_width: float,
    natr: float,
    close_px: float,
    high_52w: float,
    weights: Dict[str, float],
) -> float:
    rsi_factor = float(weights.get("rsi_factor", 1.0))
    score = (rsi14 * rsi_factor)
    if bb_width < 0.15:
        score += 50.0
    if natr > 3.0:
        score += 20.0
    if close_px >= (high_52w * 0.85):
        score += 30.0
    return max(0.0, float(score))


def calculate_backtest_quality_score(
    row_or_rsi2,
    strategy_name: str = "",
    weights: Optional[Dict[str, float]] = None,
    **kwargs,
) -> float:
    """
    Compatibility helper for diagnostics (diag_trace.py).
    Uses the same core scoring logic as the engine.
    """
    merged = dict(DEFAULT_SCORING_WEIGHTS)
    if isinstance(weights, dict):
        merged.update(weights)

    if isinstance(row_or_rsi2, (pd.Series, dict)):
        row = row_or_rsi2
        rsi14 = float(row.get("rsi14", 50) or 50)
        bb_width = float(row.get("bb_width", np.nan))
        close_px = float(row.get("close", 0.0) or 0.0)
        high_52w = float(row.get("high_52w", np.nan))
        natr = row.get("natr")
        if natr is None or not np.isfinite(natr):
            atr14 = float(row.get("atr14", 0.0) or 0.0)
            natr = (atr14 / close_px) * 100.0 if close_px > 0 and atr14 > 0 else 0.0
        else:
            natr = float(natr)
    else:
        rsi14 = float(kwargs.get("rsi14", 50) or 50)
        bb_width = float(kwargs.get("bb_width", np.nan))
        close_px = float(kwargs.get("close", 0.0) or 0.0)
        high_52w = float(kwargs.get("high_52w", np.nan))
        natr = kwargs.get("natr")
        if natr is None or not np.isfinite(natr):
            atr14 = float(kwargs.get("atr14", 0.0) or 0.0)
            natr = (atr14 / close_px) * 100.0 if close_px > 0 and atr14 > 0 else 0.0
        else:
            natr = float(natr)

    return _score_row_dual_core(rsi14, bb_width, natr, close_px, high_52w, merged)


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
    rsi14: np.ndarray
    atr14: np.ndarray
    natr: np.ndarray
    volma50: np.ndarray
    sma10: np.ndarray
    sma20: np.ndarray
    sma50: np.ndarray
    sma150: np.ndarray
    sma200: np.ndarray
    sma200slope: np.ndarray
    high52w: np.ndarray
    low52w: np.ndarray
    bbwidth: np.ndarray
    spyclose: np.ndarray
    spysma20: np.ndarray
    spysma50: np.ndarray
    spysma200: np.ndarray
    rsrating: np.ndarray
    adr_pct: np.ndarray
    prev_high: np.ndarray
    highest10_1: np.ndarray
    clv: np.ndarray
    trend_mask: np.ndarray # AUDIT FIX: Vectorized Gate


@dataclass(slots=True)
class _Candidate:
    sym: str
    entry_px: float
    stop_px: float
    score: float
    strategy_name: str
    entry_i: int
    size_scalar: float = 1.0


@dataclass(slots=True)
class PreparedBacktestData:
    enriched: Dict[str, _SymbolArrays]
    all_dates: np.ndarray


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
    if spy_df is not None and not spy_df.empty:
        spy_df = spy_df.copy()
        spy_df.columns = spy_df.columns.str.lower()
        if "sma200" not in spy_df.columns:
            spy_df["sma200"] = spy_df["close"].rolling(200).mean()

    enriched: Dict[str, _SymbolArrays] = {}

    for sym in symbols:
        df_raw = data_dict.get(sym)
        if df_raw is None or df_raw.empty:
            continue

        try:
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df, vix_df=vix_df)

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
            
            # --- VECTORIZED DATA EXTRACTION ---
            open_arr = _get_np_col(df, "open", np.nan, length=n)
            high_arr = _get_np_col(df, "high", np.nan, length=n)
            low_arr = _get_np_col(df, "low", np.nan, length=n)
            close_arr = _get_np_col(df, "close", np.nan, length=n)
            sma50_arr = _get_np_col(df, "sma50", np.nan, length=n)
            sma150_arr = _get_np_col(df, "sma150", np.nan, length=n)
            sma200_arr = _get_np_col(df, "sma200", np.nan, length=n)
            slope_arr = _get_np_col(df, "sma200_slope", 0.0, length=n)
            high52_arr = _get_np_col(df, "high_52w", np.nan, length=n)
            low52_arr = _get_np_col(df, "low_52w", np.nan, length=n)
            bb_w_arr = _get_np_col(df, "bb_width", 100.0, length=n)
            natr_arr = _get_np_col(df, "natr", 100.0, length=n)

            if not np.isfinite(low_arr).any():
                low_arr = open_arr
            if not np.isfinite(high_arr).any():
                high_arr = open_arr

            # --- AUDIT FIX: VECTORIZED MINERVINI HARD GATES (C-SPEED) ---
            # V18.3 FIX: Added 2% Tolerance to Trend Checks to survive shakeouts
            # V18.3 FIX: Decoupled VCP from Trend Mask
            with np.errstate(invalid='ignore'): 
                trend_mask = (
                    (close_arr > (sma50_arr * 0.98)) &        # 2% Tolerance
                    (sma50_arr > (sma150_arr * 0.98)) &       # 2% Tolerance
                    (sma150_arr > sma200_arr) &
                    (slope_arr > 0) &
                    (close_arr > 0.75 * high52_arr) &
                    (close_arr > 1.30 * low52_arr)
                )
            
            trend_mask = np.nan_to_num(trend_mask, nan=False).astype(bool)
            # ------------------------------------------------------------

            enriched[sym] = _SymbolArrays(
                df=df,
                index=df.index.values.astype("datetime64[ns]"),
                gidx=np.empty(n, dtype=np.int32),
                open=open_arr,
                high=high_arr,
                low=low_arr,
                close=close_arr,
                volume=_get_np_col(df, "volume", 0.0, length=n),
                rsi14=_get_np_col(df, "rsi14", 50.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                natr=natr_arr,
                volma50=_get_np_col(df, "vol_ma50", 1.0, length=n),
                sma10=_get_np_col(df, "sma10", np.nan, length=n),
                sma20=_get_np_col(df, "sma20", np.nan, length=n),
                sma50=sma50_arr,
                sma150=sma150_arr,
                sma200=sma200_arr,
                sma200slope=slope_arr,
                high52w=high52_arr,
                low52w=low52_arr,
                bbwidth=bb_w_arr,
                spyclose=_get_np_col(df, "spy_close", np.nan, length=n),
                spysma20=_get_np_col(df, "spy_sma20", np.nan, length=n),
                spysma50=_get_np_col(df, "spy_sma50", np.nan, length=n),
                spysma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
                rsrating=np.zeros(n, dtype=np.float64),
                adr_pct=_get_np_col(df, "adr_pct", 0.0, length=n),
                prev_high=_get_np_col(df, "prev_high", 0.0, length=n),
                highest10_1=_get_np_col(df, "highest10_1", 0.0, length=n),
                clv=_get_np_col(df, "clv", 0.5, length=n),
                trend_mask=trend_mask 
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
    inject_market_rs_rank(enriched, all_dates) # Auto-dynamic min_names
    for sym_data in enriched.values():
        try:
            sym_data.df["rs_rating"] = sym_data.rsrating
        except Exception:
            continue

    return PreparedBacktestData(enriched=enriched, all_dates=all_dates)


def _legacy_run_backtest(
    strategy,
    data,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    global_data=None,
    scoring_weights=None,
    pre_calculated_data: Optional["PreparedBacktestData"] = None,
    end_date=None,
    **kwargs
):
    _debug_fired = {"done": False}

    def DBG(msg: str) -> None:
        if _debug_fired["done"]:
            return
        _debug_fired["done"] = True
        print(msg)

    if hasattr(data, "enriched"):
        pre_calculated_data = data
        data = None

    def _unwrap_genome(params: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict) and isinstance(params.get("genome"), dict):
            return params["genome"]
        return params

    strategies = strategy if isinstance(strategy, (list, tuple)) else [strategy]
    strategies = [s for s in strategies if s is not None]
    if not strategies:
        return _empty_result("NoStrategy", float(start_cash), {})

    strategy_label = strategies[0].name if len(strategies) == 1 else "MultiStrategy"

    if pre_calculated_data is not None:
        prepared = pre_calculated_data
    else:
        prepared = prepare_backtest_data(data or {}, symbol_universe, start_date, global_data)
    enriched = prepared.enriched
    all_dates = prepared.all_dates

    if not enriched:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]

    compiled_strategies = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _unwrap_genome(raw_params)
        w = _ScoreWeights(1.0, 50.0, 20.0, 30.0)
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        compiled_strategies.append((strat, w, params, base_stop_mult))

    global_spy_close = np.zeros(len(all_dates), dtype=np.float64)
    global_spy_sma200 = np.zeros(len(all_dates), dtype=np.float64)

    if global_data and "SPY" in global_data:
        spy_df_raw = global_data["SPY"]
        if not spy_df_raw.empty:
            spy_df_raw = spy_df_raw.copy()
            spy_df_raw.columns = spy_df_raw.columns.str.lower()
            if "sma200" not in spy_df_raw.columns:
                 spy_df_raw["sma200"] = spy_df_raw["close"].rolling(200).mean()
            spy_aligned = spy_df_raw.reindex(all_dates).ffill().bfill()
            global_spy_close = spy_aligned["close"].fillna(0).to_numpy(dtype=np.float64)
            global_spy_sma200 = spy_aligned["sma200"].fillna(0).to_numpy(dtype=np.float64)

    # --- MAIN LOOP (Optimized) ---
    for sym, sd in enriched.items():
        n_bars = len(sd.index)
        if n_bars < 2: continue

        # Vectorized check handles all hard gates.
        # V18.5 FIX: Iterate Setup Days directly.
        # We trade on curr_i (Tomorrow) based on prev_i (Today's Setup).
        setup_indices = np.where(sd.trend_mask)[0]
        if len(setup_indices) < 1:
            DBG(f"{sym}: REJECTED - Trend Mask Empty")
            continue

        for prev_i in setup_indices:
            curr_i = prev_i + 1

            # Boundary Safety
            if curr_i >= n_bars:
                continue

            day_idx = sd.gidx[curr_i]
            if day_idx < 1 or day_idx >= len(all_dates): continue

            spy_c = global_spy_close[day_idx - 1]
            spy_200 = global_spy_sma200[day_idx - 1]
            
            # Market Regime Filter (Allow trades if SPY is missing/zero)
            if spy_c > 0 and spy_200 > 0 and spy_c < spy_200:
                DBG(f"{sym}: REJECTED - SPY Filter")
                continue 

            rs_rating = float(sd.rsrating[prev_i])
            if not np.isfinite(rs_rating) or rs_rating <= 0:
                # Fail-open if RS rating was not computed (diagnostic safety).
                rs_rating = 99.0

            for strat, w, params, base_stop_mult in compiled_strategies:
                # Genome Filters
                min_rs = float(params.get("rs_rating", 80))
                max_bb = float(params.get("bb_width_max", 0.20))
                
                if rs_rating < min_rs:
                    DBG(f"{sym}: REJECTED - RS Rating {rs_rating} < {min_rs}")
                    continue
                # VCP CHECK RE-INSERTED HERE (Soft Gate):
                width = float(sd.bbwidth[prev_i])
                if width > max_bb:
                    DBG(f"{sym}: REJECTED - VCP Width {width} > {max_bb}")
                    continue

                # Calculate Entry/Stop
                open_px = float(sd.open[curr_i])
                entry_px = open_px
                stop_px = 0.0
                
                stop_type = str(params.get("stop_loss_type", "atr")).lower()
                if "low" in stop_type:
                    day_low = float(sd.low[curr_i])
                    stop_px = day_low * 0.99
                else:
                    atr = float(sd.atr14[prev_i])
                    stop_px = entry_px - (atr * base_stop_mult)

                score = _score_row_dual_core(
                    sd.rsi14[prev_i], sd.bbwidth[prev_i], sd.natr[prev_i],
                    sd.close[prev_i], sd.high52w[prev_i], {"rsi_factor":1.0}
                )

                candidates_by_day[day_idx].append(
                    _Candidate(sym, entry_px, stop_px, score, strat.name, curr_i)
                )
                DBG(f"{sym}: ACCEPTED - Trade Generated")

    # --- SIMULATION LOOP ---
    portfolio = {s.name: {"cash": float(start_cash), "positions": {}} for s in strategies}
    final_results = []

    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None

    for strat in strategies:
        port = portfolio[strat.name]
        cash = port["cash"]
        positions = port["positions"]
        trades_list = []
        equity_curve = []
        
        params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        max_pos = int(params.get("max_positions", 10) or 10)
        risk_per_trade = float(params.get("risk_per_trade", 0.01) or 0.01)
        max_pos_size_pct = float(params.get("max_pos_size_pct", 0.30) or 0.30)
        partial_profit_day = int(params.get("partial_profit_day", 4) or 4)

        for day_idx, candidates in enumerate(candidates_by_day):
            # AUDIT FIX: Respect Start/End dates
            current_dt_np = all_dates[day_idx]
            if start_ts and current_dt_np < start_ts.to_datetime64(): continue
            if end_ts and current_dt_np > end_ts.to_datetime64(): break

            # 1. Manage Positions
            to_remove = []
            for sym, pos in positions.items():
                sym_data = enriched[sym]
                # Check for exit (simplified for speed)
                curr_loc_arr = np.searchsorted(sym_data.gidx, [day_idx])
                if curr_loc_arr[0] >= len(sym_data.close): continue
                loc = curr_loc_arr[0]
                
                # Check actual date match
                if sym_data.gidx[loc] != day_idx: continue

                current_close = float(sym_data.close[loc])
                current_low = float(sym_data.low[loc])
                
                # Exit Logic
                should_exit = False
                exit_px = current_close
                
                # Stop Loss
                if current_low < pos["stop_price"]:
                    should_exit = True
                    exit_px = min(float(sym_data.open[loc]), pos["stop_price"])
                
                # Time Stop / Trailing (Generic)
                days_held = day_idx - pos["entry_day_idx"]
                if days_held > 50: # Hard max hold
                     should_exit = True
                
                if should_exit:
                    shares = pos["shares"]
                    proceeds = shares * exit_px
                    cash += proceeds
                    trades_list.append({
                        "Symbol": sym, "Entry": pos["entry_price"], "Exit": exit_px,
                        "PnL": proceeds - (shares * pos["entry_price"]),
                        "Return %": (exit_px/pos["entry_price"] - 1)*100
                    })
                    to_remove.append(sym)
                    continue
                
                # Update MTM
                pos["last_price"] = current_close

            for sym in to_remove:
                del positions[sym]

            # 2. Enter New Trades
            day_candidates = [c for c in candidates if c.strategy_name == strat.name]
            day_candidates.sort(key=lambda x: x.score, reverse=True)

            for cand in day_candidates:
                if len(positions) >= max_pos: break
                if cand.sym in positions: continue
                
                mtm_equity = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
                
                risk_amt = mtm_equity * risk_per_trade
                dist = max(cand.entry_px - cand.stop_px, cand.entry_px * 0.005)
                shares = int(risk_amt / dist)
                
                # Caps
                max_cap = mtm_equity * max_pos_size_pct
                if shares * cand.entry_px > max_cap:
                    shares = int(max_cap / cand.entry_px)
                if shares * cand.entry_px > cash:
                    shares = int(cash / cand.entry_px)
                
                if shares == 0:
                    sym = cand.sym
                    DBG(f"{sym}: REJECTED - Zero Shares")
                    continue

                if shares > 0:
                    cash -= shares * cand.entry_px
                    positions[cand.sym] = {
                        "entry_price": cand.entry_px,
                        "stop_price": cand.stop_px,
                        "shares": shares,
                        "entry_day_idx": day_idx,
                        "last_price": cand.entry_px
                    }

            # 3. Record Equity (Lazy Timestamp)
            mtm = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
            if day_idx % 5 == 0: # Record every 5 days to save memory
                equity_curve.append({"Date": all_dates[day_idx], "Equity": mtm})

        # Finalize
        final_val = mtm
        df_trades = pd.DataFrame(trades_list)
        win_rate = 0.0
        if not df_trades.empty:
            win_rate = (len(df_trades[df_trades["PnL"] > 0]) / len(df_trades)) * 100

        res = _empty_result(strat.name, start_cash, params)
        res.update({
            "final_value": final_val,
            "total_trades": len(trades_list),
            "hit_rate": win_rate,
            "cagr": ((final_val / start_cash) ** (365 / len(all_dates)) - 1) * 100 if len(all_dates) > 365 else 0.0,
            "equity_curve": equity_curve,
            "trades_list": trades_list
        })
        final_results.append(res)

    if len(final_results) == 1:
        return final_results[0]
    return final_results

# Backwards compatibility
run_backtest = _legacy_run_backtest

def calculate_stop_price(entry_price, atr, multiplier):
    return entry_price - (atr * multiplier)

def _run_cli() -> int:
    import argparse
    from datetime import datetime
    from data.loader import fetch_data_pack
    from data.universe import get_universe_symbols

    parser = argparse.ArgumentParser(description="Run a headless backtest from engine.py")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--start", dest="start_date", default=None)
    parser.add_argument("--end", dest="end_date", default=None)
    args = parser.parse_args()

    config_path = os.path.join("config", "generated_strategies.json")
    with open(config_path, "r") as f:
        strategies_config = json.load(f)

    strat_configs = [s for s in strategies_config if s.get("name") == args.strategy]
    if not strat_configs:
        print(f"ERROR: Strategy '{args.strategy}' not found.")
        return 1

    strategies = load_strategies(strat_configs)
    symbols = get_universe_symbols("RUSSELL3000")
    
    data = fetch_data_pack(symbols, days=5040, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=5040, backtest_mode=True) or {}
    
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    result = run_backtest(strategies, prepared, start_cash=100000.0, start_date=args.start_date, end_date=args.end_date, global_data=g_data)
    
    if isinstance(result, list):
        if not result: return 1
        result = result[0]

    print(f"Final Value: ${result['final_value']:,.2f} | Trades: {result['total_trades']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(_run_cli())
