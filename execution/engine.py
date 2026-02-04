from __future__ import annotations

import concurrent.futures
import json
import operator
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, SMAIndicator

# FIX: Add project root to path so we can import 'strategies'
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.strategy_loader import load_strategies
from execution.parity import (
    apply_strategy_score_multipliers,
    DEFAULT_SCORING_WEIGHTS,
)
from execution.shared_logic import _generic_exit_decision

# --- CONFIGURATION ---
MIN_BARS = 60
MIN_ENTRY_SCORE = 120.0
SUPER_SIGNAL_NAME = "SUPER SIGNAL (Wealth + Income)"

_ATR_HIGH_THRESH_PCT = 3.0
_ATR_MED_THRESH_PCT = 2.0
_VOL_REL_THRESH = 1.5
# AUDIT UPDATE: Tighter VCP threshold for Minervini compliance
_VCP_BB_WIDTH_THRESH = 0.15

_DEBUG_TRAIL_ACTIVATION = os.environ.get("APEX_DEBUG_TRAIL_ACTIVATION", "").strip() not in ("", "0", "false", "False")


def compute_stop_fill(open_px: float, high_px: float, trigger_px: float, stop_limit_pct: Optional[float]) -> Tuple[bool, float]:
    """
    Daily-bar approximation for buy-stop-limit fills.
    - If the day gaps above the limit, no fill.
    - If open >= trigger and open <= limit, fill at open.
    - If high >= trigger and open < trigger, fill at trigger.
    """
    try:
        open_val = float(open_px)
        high_val = float(high_px)
        trig = float(trigger_px)
    except Exception:
        return False, float("nan")

    if not np.isfinite(open_val) or not np.isfinite(high_val) or not np.isfinite(trig):
        return False, float("nan")
    if trig <= 0:
        return False, float("nan")

    if stop_limit_pct is None:
        limit_mult = 1.0
    else:
        try:
            limit_mult = 1.0 + float(stop_limit_pct)
        except Exception:
            limit_mult = 1.0
    limit_px = trig * limit_mult

    if open_val >= limit_px:
        return False, float("nan")
    if open_val >= trig:
        return True, open_val
    if high_val >= trig:
        return True, trig
    return False, float("nan")


def get_sector(symbol: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"


_SECTOR_MAP: Dict[str, str] | None = None


def _load_sector_map() -> Dict[str, str]:
    global _SECTOR_MAP
    if _SECTOR_MAP is not None:
        return _SECTOR_MAP
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "sectors.json"))
    try:
        with open(path, "r") as f:
            data = json.load(f)
        _SECTOR_MAP = {str(k).upper(): str(v) for k, v in (data or {}).items()}
    except Exception:
        _SECTOR_MAP = {}
    return _SECTOR_MAP


def _compute_sector_rs(enriched: Dict[str, "_SymbolArrays"], all_dates: np.ndarray) -> Dict[str, np.ndarray]:
    sector_map = _load_sector_map()
    n_dates = len(all_dates)
    if n_dates == 0:
        return {}

    sector_sum: Dict[str, np.ndarray] = {}
    sector_cnt: Dict[str, np.ndarray] = {}
    for sym, sd in enriched.items():
        sec = sector_map.get(sym.upper())
        if not sec:
            sec = get_sector(sym)
        if not sec:
            continue
        if sec not in sector_sum:
            sector_sum[sec] = np.zeros(n_dates, dtype=np.float32)
            sector_cnt[sec] = np.zeros(n_dates, dtype=np.int32)
        close = sd.close
        gidx = sd.gidx
        valid = np.isfinite(close)
        if not np.any(valid):
            continue
        np.add.at(sector_sum[sec], gidx[valid], close[valid].astype(np.float32))
        np.add.at(sector_cnt[sec], gidx[valid], 1)

    if not sector_sum:
        return {}

    sector_names = list(sector_sum.keys())
    ret_mat = np.full((n_dates, len(sector_names)), np.nan, dtype=np.float32)
    lookback = 63  # ~3 months
    for j, sec in enumerate(sector_names):
        sum_arr = sector_sum[sec]
        cnt_arr = sector_cnt[sec]
        with np.errstate(divide="ignore", invalid="ignore"):
            close = np.where(cnt_arr > 0, sum_arr / cnt_arr, np.nan)
        shifted = np.roll(close, lookback)
        shifted[:lookback] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (close / shifted) - 1.0
        ret_mat[:, j] = ret

    sector_rs = {sec: np.zeros(n_dates, dtype=np.float32) for sec in sector_names}
    for i in range(n_dates):
        row = ret_mat[i]
        m = np.isfinite(row)
        k = int(m.sum())
        if k < 1:
            continue
        vals = row[m]
        order = np.argsort(vals, kind="quicksort")
        ranks = np.empty_like(order, dtype=np.int32)
        ranks[order] = np.arange(k, dtype=np.int32)
        pct = (ranks / (k - 1)) if k > 1 else np.zeros(k, dtype=np.float32)
        rating = 1.0 + 98.0 * pct
        idxs = np.where(m)[0]
        for idx, r in zip(idxs, rating):
            sec = sector_names[idx]
            sector_rs[sec][i] = r
    return sector_rs


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
        min_names = 1
    
    # Build matrix: rows=dates, cols=symbols
    close_mat = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    for j, sym in enumerate(syms):
        sd = enriched[sym]
        valid_mask = (sd.gidx >= 0) & (sd.gidx < n_days)
        close_mat[sd.gidx[valid_mask], j] = sd.close[valid_mask].astype(np.float32)

    tmp = np.zeros((n_days, n_syms), dtype=np.float32)
    momentum_rank = np.zeros((n_days, n_syms), dtype=np.float32)
    
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

    # Momentum rank based on 6-month return
    mom_lb = 126
    shifted_mom = np.roll(close_mat, mom_lb, axis=0)
    shifted_mom[:mom_lb, :] = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        mom_ret = (close_mat / shifted_mom) - 1.0
    mom_score = np.where(enough & np.isfinite(mom_ret), mom_ret, np.nan)
    
    chunk = 250
    for start in range(0, n_days, chunk):
        end = min(n_days, start + chunk)
        block = score[start:end, :]
        mom_block = mom_score[start:end, :]
        
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

            mom_row = mom_block[i]
            mm = np.isfinite(mom_row)
            km = int(mm.sum())
            if km >= min_names:
                vals_m = mom_row[mm]
                order_m = np.argsort(vals_m, kind="quicksort")
                ranks_m = np.empty_like(order_m, dtype=np.int32)
                ranks_m[order_m] = np.arange(km, dtype=np.int32)
                pct_m = (ranks_m / (km - 1)) if km > 1 else np.zeros(km, dtype=np.float32)
                momentum_rank[start + i, mm] = 1.0 + 98.0 * pct_m

    for j, sym in enumerate(syms):
        sd = enriched[sym]
        valid_indices = (sd.gidx >= 0) & (sd.gidx < n_days)
        if np.any(valid_indices):
            sd.rsrating[valid_indices] = rs_rating[sd.gidx[valid_indices], j]
            sd.momrank[valid_indices] = momentum_rank[sd.gidx[valid_indices], j]


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

        # Qullamaggie ADR (avg of (high/low - 1) * 100)
        with np.errstate(divide="ignore", invalid="ignore"):
            adr_q = (df["high"] / df["low"].replace(0, np.nan) - 1.0) * 100.0
        df["adr_pct_q"] = adr_q.rolling(20).mean().fillna(0.0).replace([np.inf, -np.inf], 0.0)

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
        df["true_range"] = tr
        df["atr14"] = tr.rolling(14).mean()
        df["tr_ma50"] = tr.rolling(50).mean()
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

        # 1/3/6-month returns (approx trading days)
        df["ret_1m"] = df["close"].pct_change(21) * 100.0
        df["ret_3m"] = df["close"].pct_change(63) * 100.0
        df["ret_6m"] = df["close"].pct_change(126) * 100.0
        
        with np.errstate(divide="ignore", invalid="ignore"):
            df["clv"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
        df["clv"] = df["clv"].fillna(0.5)

        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        
        df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
        df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ma50"] = df["volume"].rolling(50).mean()
        df["vol_ma10"] = df["volume"].rolling(10).mean()
        df["vol_dryup"] = (df["vol_ma10"] < df["vol_ma50"]).astype(float)

        df["highest10"] = df["high"].rolling(10).max()
        df["highest10_1"] = df["highest10"].shift(1) 

        if spy_df is not None and not spy_df.empty:
            spy_aligned = spy_df["close"].reindex(df.index).ffill().bfill()
            df["rs_ratio"] = df["close"] / spy_aligned
            df["spy_close"] = spy_aligned
            df["spy_sma20"] = spy_aligned.rolling(20).mean()
            df["spy_sma50"] = spy_aligned.rolling(50).mean()
            df["rs_ratio_sma50"] = df["rs_ratio"].rolling(50).mean()
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
            df["rs_ratio_sma50"] = np.nan

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
        # AUDIT FIX: Allow IPOs (<252 days) to have valid 52w Highs
        df["high_52w"] = df["high"].rolling(252, min_periods=1).max()
        df["low_52w"] = df["low"].rolling(252, min_periods=1).min()

        with np.errstate(divide="ignore", invalid="ignore"):
            df["pct_off_high_52w"] = ((df["high_52w"] - df["close"]) / df["high_52w"]) * 100.0
            df["pct_above_low_52w"] = ((df["close"] - df["low_52w"]) / df["low_52w"]) * 100.0
        df["pct_off_high_52w"] = df["pct_off_high_52w"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        df["pct_above_low_52w"] = df["pct_above_low_52w"].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # AUDIT FIX: Add Breakout Trigger (Donchian High)
        # This detects if price is breaking out of the VCP base.
        high_20 = df["high"].rolling(window=20).max()
        df["high_20"] = high_20
        # Also shift it so we compare today's close vs yesterday's high (true breakout)
        df["high_20_prev"] = high_20.shift(1)
        
        # AUDIT FIX: Fill NaN slope with 0.0 to prevent hard gates from rejecting all trades
        df["sma200_slope"] = df["sma200"].diff(22).fillna(0.0)
        df["sma200_slope_1m"] = df["sma200_slope"]
        
        df["std_20"] = df["close"].rolling(20).std()
        df["bb_width"] = (4 * df["std_20"]) / (df["close"].rolling(20).mean() + 1e-9)

        # Range contraction proxies for VCP
        def _range_pct(window: int) -> pd.Series:
            hi = df["high"].rolling(window).max()
            lo = df["low"].rolling(window).min()
            return ((hi - lo) / lo.replace(0, np.nan)) * 100.0

        df["range_pct_5"] = _range_pct(5)
        df["range_pct_10"] = _range_pct(10)
        df["range_pct_20"] = _range_pct(20)
        df["range_pct_40"] = _range_pct(40)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["vcp_tightness"] = df["range_pct_5"] / df["range_pct_20"].replace(0, np.nan)
        df["vcp_tightness"] = df["vcp_tightness"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        
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
    # AUDIT FIX: REWARD TIGHTNESS (Low Volatility)
    if natr < 1.5:
        score += 30.0
    elif natr < 2.5:
        score += 15.0
    elif natr > 3.0:
        # score -= 100.0 # Disabled: Allow high volatility for "Operation Leverage"
        pass
        
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
    momrank: np.ndarray
    adr_pct: np.ndarray
    prev_high: np.ndarray
    highest10_1: np.ndarray
    gap_pct: np.ndarray # Start with gap logic
    clv: np.ndarray
    trend_mask: np.ndarray # AUDIT FIX: Vectorized Gate
    adx: np.ndarray
    rs_ratio: np.ndarray
    rs_ratio_sma50: np.ndarray


@dataclass(slots=True)
class _Candidate:
    sym: str
    entry_px: float
    stop_px: float
    score: float
    strategy_name: str
    entry_i: int
    stop_limit_pct: float = 0.0
    size_scalar: float = 1.0
    signal_mode: str = ""
    entry_type: str = ""


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
            sma20_arr = _get_np_col(df, "sma20", np.nan, length=n)
            gap_pct_arr = _get_np_col(df, "gap_pct", 0.0, length=n)
            slope_arr = _get_np_col(df, "sma200_slope", 0.0, length=n)
            high52_arr = _get_np_col(df, "high_52w", np.nan, length=n)
            low52_arr = _get_np_col(df, "low_52w", np.nan, length=n)
            bb_w_arr = _get_np_col(df, "bb_width", 100.0, length=n)
            bb_w_arr = _get_np_col(df, "bb_width", 100.0, length=n)
            natr_arr = _get_np_col(df, "natr", 100.0, length=n)
            
            # Phase 4 Upgrades
            adx_arr = _get_np_col(df, "adx", 0.0, length=n)
            rs_ratio_arr = _get_np_col(df, "rs_ratio", 0.0, length=n)
            rs_ratio_sma50_arr = _get_np_col(df, "rs_ratio_sma50", 0.0, length=n)

            if not np.isfinite(low_arr).any():
                low_arr = open_arr
            if not np.isfinite(high_arr).any():
                high_arr = open_arr

            # AUDIT FIX: UNLOCK DATA - Allow ALL bars initially, filter in Strategy later.
            trend_mask = np.ones(n, dtype=bool)

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
                sma20=sma20_arr,
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
                momrank=np.zeros(n, dtype=np.float64),
                adr_pct=_get_np_col(df, "adr_pct", 0.0, length=n),
                prev_high=_get_np_col(df, "prev_high", 0.0, length=n),
                highest10_1=_get_np_col(df, "highest10_1", 0.0, length=n),
                gap_pct=gap_pct_arr,
                clv=_get_np_col(df, "clv", 0.5, length=n),
                trend_mask=trend_mask,
                adx=adx_arr,
                rs_ratio=rs_ratio_arr,
                rs_ratio_sma50=rs_ratio_sma50_arr
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
            sym_data.df["momentum_rank"] = sym_data.momrank
        except Exception:
            continue

    # --- INJECT SECTOR RELATIVE STRENGTH ---
    sector_rs_map = _compute_sector_rs(enriched, all_dates)
    sector_map = _load_sector_map()
    for sym, sym_data in enriched.items():
        try:
            sec = sector_map.get(sym.upper()) or get_sector(sym)
            rs_arr = sector_rs_map.get(sec)
            if rs_arr is None:
                sym_data.df["sector_rs"] = 50.0
            else:
                sym_data.df["sector_rs"] = rs_arr[sym_data.gidx]
        except Exception:
            sym_data.df["sector_rs"] = 50.0

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
    # --- SILENT MODE (OPTIMIZATION SPEED) ---
    # Logs disabled to prevent terminal crashes and maximize CPU for math.
    def DBG(msg: str) -> None:
        pass
    # ----------------------------------------

    if hasattr(data, "enriched"):
        pre_calculated_data = data
        data = None

    def _unwrap_genome(params: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict) and isinstance(params.get("genome"), dict):
            return params["genome"]
        return params

    def _flatten_params(raw_params: Dict[str, Any]) -> Dict[str, Any]:
        params = _unwrap_genome(raw_params) or {}
        if not isinstance(params, dict):
            return {}
        params = dict(params)
        # AUDIT FIX: Flatten nested parameters so engine sees risk/execution settings
        params.update(params.get("risk_parameters", {}) or {})
        params.update(params.get("execution_parameters", {}) or {})
        return params

    _OPS = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
    }

    def _resolve_rule_value(row: pd.Series, rule: Dict[str, Any]) -> float:
        if "ref" in rule:
            base = row.get(rule.get("ref"), np.nan)
            try:
                base = float(base)
            except Exception:
                return float("nan")
            mult = rule.get("mult")
            if mult is None and "val" in rule:
                mult = rule.get("val", 1.0)
            if mult is not None:
                try:
                    base *= float(mult)
                except Exception:
                    return float("nan")
            return base
        if "val" in rule:
            try:
                return float(rule.get("val"))
            except Exception:
                return float("nan")
        return float("nan")

    def _rule_pass(row: pd.Series, rule: Dict[str, Any]) -> bool:
        col = rule.get("col")
        op = _OPS.get(rule.get("op"))
        if not col or op is None:
            return False
        val_a = row.get(col, np.nan)
        try:
            val_a = float(val_a)
        except Exception:
            return False
        val_b = _resolve_rule_value(row, rule)
        if not np.isfinite(val_a) or not np.isfinite(val_b):
            return False
        return bool(op(val_a, val_b))

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

    debug_counts = {
        "n_universe": np.full(len(all_dates), len(enriched), dtype=np.int32),
        "n_trend": np.zeros(len(all_dates), dtype=np.int32),
        "n_rs": np.zeros(len(all_dates), dtype=np.int32),
        "n_vcp": np.zeros(len(all_dates), dtype=np.int32),
    }

    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]

    compiled_strategies = []
    for strat in strategies:
        raw_params = getattr(strat, "params", getattr(strat, "genome", {})) or {}
        params = _flatten_params(raw_params)
        w = _ScoreWeights(1.0, 50.0, 20.0, 30.0)
        base_stop_mult = float(params.get("stop_loss_atr_bull", params.get("stop_loss_atr", 3.0)) or 3.0)
        compiled_strategies.append((strat, w, params, base_stop_mult))

    global_spy_close = np.zeros(len(all_dates), dtype=np.float64)
    global_spy_sma200 = np.zeros(len(all_dates), dtype=np.float64)
    global_spy_sma150 = np.zeros(len(all_dates), dtype=np.float64)

    if global_data and "SPY" in global_data:
        spy_df_raw = global_data["SPY"]
        if not spy_df_raw.empty:
            spy_df_raw = spy_df_raw.copy()
            spy_df_raw.columns = spy_df_raw.columns.str.lower()
            if "sma200" not in spy_df_raw.columns:
                 spy_df_raw["sma200"] = spy_df_raw["close"].rolling(200).mean()
            if "sma150" not in spy_df_raw.columns:
                 spy_df_raw["sma150"] = spy_df_raw["close"].rolling(150).mean()
            spy_aligned = spy_df_raw.reindex(all_dates).ffill().bfill()
            global_spy_close = spy_aligned["close"].fillna(0).to_numpy(dtype=np.float64)
            global_spy_sma200 = spy_aligned["sma200"].fillna(0).to_numpy(dtype=np.float64)
            global_spy_sma150 = spy_aligned["sma150"].fillna(0).to_numpy(dtype=np.float64)


    # --- MAIN LOOP (Optimized) ---
    batch_size = 0
    if compiled_strategies:
        try:
            batch_size = int(compiled_strategies[0][2].get("batch_size", 0) or 0)
        except Exception:
            batch_size = 0
    items = list(enriched.items())
    if batch_size and batch_size > 0:
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
    else:
        batches = [items]

    for batch in batches:
        for sym, sd in batch:
            n_bars = len(sd.index)
            if n_bars < 2:
                continue

            # UNLOCK: evaluate all possible setup days
            setup_indices = np.arange(0, n_bars - 1, dtype=np.int32)
            if len(setup_indices) < 1:
                DBG(f"{sym}: REJECTED - Trend Mask Empty")
                continue

            for prev_i in setup_indices:
                curr_i = prev_i + 1

                # Boundary Safety
                if curr_i >= n_bars:
                    continue

                day_idx = sd.gidx[curr_i]
                if day_idx < 1 or day_idx >= len(all_dates):
                    continue

                debug_counts["n_trend"][day_idx] += 1

                rs_rating = float(sd.rsrating[prev_i])
                if not np.isfinite(rs_rating) or rs_rating <= 0:
                    # Fail-open if RS rating was not computed (diagnostic safety).
                    rs_rating = 99.0
                else:
                    debug_counts["n_rs"][day_idx] += 1

                for strat, w, params, base_stop_mult in compiled_strategies:
                    # Optional market regime filter (per strategy)
                    exposure_mode = str(params.get("market_exposure_mode", "")).lower()
                    regime_filter = bool(params.get("regime_filter", False))
                    market_mode = str(params.get("market_filter_mode", "")).lower()
                    if market_mode in {"traffic_light", "spy_sma200", "sma200"}:
                        regime_filter = True
                    # Hybrid exposure skips hard regime gating
                    if exposure_mode in {"hybrid", "scaled", "exposure"}:
                        regime_filter = False
                    if regime_filter:
                        spy_c = global_spy_close[day_idx - 1]
                        spy_200 = global_spy_sma200[day_idx - 1]
                        if spy_c > 0 and spy_200 > 0 and spy_c < spy_200:
                            continue

                    # Optional trend mode gate
                    mode = params.get("trend_mode")
                    if mode:
                        price_yesterday = float(sd.close[prev_i])
                        sma50_val = float(sd.sma50[prev_i])
                        sma200_val = float(sd.sma200[prev_i])
                        sma150_val = float(sd.sma150[prev_i])
                        if mode == "sma50" and price_yesterday < sma50_val:
                            continue
                        if mode == "sma200" and price_yesterday < sma200_val:
                            continue
                        if mode == "strict":
                            if not (price_yesterday > sma50_val and sma50_val > sma150_val and sma150_val > sma200_val):
                                continue

                    # Optional RS floor
                    min_rs = params.get("rs_floor")
                    if min_rs is not None and rs_rating < float(min_rs):
                        continue

                    # Optional RS ratio trend gate
                    if bool(params.get("use_rs_ratio_gate", False)):
                        if sd.rs_ratio[prev_i] < sd.rs_ratio_sma50[prev_i]:
                            continue

                    # Optional ADX gate
                    adx_min = params.get("adx_min")
                    if adx_min is not None and sd.adx[prev_i] < float(adx_min):
                        continue

                    # Optional BB width gate
                    max_bb = params.get("bb_width_max")
                    if max_bb is not None and sd.bbwidth[prev_i] > float(max_bb):
                        continue

                    # Optional volume gate (signal day)
                    vol_mult = params.get("vol_mult")
                    if vol_mult is not None:
                        vol_today = float(sd.volume[prev_i])
                        vol_avg = float(sd.volma50[prev_i])
                        if vol_today < (vol_avg * float(vol_mult)):
                            continue

                    # VCP candidate density (base structure) — debug only
                    row = sd.df.iloc[prev_i]
                    try:
                        rp5 = float(row.get("range_pct_5", np.nan))
                        rp10 = float(row.get("range_pct_10", np.nan))
                        rp20 = float(row.get("range_pct_20", np.nan))
                        rp40 = float(row.get("range_pct_40", np.nan))
                        base_depth = float(row.get("range_pct_20", np.nan))
                        last_contraction = min(rp5, rp10)
                        contractions = 0
                        if np.isfinite(rp10) and rp10 <= 20.0:
                            contractions += 1
                        if np.isfinite(rp20) and rp20 <= 25.0:
                            contractions += 1
                        if np.isfinite(rp40) and rp40 <= 30.0:
                            contractions += 1
                        vol_prev = float(sd.volume[prev_i - 1]) if prev_i > 0 else np.nan
                        vol_ma50_prev = float(sd.volma50[prev_i - 1]) if prev_i > 0 else np.nan
                        vcp_candidate = (
                            np.isfinite(last_contraction)
                            and last_contraction <= 10.0
                            and np.isfinite(base_depth)
                            and base_depth <= 30.0
                            and contractions >= 2
                            and (
                                not np.isfinite(vol_ma50_prev)
                                or vol_ma50_prev <= 0
                                or vol_prev <= (vol_ma50_prev * 0.75)
                            )
                        )
                        if vcp_candidate:
                            rs_floor = params.get("rs_min", params.get("rs_floor", 0.0)) or 0.0
                            if rs_floor <= 0 or rs_rating >= float(rs_floor):
                                debug_counts["n_vcp"][day_idx] += 1
                    except Exception:
                        pass

                    # Strategy-specific entry logic
                    decision = strat.entry(sd.df, prev_i)
                    if not decision:
                        continue

                    row = sd.df.iloc[prev_i]
                    trigger = None
                    stop_px = None
                    stop_limit_pct = params.get("stop_limit_pct")
                    entry_day_idx = day_idx
                    entry_i = curr_i
                    cand_signal_mode = str(params.get("signal_mode", "after_close"))
                    entry_type = ""

                    if isinstance(decision, dict):
                        trigger = decision.get("trigger_price")
                        stop_px = decision.get("stop_price")
                        if stop_limit_pct is None:
                            stop_limit_pct = decision.get("stop_limit_pct")
                        entry_type = str(decision.get("entry_type", "") or "")
                        entry_timing = decision.get("entry_timing") or decision.get("signal_mode")
                        if isinstance(entry_timing, str):
                            timing = entry_timing.lower()
                            if timing in {"same_day", "same_day_open", "same_day_close"}:
                                entry_i = prev_i
                                entry_day_idx = sd.gidx[prev_i]
                                if timing == "same_day_close":
                                    cand_signal_mode = "close"
                                else:
                                    cand_signal_mode = "open"
                            elif timing in {"open", "close"}:
                                cand_signal_mode = timing

                    if trigger is None:
                        stop_buy_ref = params.get("stop_buy_ref")
                        if stop_buy_ref:
                            pivot = row.get(stop_buy_ref, np.nan)
                            if np.isfinite(pivot):
                                trigger = float(pivot) * float(params.get("stop_buy_mult", 1.0))
                    if trigger is None:
                        trigger = float(row.get("close", np.nan))

                    if stop_px is None:
                        stop_type = str(params.get("stop_loss_type", "atr")).lower()
                        if "low" in stop_type:
                            stop_px = float(row.get("low", np.nan))
                        else:
                            atr = float(row.get("atr14", 0.0) or 0.0)
                            stop_px = float(trigger) - (atr * base_stop_mult)

                    if not np.isfinite(trigger) or not np.isfinite(stop_px) or stop_px <= 0:
                        continue

                    max_stop_pct = params.get("max_stop_pct")
                    if isinstance(decision, dict) and decision.get("max_stop_pct") is not None:
                        max_stop_pct = decision.get("max_stop_pct")
                    if max_stop_pct is not None:
                        stop_width = (float(trigger) - float(stop_px)) / float(trigger)
                        if stop_width > float(max_stop_pct):
                            continue

                    if stop_limit_pct is None:
                        stop_limit_pct = 0.02

                    score_mode = str(params.get("score_mode", "")).lower()
                    if score_mode == "momentum":
                        rs_val = float(sd.rsrating[prev_i])
                        mom_val = float(sd.momrank[prev_i])
                        if not np.isfinite(rs_val):
                            rs_val = 0.0
                        if not np.isfinite(mom_val):
                            mom_val = 0.0
                        score = (0.7 * rs_val) + (0.3 * mom_val)
                    else:
                        score = _score_row_dual_core(
                            sd.rsi14[prev_i], sd.bbwidth[prev_i], sd.natr[prev_i],
                            sd.close[prev_i], sd.high52w[prev_i], {"rsi_factor": 1.0}
                        )

                    if entry_day_idx < 0 or entry_day_idx >= len(all_dates):
                        continue
                    candidates_by_day[entry_day_idx].append(
                        _Candidate(
                            sym,
                            float(trigger),
                            float(stop_px),
                            score,
                            strat.name,
                            entry_i,
                            float(stop_limit_pct),
                            1.0,
                            str(cand_signal_mode),
                            entry_type,
                        )
                    )
                    curr_date_str = str(all_dates[entry_day_idx])[:10]
                    DBG(f"[{curr_date_str}] {sym}: ACCEPTED (Score: {score:.1f})")

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
        trade_outcomes = []
        
        params = _flatten_params(getattr(strat, "params", getattr(strat, "genome", {})) or {})
        max_pos = int(params.get("max_positions", 10) or 10)
        risk_per_trade = float(params.get("risk_per_trade", 0.01) or 0.01)
        max_pos_size_pct = float(params.get("max_pos_size_pct", 0.30) or 0.30)
        partial_profit_day = int(params.get("partial_profit_day", 4) or 4)
        partial_profit_r = float(params.get("partial_profit_r", 2.0))

        for day_idx, candidates in enumerate(candidates_by_day):
            # AUDIT FIX: Respect Start/End dates
            current_dt_np = all_dates[day_idx]
            if start_ts and current_dt_np < start_ts.to_datetime64(): continue
            if end_ts and current_dt_np > end_ts.to_datetime64(): break

            # Market exposure regime (hybrid scaling)
            exposure_mode = str(params.get("market_exposure_mode", "")).lower()
            spy_c = global_spy_close[day_idx]
            spy_200 = global_spy_sma200[day_idx]
            market_is_bull = True
            if spy_c > 0 and spy_200 > 0:
                market_is_bull = spy_c >= spy_200

            max_pos_today = max_pos
            if exposure_mode in {"hybrid", "scaled", "exposure"} and not market_is_bull:
                bear_max = int(params.get("bear_max_positions", 2) or 2)
                max_pos_today = max(1, min(max_pos, bear_max))

            stop_loss_atr_bull = float(params.get("stop_loss_atr_bull", params.get("stop_loss_atr", 3.0)) or 3.0)
            stop_loss_atr_bear = float(params.get("stop_loss_atr_bear", params.get("bear_stop_loss_atr", 0.5)) or 0.5)

            base_max_total = float(params.get("max_total_exposure_pct", 1.0) or 1.0)
            max_total_bull = float(params.get("max_total_exposure_pct_bull", base_max_total) or base_max_total)
            max_total_bear = float(params.get("max_total_exposure_pct_bear", base_max_total) or base_max_total)
            max_total_exposure_pct = max_total_bull if market_is_bull else max_total_bear
            allow_margin = bool(params.get("allow_margin", False) or max_total_exposure_pct > 1.0 or max_pos_size_pct > 1.0)

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
                current_open = float(sym_data.open[loc])
                current_high = float(sym_data.high[loc])

                # Execute pending pyramid add (next-day at open)
                if pos.get("pyramid_pending") and pos.get("pyramid_pending_day") is not None:
                    if day_idx > int(pos.get("pyramid_pending_day")):
                        pos["pyramid_pending"] = False
                        add_fraction = float(pos.get("pyramid_fraction", 0.5) or 0.5)
                        add_shares = int(pos.get("shares", 0) * add_fraction)
                        if add_shares > 0 and len(positions) < max_pos_today:
                            mtm_equity = cash + sum(p["shares"] * p.get("last_price", 0.0) for p in positions.values())
                            if mtm_equity <= 0:
                                mtm_equity = cash
                            max_cap = mtm_equity * max_pos_size_pct
                            current_value = pos.get("shares", 0) * current_open
                            remaining_cap = max_cap - current_value
                            if remaining_cap > 0:
                                cap_shares = int(remaining_cap / current_open)
                                add_shares = min(add_shares, cap_shares)

                            gross_exposure = sum(p["shares"] * p.get("last_price", 0.0) for p in positions.values())
                            max_gross = mtm_equity * max_total_exposure_pct
                            remaining_gross = max_gross - gross_exposure
                            if remaining_gross > 0:
                                cap_gross_shares = int(remaining_gross / current_open)
                                add_shares = min(add_shares, cap_gross_shares)

                            if add_shares > 0:
                                add_cost = add_shares * current_open
                                if (not allow_margin) and add_cost > cash:
                                    add_shares = int(cash / current_open)
                                    add_cost = add_shares * current_open

                            if add_shares > 0:
                                cash -= add_cost
                                old_shares = pos.get("shares", 0)
                                old_cost = old_shares * pos.get("entry_price", current_open)
                                new_total_shares = old_shares + add_shares
                                if new_total_shares > 0:
                                    pos["entry_price"] = (old_cost + add_cost) / new_total_shares
                                    pos["shares"] = new_total_shares
                                    if pos.get("pyramid_stop_to_avg_cost", True):
                                        pos["stop_price"] = pos["entry_price"]
                                    pos["pyramids"] = int(pos.get("pyramids", 0) or 0) + 1
                                    pos["initial_risk"] = max(pos["entry_price"] - pos["stop_price"], pos["entry_price"] * 0.001)
                                    trades_list.append({
                                        "Symbol": sym, "Entry": pos["entry_price"], "Exit": current_open,
                                        "PnL": 0.0,
                                        "Return %": 0.0,
                                        "Reason": "PYRAMID_ADD"
                                    })
                
                # Exit Logic
                should_exit = False
                exit_px = current_close
                reason = None
                
                # Hybrid regime: tighten stops in bear markets instead of forcing liquidation
                if exposure_mode in {"hybrid", "scaled", "exposure"} and not market_is_bull:
                    atr_val = float(sym_data.atr14[loc] or 0.0)
                    if atr_val > 0:
                        tight_stop = current_close - (atr_val * stop_loss_atr_bear)
                        if np.isfinite(tight_stop) and tight_stop > pos["stop_price"]:
                            pos["stop_price"] = tight_stop

                # Optional forced regime exit (legacy behavior)
                enforce_regime_exit = bool(params.get("regime_exit", False))
                if exposure_mode in {"filter", "hard"}:
                    enforce_regime_exit = True
                if enforce_regime_exit:
                    regime_ma_type = params.get("regime_ma", "sma200")
                    regime_threshold = global_spy_sma150[day_idx] if regime_ma_type == "sma150" else global_spy_sma200[day_idx]
                    if global_spy_close[day_idx] < regime_threshold:
                        should_exit = True
                        exit_px = current_close
                        reason = "MARKET_REGIME_EXIT"

                if not should_exit:
                    # SQUAT EXIT: If Day 1 Close < Pivot, Exit Immediately at Open
                    # We are on Day 2 (day_idx). Entry was Day 1 (entry_day_idx).
                    if (day_idx - pos["entry_day_idx"]) == 1:
                        if loc > 0:
                            entry_day_close = float(sym_data.close[loc - 1])
                            pivot_val = pos.get("pivot", -1.0)
                            if pivot_val > 0 and entry_day_close < pivot_val:
                                should_exit = True
                                exit_px = float(sym_data.open[loc]) # Exit at Open
                if not should_exit:
                    # TIME STOP: Dynamic patience window
                    days_held = day_idx - int(pos.get("entry_day_idx", day_idx))
                    time_stop_days = int(params.get("time_stop_days", params.get("time_stop", 7)) or 7)
                    if days_held >= time_stop_days:
                        profit_pct = (current_close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0
                        if profit_pct < 0.005:
                            should_exit = True
                            exit_px = current_close
                            reason = "TIME_STOP"
                if not should_exit:
                    # --- PARTIAL PROFIT LOGIC (PATCHED) ---
                    # 1. Extract Flags (Default to False for safety)
                    pp_mode = str(params.get("partial_profit_mode", "")).lower()
                    enable_pp = bool(params.get("enable_partial_profit", False))
                    if pp_mode in {"time", "days"}:
                        enable_pp = True
                    if pp_mode in {"none", "off"}:
                        enable_pp = False

                    move_be = bool(params.get("move_stop_to_be", True))
                    breakeven_at = float(params.get("breakeven_at_pct", 0.0) or 0.0)
                    pp_day = int(params.get("partial_profit_after_days", params.get("partial_profit_day", 0)) or 0)
                    pp_r = float(params.get("partial_profit_r", 2.0))
                    pp_frac = float(params.get("partial_profit_fraction", 0.5))
                    pp_pct = float(params.get("partial_profit_pct", 0.0) or 0.0)
                    
                    # 2. Gate Execution
                    risk_per_share = float(pos.get("initial_risk", 0.0) or 0.0)
                    days_held = day_idx - int(pos.get("entry_day_idx", day_idx))
                    
                    if enable_pp and risk_per_share > 0 and not pos.get("partial_taken", False):
                        # Time Gate: Must hold for at least X days
                        if days_held >= pp_day:
                            if pp_mode in {"time", "days"}:
                                if current_close <= pos["entry_price"]:
                                    pass
                                else:
                                    shares_to_sell = max(1, int(pos["shares"] * pp_frac))
                                    shares_to_sell = min(shares_to_sell, pos["shares"])

                                    proceeds = shares_to_sell * current_close
                                    cash += proceeds
                                    pos["shares"] -= shares_to_sell

                                    trades_list.append({
                                        "Symbol": sym, "Entry": pos["entry_price"], "Exit": current_close,
                                        "PnL": proceeds - (shares_to_sell * pos["entry_price"]),
                                        "Return %": (current_close/pos["entry_price"] - 1)*100,
                                        "Reason": f"PARTIAL_TIME_{pp_day}D"
                                    })

                                    if move_be:
                                        profit_pct = (current_close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0
                                        if breakeven_at <= 0 or profit_pct >= breakeven_at:
                                            pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])

                                    pos["partial_taken"] = True
                            else:
                                if pp_pct > 0 and pp_mode in {"pct", "percent", "percentage"}:
                                    target_px = pos["entry_price"] * (1.0 + pp_pct)
                                    if current_close >= target_px:
                                        shares_to_sell = max(1, int(pos["shares"] * pp_frac))
                                        shares_to_sell = min(shares_to_sell, pos["shares"])

                                        proceeds = shares_to_sell * current_close
                                        cash += proceeds
                                        pos["shares"] -= shares_to_sell

                                        trades_list.append({
                                            "Symbol": sym, "Entry": pos["entry_price"], "Exit": current_close,
                                            "PnL": proceeds - (shares_to_sell * pos["entry_price"]),
                                            "Return %": (current_close/pos["entry_price"] - 1)*100,
                                            "Reason": f"PARTIAL_PCT_{pp_pct:.2f}"
                                        })

                                        if move_be:
                                            profit_pct = (current_close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0
                                            if breakeven_at <= 0 or profit_pct >= breakeven_at:
                                                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])

                                        pos["partial_taken"] = True
                                else:
                                    target_px = pos["entry_price"] + (pp_r * risk_per_share)

                                # Price Trigger
                                    if current_close >= target_px:
                                        # Execute Sell
                                        shares_to_sell = max(1, int(pos["shares"] * pp_frac))
                                        shares_to_sell = min(shares_to_sell, pos["shares"])
                                        
                                        proceeds = shares_to_sell * current_close
                                        cash += proceeds
                                        pos["shares"] -= shares_to_sell
                                    
                                        # Log
                                        trades_list.append({
                                            "Symbol": sym, "Entry": pos["entry_price"], "Exit": current_close,
                                            "PnL": proceeds - (shares_to_sell * pos["entry_price"]),
                                            "Return %": (current_close/pos["entry_price"] - 1)*100,
                                            "Reason": f"PARTIAL_PROFIT_{pp_r}R"
                                        })
                                    
                                        # 3. Optional: Move Stop to Breakeven
                                        if move_be:
                                            profit_pct = (current_close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0
                                            if breakeven_at <= 0 or profit_pct >= breakeven_at:
                                                pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
                                            
                                        pos["partial_taken"] = True

                if not should_exit:
                    # --- SPLIT EXIT LOGIC (Fast SMA partial, Slow SMA full) ---
                    split_exit = bool(params.get("split_exit", False) or params.get("exit_sma_fast") or params.get("exit_sma_slow"))
                    if split_exit and not pos.get("partial_taken", False):
                        fast_sma = params.get("exit_sma_fast") or params.get("exit_ma") or "sma20"
                        fast_val = float(sym_data.df.iloc[loc].get(fast_sma, 0.0))
                        if fast_val > 0 and current_close < fast_val:
                            frac = float(params.get("partial_profit_fraction", 0.5) or 0.5)
                            shares_to_sell = max(1, int(pos["shares"] * frac))
                            shares_to_sell = min(shares_to_sell, pos["shares"])
                            proceeds = shares_to_sell * current_close
                            cash += proceeds
                            pos["shares"] -= shares_to_sell
                            trades_list.append({
                                "Symbol": sym, "Entry": pos["entry_price"], "Exit": current_close,
                                "PnL": proceeds - (shares_to_sell * pos["entry_price"]),
                                "Return %": (current_close/pos["entry_price"] - 1)*100,
                                "Reason": f"PARTIAL_{fast_sma.upper()}"
                            })
                            # Optional breakeven stop after meaningful profit
                            move_be = bool(params.get("move_stop_to_be", True))
                            breakeven_at = float(params.get("breakeven_at_pct", 0.0) or 0.0)
                            if move_be:
                                profit_pct = (current_close - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0
                                if breakeven_at <= 0 or profit_pct >= breakeven_at:
                                    pos["stop_price"] = max(pos["stop_price"], pos["entry_price"])
                            pos["partial_taken"] = True

                if not should_exit:
                    # AUDIT FIX: Use the Strategy's sophisticated exit logic (Trailing Stops, SMA Breaks)
                    # AUDIT FIX: Translate Global Day Index to Local Data Index to prevent IPO crashes
                    # pos["entry_day_idx"] is Global (e.g., 2000), but this stock might only have 500 rows.
                    entry_loc_search = np.searchsorted(sym_data.gidx, [pos["entry_day_idx"]])
                    entry_loc = int(entry_loc_search[0]) if len(entry_loc_search) > 0 else 0

                    # Safety Clamp: Ensure we don't access out of bounds if data is misaligned
                    if entry_loc >= len(sym_data.close):
                        entry_loc = 0

                    exit_params = params
                    if params.get("exit_sma_slow") and not params.get("exit_ma_after_partial"):
                        exit_params = dict(params)
                        exit_params["exit_ma_after_partial"] = params.get("exit_sma_slow")

                    strat_exit, new_stop, target_px = _generic_exit_decision(
                        exit_params,
                        sym_data,
                        loc,
                        entry_loc,
                        pos["entry_price"],
                        pos["stop_price"],
                        partial_taken=pos.get("partial_taken", False)
                    )

                    if new_stop is not None and new_stop > pos["stop_price"]:
                        if params.get("use_trailing_stop", True):
                            pos["stop_price"] = new_stop

                    if strat_exit:
                        should_exit = True
                        if target_px is not None:
                            exit_px = target_px
                        elif current_low < pos["stop_price"]:
                            exit_px = min(float(sym_data.open[loc]), pos["stop_price"])
                        else:
                            exit_px = current_close
                        reason = "STRATEGY_EXIT"

                if not should_exit:
                    # Schedule pyramiding for next session (after-close decision)
                    pyramid_cfg = None
                    if hasattr(strat, "pyramid"):
                        try:
                            pyramid_cfg = strat.pyramid(sym_data.df, loc, pos)
                        except Exception:
                            pyramid_cfg = None
                    if pyramid_cfg is None:
                        threshold = float(params.get("pyramid_threshold", 0.0) or 0.0)
                        if threshold > 0:
                            entry_px = float(pos.get("entry_price", 0.0) or 0.0)
                            if entry_px > 0:
                                profit_pct = (current_close - entry_px) / entry_px
                                if profit_pct >= threshold:
                                    pyramid_cfg = {
                                        "add_fraction": float(params.get("pyramid_fraction", 0.5) or 0.5),
                                        "stop_to_avg_cost": bool(params.get("pyramid_stop_to_avg_cost", True)),
                                    }

                    if pyramid_cfg and len(positions) < max_pos_today:
                        max_adds = int(params.get("pyramid_max_adds", 1) or 1)
                        if int(pos.get("pyramids", 0) or 0) < max_adds and not pos.get("pyramid_pending", False):
                            pos["pyramid_pending"] = True
                            pos["pyramid_pending_day"] = day_idx
                            pos["pyramid_fraction"] = float(pyramid_cfg.get("add_fraction", 0.5) or 0.5)
                            pos["pyramid_stop_to_avg_cost"] = bool(pyramid_cfg.get("stop_to_avg_cost", True))
                
                if should_exit:
                    shares = pos["shares"]
                    proceeds = shares * exit_px
                    cash += proceeds
                    trade_outcomes.append(1 if (proceeds - (shares * pos["entry_price"])) > 0 else 0)
                    trades_list.append({
                        "Symbol": sym, "Entry": pos["entry_price"], "Exit": exit_px,
                        "PnL": proceeds - (shares * pos["entry_price"]),
                        "Return %": (exit_px/pos["entry_price"] - 1)*100,
                        "Reason": reason
                    })
                    to_remove.append(sym)
                    continue
                
                # Update MTM
                pos["last_price"] = current_close



            for sym in to_remove:
                del positions[sym]

            # 2. Enter New Trades
            # --- FIX: BYPASS NAME CHECK FOR SINGLE STRATEGY (GA MODE) ---
            # Prevents string mismatch bugs (e.g. Risk0.01 vs Risk0.010) from killing trades.
            # Minervini filters (RS/VCP) have already run upstream, so these candidates are valid.
            
            if len(strategies) == 1:
                day_candidates = candidates
            else:
                day_candidates = [c for c in candidates if c.strategy_name == strat.name]
            # -----------------------------------------------------------
            day_candidates.sort(key=lambda x: x.score, reverse=True)

            for cand in day_candidates:
                if len(positions) >= max_pos_today: break
                if cand.sym in positions: continue

                sym_data = enriched.get(cand.sym)
                if sym_data is None:
                    continue
                entry_loc = int(cand.entry_i)
                if entry_loc <= 0 or entry_loc >= len(sym_data.df):
                    continue

                open_px = float(sym_data.open[entry_loc])
                high_px = float(sym_data.high[entry_loc])
                close_px = float(sym_data.close[entry_loc])
                entry_type = str(getattr(cand, "entry_type", "") or "").lower()

                signal_mode = str(cand.signal_mode or params.get("signal_mode", "after_close")).lower()
                if signal_mode in {"market", "open", "moo"}:
                    filled, fill_px = True, open_px
                elif signal_mode in {"close", "moc"}:
                    filled, fill_px = True, close_px
                else:
                    filled, fill_px = compute_stop_fill(open_px, high_px, cand.entry_px, cand.stop_limit_pct)
                if not filled:
                    continue

                entry_px = float(fill_px)
                stop_px = float(cand.stop_px)
                if not np.isfinite(entry_px) or not np.isfinite(stop_px):
                    continue

                # Hybrid bear regime: tighten stop using ATR
                if exposure_mode in {"hybrid", "scaled", "exposure"} and not market_is_bull and entry_type != "ep":
                    atr_val = float(sym_data.atr14[entry_loc] or 0.0)
                    if atr_val > 0:
                        tight_stop = entry_px - (atr_val * stop_loss_atr_bear)
                        if np.isfinite(tight_stop) and tight_stop > stop_px:
                            stop_px = tight_stop

                if stop_px >= entry_px:
                    continue
                
                mtm_equity = cash + sum(p["shares"] * p["last_price"] for p in positions.values())

                # Progressive Exposure Rule
                if len(trade_outcomes) >= 10:
                    recent = trade_outcomes[-10:]
                    batting_avg = sum(recent) / len(recent)
                    size_scalar = 0.25 if batting_avg < 0.40 else 1.0
                else:
                    size_scalar = 1.0

                risk_amt = mtm_equity * risk_per_trade * size_scalar
                dist = max(entry_px - stop_px, entry_px * 0.005)
                shares = int(risk_amt / dist)
                
                # Caps (entry-type specific)
                local_max_pos = max_pos_size_pct
                if entry_type == "ep":
                    local_max_pos = min(local_max_pos, float(params.get("ep_max_pos_size_pct", 0.15) or 0.15))
                elif entry_type == "vcp":
                    local_max_pos = min(local_max_pos, float(params.get("vcp_max_pos_size_pct", 0.25) or 0.25))

                max_cap = mtm_equity * local_max_pos
                if shares * entry_px > max_cap:
                    shares = int(max_cap / entry_px)
                gross_exposure = sum(p["shares"] * p.get("last_price", 0.0) for p in positions.values())
                max_gross = mtm_equity * max_total_exposure_pct
                remaining_gross = max_gross - gross_exposure
                if remaining_gross <= 0:
                    shares = 0
                else:
                    max_gross_shares = int(remaining_gross / entry_px)
                    if shares > max_gross_shares:
                        shares = max_gross_shares

                if (not allow_margin) and shares * entry_px > cash:
                    shares = int(cash / entry_px)
                
                if shares == 0:
                    sym = cand.sym
                    DBG(f"{sym}: REJECTED - Zero Shares")
                    continue

                if shares > 0:
                    cash -= shares * entry_px
                    positions[cand.sym] = {
                        "entry_price": entry_px,
                        "stop_price": stop_px,
                        "shares": shares,
                        "entry_day_idx": day_idx,
                        "last_price": entry_px,
                        "partial_taken": False,
                        "pyramids": 0,
                        "pyramid_pending": False,
                        "initial_risk": max(entry_px - stop_px, entry_px * 0.001),
                        "pivot": cand.entry_px,
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

        # AUDIT FIX: Use Calendar Days for accurate CAGR, not Trading Days
        if start_date and end_date:
            total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
        elif len(all_dates) > 1:
            total_days = (pd.Timestamp(all_dates[-1]) - pd.Timestamp(all_dates[0])).days
        else:
            total_days = 0
        years = max(total_days / 365.25, 0.1)  # Avoid div/0
        years = max(total_days / 365.25, 0.1)  # Avoid div/0
        cagr = ((final_val / start_cash) ** (1 / years)) - 1
        
        # Calculate Max Drawdown from Equity Curve
        max_dd = 0.0
        if equity_curve:
            peaks = pd.Series([x["Equity"] for x in equity_curve]).cummax()
            drawdowns = (pd.Series([x["Equity"] for x in equity_curve]) - peaks) / peaks
            max_dd = abs(drawdowns.min()) if not drawdowns.empty else 0.0

        res = _empty_result(strat.name, start_cash, params)
        gate_audit = getattr(strat, "_gate_counts", None)
        if isinstance(gate_audit, dict):
            res["gate_audit"] = dict(gate_audit)
        res.update({
            "final_value": final_val,
            "max_drawdown_pct": max_dd,
            "total_trades": len(trades_list),
            "total_trades": len(trades_list),
            "hit_rate": win_rate,
            "cagr": cagr,
            "equity_curve": equity_curve,
            "trades_list": trades_list
        })
        final_results.append(res)

    if len(final_results) == 1:
        avg_universe = float(np.mean(debug_counts["n_universe"])) if len(all_dates) else 0.0
        avg_trend = float(np.mean(debug_counts["n_trend"])) if len(all_dates) else 0.0
        avg_rs = float(np.mean(debug_counts["n_rs"])) if len(all_dates) else 0.0
        avg_vcp = float(np.mean(debug_counts["n_vcp"])) if len(all_dates) else 0.0
        print(f"Avg Universe: {avg_universe:.1f}")
        print(f"Avg Trend Candidates: {avg_trend:.1f}")
        print(f"Avg RS Candidates: {avg_rs:.1f}")
        print(f"Avg VCP Candidates: {avg_vcp:.2f}")
        return final_results[0]
    avg_universe = float(np.mean(debug_counts["n_universe"])) if len(all_dates) else 0.0
    avg_trend = float(np.mean(debug_counts["n_trend"])) if len(all_dates) else 0.0
    avg_rs = float(np.mean(debug_counts["n_rs"])) if len(all_dates) else 0.0
    avg_vcp = float(np.mean(debug_counts["n_vcp"])) if len(all_dates) else 0.0
    print(f"Avg Universe: {avg_universe:.1f}")
    print(f"Avg Trend Candidates: {avg_trend:.1f}")
    print(f"Avg RS Candidates: {avg_rs:.1f}")
    print(f"Avg VCP Candidates: {avg_vcp:.2f}")
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
    if len(symbols) < 100:
        raise ValueError("CRITICAL: Universe failed to load. Aborting backtest.")
    
    data = fetch_data_pack(symbols, days=5040, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=5040, backtest_mode=True) or {}
    
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    result = run_backtest(strategies, prepared, start_cash=100000.0, start_date=args.start_date, end_date=args.end_date, global_data=g_data)
    
    if isinstance(result, list):
        if not result: return 1
        result = result[0]

    print(f"Final Value: ${result['final_value']:,.2f} | Trades: {result['total_trades']} | Win Rate: {result['hit_rate']:.1f}%")
    return 0

if __name__ == "__main__":
    raise SystemExit(_run_cli())
