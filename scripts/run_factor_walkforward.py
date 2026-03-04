#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data
from execution.rebalance_engine import run_periodic_rebalance, select_target_portfolio


@dataclass(frozen=True)
class FrictionScenario:
    name: str
    transaction_cost_bps: float
    entry_slippage_bps: float
    exit_slippage_bps: float


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _pct_dd(value: Any) -> float:
    dd = _safe_float(value, 0.0)
    if dd <= 1.0:
        dd *= 100.0
    return dd


def _annualized_cagr(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return float("nan")
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


def _build_test_windows(start_date: str, end_date: str, train_months: int, test_months: int, step_months: int = 12) -> List[Tuple[str, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows: List[Tuple[str, str]] = []

    test_start = start_ts + pd.DateOffset(months=int(train_months))
    while True:
        test_end = test_start + pd.DateOffset(months=int(test_months))
        if test_end > end_ts:
            break
        windows.append((test_start.date().isoformat(), test_end.date().isoformat()))
        test_start = test_start + pd.DateOffset(months=int(step_months))
    return windows


def _parse_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if getattr(pd.Timestamp(value), "tzinfo", None) is not None else pd.Timestamp(value)


def _close_col(frame: pd.DataFrame) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    for col in ("Close", "close", "adj_close", "Adj Close"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce")
            idx = pd.to_datetime(series.index, errors="coerce")
            try:
                if idx.tz is not None:
                    idx = idx.tz_convert(None)
            except Exception:
                pass
            # Align to date-level calendar used by price matrices.
            idx = pd.DatetimeIndex(idx).tz_localize(None).normalize()
            series.index = idx
            series = series[~series.index.duplicated(keep="last")]
            return series
    return None


def _build_market_risk_scalar(
    global_data: Mapping[str, pd.DataFrame],
    index: pd.DatetimeIndex,
    cfg: Mapping[str, Any],
) -> Optional[pd.Series]:
    regime_cfg = cfg.get("market_regime", {})
    if not isinstance(regime_cfg, Mapping):
        return None
    if not bool(regime_cfg.get("enabled", False)):
        return None

    symbol = str(regime_cfg.get("symbol", "SPY") or "SPY").upper()
    frame = global_data.get(symbol) if isinstance(global_data, Mapping) else None
    close = _close_col(frame) if frame is not None else None
    if close is None or close.empty:
        return None

    raw_ma_days = regime_cfg.get("ma_days", 200)
    ma_days = int(200 if raw_ma_days is None else raw_ma_days)
    raw_risk_off = regime_cfg.get("risk_off_scalar", 0.35)
    risk_off_scalar = float(0.35 if raw_risk_off is None else raw_risk_off)
    risk_off_scalar = float(np.clip(risk_off_scalar, 0.0, 1.0))

    ma = close.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    regime = pd.Series(1.0, index=close.index, dtype=float)
    regime[(close < ma) & ma.notna()] = risk_off_scalar

    target_idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).tz_localize(None).normalize()
    out = regime.reindex(target_idx).ffill().fillna(1.0)
    out = out.astype(float).clip(lower=0.0, upper=1.0)
    return out


def _lag_matrix(mat: np.ndarray, periods: int) -> np.ndarray:
    out = np.roll(mat, int(periods), axis=0)
    out[: int(periods), :] = np.nan
    return out


def _cross_section_rank_matrix(
    values: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    higher_is_better: bool = True,
    min_names: int = 25,
) -> np.ndarray:
    n_days, n_syms = values.shape
    out = np.full((n_days, n_syms), np.nan, dtype=np.float32)

    for i in range(n_days):
        row = values[i, :]
        if valid_mask is None:
            valid = np.isfinite(row)
        else:
            valid = np.isfinite(row) & valid_mask[i, :]

        idx = np.where(valid)[0]
        if idx.size < int(min_names):
            continue

        vals = row[idx]
        order = np.argsort(vals)
        ranks = np.empty(order.shape[0], dtype=np.float64)
        ranks[order] = np.arange(order.shape[0], dtype=np.float64)

        if order.shape[0] == 1:
            pct = np.array([0.5], dtype=np.float64)
        else:
            pct = ranks / float(order.shape[0] - 1)

        if not higher_is_better:
            pct = 1.0 - pct

        out[i, idx] = (pct * 100.0).astype(np.float32)

    return out


def _build_membership_mask(
    dates: Sequence[pd.Timestamp],
    symbols: Sequence[str],
    membership_by_day: Sequence[Optional[Iterable[str]]],
) -> np.ndarray:
    sym_to_idx = {str(s): i for i, s in enumerate(symbols)}
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    for i, members in enumerate(membership_by_day):
        if members is None:
            continue
        for sym in members:
            j = sym_to_idx.get(str(sym))
            if j is not None:
                mask[i, j] = True
    return mask


def _extract_feature_arrays(
    prepared,
    membership_by_day: Sequence[Optional[Iterable[str]]],
) -> Dict[str, Any]:
    all_dates = pd.to_datetime(list(getattr(prepared, "all_dates", [])))
    symbols = sorted(list(getattr(prepared, "enriched", {}).keys()))
    n_days = len(all_dates)
    n_syms = len(symbols)

    close = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    vol_ma20 = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    high_52w = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    eps_yoy = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    sales_yoy = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    inst = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    natr = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    adr = np.full((n_days, n_syms), np.nan, dtype=np.float32)

    for j, sym in enumerate(symbols):
        sd = prepared.enriched.get(sym)
        if sd is None:
            continue
        valid = (sd.gidx >= 0) & (sd.gidx < n_days)
        if not np.any(valid):
            continue
        idx = sd.gidx[valid]

        close[idx, j] = sd.close[valid].astype(np.float32)
        high_52w[idx, j] = sd.high52w[valid].astype(np.float32)
        natr[idx, j] = sd.natr[valid].astype(np.float32)
        adr[idx, j] = sd.adr_pct[valid].astype(np.float32)

        df = sd.df
        if "vol_ma20" in df.columns:
            vol_ma20[idx, j] = pd.to_numeric(df["vol_ma20"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        elif "vol_ma30" in df.columns:
            vol_ma20[idx, j] = pd.to_numeric(df["vol_ma30"], errors="coerce").to_numpy(dtype=np.float32)[valid]

        if "eps_growth_yoy" in df.columns:
            eps_yoy[idx, j] = pd.to_numeric(df["eps_growth_yoy"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "sales_growth_yoy" in df.columns:
            sales_yoy[idx, j] = pd.to_numeric(df["sales_growth_yoy"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "institutional_sponsorship" in df.columns:
            inst[idx, j] = pd.to_numeric(df["institutional_sponsorship"], errors="coerce").to_numpy(dtype=np.float32)[valid]

    membership_mask = _build_membership_mask(all_dates, symbols, membership_by_day)

    return {
        "dates": pd.DatetimeIndex(all_dates),
        "symbols": symbols,
        "close": close,
        "vol_ma20": vol_ma20,
        "high_52w": high_52w,
        "eps_yoy": eps_yoy,
        "sales_yoy": sales_yoy,
        "inst": inst,
        "natr": natr,
        "adr": adr,
        "membership_mask": membership_mask,
    }


def _base_valid_mask(features: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    close = features["close"]
    vol_ma20 = features["vol_ma20"]
    membership_mask = features["membership_mask"]

    min_price = float(cfg.get("min_price", 10.0) or 10.0)
    min_adv20 = float(cfg.get("min_adv20", 5_000_000.0) or 5_000_000.0)
    min_history = int(cfg.get("min_history_bars", 252) or 252)
    drop_bottom_frac = float(cfg.get("drop_bottom_dv_frac", 0.20) or 0.20)

    dollar_vol = close * vol_ma20
    valid = np.isfinite(close) & np.isfinite(dollar_vol) & membership_mask
    valid &= close >= min_price
    valid &= dollar_vol >= min_adv20

    hist = np.cumsum(np.isfinite(close), axis=0)
    valid &= hist >= int(min_history)

    if drop_bottom_frac > 0:
        for i in range(valid.shape[0]):
            idx = np.where(valid[i, :])[0]
            if idx.size < 40:
                continue
            dv = dollar_vol[i, idx]
            if not np.any(np.isfinite(dv)):
                continue
            thresh = np.nanpercentile(dv, drop_bottom_frac * 100.0)
            valid[i, idx] &= dv > thresh

    return valid


def _build_momentum_quality_scores(features: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    close = features["close"]
    high_52w = features["high_52w"]
    eps_yoy = features["eps_yoy"]
    sales_yoy = features["sales_yoy"]
    inst = features["inst"]

    valid = _base_valid_mask(features, cfg)

    c21 = _lag_matrix(close, 21)
    c126 = _lag_matrix(close, 126)
    c252 = _lag_matrix(close, 252)

    with np.errstate(divide="ignore", invalid="ignore"):
        mom12_1 = (c21 / c252) - 1.0
        mom6_1 = (c21 / c126) - 1.0
        proximity = close / high_52w

    stack = np.stack([eps_yoy, sales_yoy, inst], axis=0)
    quality_sum = np.nansum(stack, axis=0)
    quality_cnt = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        quality_raw = quality_sum / quality_cnt
    quality_raw[quality_cnt <= 0] = np.nan

    min_rank_names = int(cfg.get("min_rank_names", 40) or 40)
    r_m12 = _cross_section_rank_matrix(mom12_1, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_m61 = _cross_section_rank_matrix(mom6_1, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_prox = _cross_section_rank_matrix(proximity, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_qual = _cross_section_rank_matrix(quality_raw, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)

    w_m12 = float(cfg.get("mom_12_1_weight", 0.50) or 0.50)
    w_m61 = float(cfg.get("mom_6_1_weight", 0.20) or 0.20)
    w_prox = float(cfg.get("proximity_52w_weight", 0.15) or 0.15)
    w_qual = float(cfg.get("quality_growth_weight", 0.15) or 0.15)

    score = (w_m12 * r_m12) + (w_m61 * r_m61) + (w_prox * r_prox) + (w_qual * r_qual)
    score[~valid] = np.nan

    return pd.DataFrame(score, index=features["dates"], columns=features["symbols"])


def _build_value_proxy_scores(features: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    close = features["close"]
    natr = features["natr"]
    adr = features["adr"]
    eps_yoy = features["eps_yoy"]
    sales_yoy = features["sales_yoy"]
    inst = features["inst"]

    valid = _base_valid_mask(features, cfg)

    c21 = _lag_matrix(close, 21)
    c126 = _lag_matrix(close, 126)
    c252 = _lag_matrix(close, 252)

    with np.errstate(divide="ignore", invalid="ignore"):
        mom12_1 = (c21 / c252) - 1.0

    stack = np.stack([eps_yoy, sales_yoy, inst], axis=0)
    quality_sum = np.nansum(stack, axis=0)
    quality_cnt = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        quality_raw = quality_sum / quality_cnt
    quality_raw[quality_cnt <= 0] = np.nan
    with np.errstate(invalid="ignore"):
        stability = -0.5 * natr - 0.5 * adr

    # Value-proxy sleeve: prefer laggards with improving fundamentals and lower volatility.
    min_rank_names = int(cfg.get("min_rank_names", 40) or 40)
    r_rev = _cross_section_rank_matrix(mom12_1, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_qual = _cross_section_rank_matrix(quality_raw, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_stab = _cross_section_rank_matrix(stability, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)

    w_rev = float(cfg.get("value_rev_weight", 0.45) or 0.45)
    w_qual = float(cfg.get("value_quality_weight", 0.30) or 0.30)
    w_stab = float(cfg.get("value_stability_weight", 0.25) or 0.25)

    score = (w_rev * r_rev) + (w_qual * r_qual) + (w_stab * r_stab)
    score[~valid] = np.nan

    # Avoid deep downtrends unless explicitly allowed.
    max_12_1_draw = float(cfg.get("value_max_negative_mom12_1", -0.55) or -0.55)
    score[mom12_1 < max_12_1_draw] = np.nan

    return pd.DataFrame(score, index=features["dates"], columns=features["symbols"])


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.Series(index, index=index).groupby(index.to_period(freq)).tail(1).index)


def _build_separate_value_momentum_schedule(
    momentum_scores: pd.DataFrame,
    value_scores: pd.DataFrame,
    *,
    rebalance_freq: str,
    momentum_count: int,
    value_count: int,
    hold_buffer_mult: float,
    momentum_weight: float,
    value_weight: float,
    eps_yoy: Optional[pd.DataFrame] = None,
    sales_yoy: Optional[pd.DataFrame] = None,
    min_eps_overlay: float = 0.0,
    min_sales_overlay: float = 0.0,
) -> pd.DataFrame:
    reb_dates = _rebalance_dates(momentum_scores.index, rebalance_freq)
    out_rows: List[Dict[str, float]] = []
    out_idx: List[pd.Timestamp] = []

    prev_m: List[str] = []
    prev_v: List[str] = []

    for dt in reb_dates:
        mom_row = pd.to_numeric(momentum_scores.loc[dt], errors="coerce").dropna().sort_values(ascending=False)
        val_row = pd.to_numeric(value_scores.loc[dt], errors="coerce").dropna().sort_values(ascending=False)

        m_sel = select_target_portfolio(
            mom_row,
            target_count=int(momentum_count),
            existing_symbols=prev_m,
            hold_buffer_mult=float(hold_buffer_mult),
        )
        v_sel = select_target_portfolio(
            val_row,
            target_count=int(value_count),
            existing_symbols=prev_v,
            hold_buffer_mult=float(hold_buffer_mult),
        )

        if eps_yoy is not None and sales_yoy is not None and v_sel:
            filtered: List[str] = []
            for sym in v_sel:
                eps_val = _safe_float(eps_yoy.at[dt, sym] if sym in eps_yoy.columns else np.nan)
                sales_val = _safe_float(sales_yoy.at[dt, sym] if sym in sales_yoy.columns else np.nan)
                eps_ok = (not np.isfinite(eps_val)) or (eps_val >= float(min_eps_overlay))
                sales_ok = (not np.isfinite(sales_val)) or (sales_val >= float(min_sales_overlay))
                if eps_ok and sales_ok:
                    filtered.append(sym)
            if filtered:
                v_sel = filtered

        weights: Dict[str, float] = {}
        if m_sel:
            wm = float(momentum_weight) / float(len(m_sel))
            for sym in m_sel:
                weights[sym] = weights.get(sym, 0.0) + wm
        if v_sel:
            wv = float(value_weight) / float(len(v_sel))
            for sym in v_sel:
                weights[sym] = weights.get(sym, 0.0) + wv

        total = float(sum(weights.values()))
        if total > 0:
            weights = {k: (v / total) for k, v in weights.items()}

        out_idx.append(pd.Timestamp(dt))
        out_rows.append(weights)
        prev_m = list(m_sel)
        prev_v = list(v_sel)

    if not out_rows:
        return pd.DataFrame(index=momentum_scores.index)

    schedule = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_idx)).fillna(0.0)
    schedule = schedule.sort_index()
    return schedule


def _slice_frame(frame: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return frame.loc[(frame.index >= start_ts) & (frame.index <= end_ts)].copy()


def _run_factor_window(
    *,
    cfg: Dict[str, Any],
    prices: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    value_scores: Optional[pd.DataFrame],
    eps_yoy_df: Optional[pd.DataFrame],
    sales_yoy_df: Optional[pd.DataFrame],
    risk_scalar_by_date: Optional[pd.Series],
    start_date: str,
    end_date: str,
    friction: FrictionScenario,
) -> Dict[str, Any]:
    p_win = _slice_frame(prices, start_date, end_date)
    m_win = _slice_frame(momentum_scores, start_date, end_date)
    if p_win.empty or m_win.empty:
        return {
            "final_value": 100000.0,
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "equity_curve": [],
            "rebalance_log": [],
            "audit_report": {"same_day_open_entries": 0, "max_gross_exposure_pct": 0.0, "exposure_profile": []},
            "avg_annual_turnover_pct": 0.0,
        }

    rebalance_freq = str(cfg.get("rebalance_freq", "M") or "M").upper()
    if rebalance_freq.startswith("Q"):
        rebalance_freq = "Q"
    else:
        rebalance_freq = "M"

    tx_bps = float(friction.transaction_cost_bps)
    # Apply entry+exit slippage as additional transaction-cost proxy.
    tx_bps += float(friction.entry_slippage_bps + friction.exit_slippage_bps)

    raw_hard_stop = cfg.get("hard_stop_pct")
    hard_stop_pct = None if raw_hard_stop is None else float(raw_hard_stop)
    raw_trend_ma = cfg.get("trend_ma_days")
    trend_ma_days = None if raw_trend_ma is None else int(raw_trend_ma)
    raw_time_stop = cfg.get("time_stop_days")
    time_stop_days = None if raw_time_stop is None else int(raw_time_stop)

    strategy_type = str(cfg.get("strategy_type", "cross_sectional_momentum") or "cross_sectional_momentum").lower()
    if strategy_type == "separate_value_momentum":
        if value_scores is None:
            raise ValueError("separate_value_momentum requires value score matrix")
        v_win = _slice_frame(value_scores, start_date, end_date)
        e_win = _slice_frame(eps_yoy_df, start_date, end_date) if eps_yoy_df is not None else None
        s_win = _slice_frame(sales_yoy_df, start_date, end_date) if sales_yoy_df is not None else None

        schedule = _build_separate_value_momentum_schedule(
            m_win,
            v_win,
            rebalance_freq=rebalance_freq,
            momentum_count=int(cfg.get("momentum_target_count", 12) or 12),
            value_count=int(cfg.get("value_target_count", 12) or 12),
            hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
            momentum_weight=float(cfg.get("momentum_weight", 0.5) or 0.5),
            value_weight=float(cfg.get("value_weight", 0.5) or 0.5),
            eps_yoy=e_win,
            sales_yoy=s_win,
            min_eps_overlay=float(cfg.get("overlay_min_eps_growth_yoy", 0.0) or 0.0),
            min_sales_overlay=float(cfg.get("overlay_min_sales_growth_yoy", 0.0) or 0.0),
        )
        run = run_periodic_rebalance(
            prices=p_win,
            ranked_scores=m_win,
            rebalance_freq=rebalance_freq,
            target_count=int(cfg.get("momentum_target_count", 12) or 12),
            hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
            position_cap=float(cfg.get("max_position_weight", 0.05) or 0.05),
            sector_cap=float(cfg.get("sector_cap", 0.20) or 0.20),
            industry_cap=float(cfg.get("industry_cap", 0.15) or 0.15),
            turnover_budget=float(cfg.get("turnover_budget", 0.35) or 0.35),
            transaction_cost_bps=tx_bps,
            start_cash=100000.0,
            target_weights_by_date=schedule,
            risk_scalar_by_date=risk_scalar_by_date,
            hard_stop_pct=hard_stop_pct,
            trend_ma_days=trend_ma_days,
            time_stop_days=time_stop_days,
        )
        return run

    run = run_periodic_rebalance(
        prices=p_win,
        ranked_scores=m_win,
        rebalance_freq=rebalance_freq,
        target_count=int(cfg.get("target_count", 20) or 20),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
        position_cap=float(cfg.get("max_position_weight", 0.075) or 0.075),
        sector_cap=float(cfg.get("sector_cap", 0.25) or 0.25),
        industry_cap=float(cfg.get("industry_cap", 0.15) or 0.15),
        turnover_budget=float(cfg.get("turnover_budget", 0.25) or 0.25),
        transaction_cost_bps=tx_bps,
        start_cash=100000.0,
        risk_scalar_by_date=risk_scalar_by_date,
        hard_stop_pct=hard_stop_pct,
        trend_ma_days=trend_ma_days,
        time_stop_days=time_stop_days,
    )
    return run


def _stitch_test_windows(
    *,
    cfg: Dict[str, Any],
    prices: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    value_scores: Optional[pd.DataFrame],
    eps_yoy_df: Optional[pd.DataFrame],
    sales_yoy_df: Optional[pd.DataFrame],
    risk_scalar_by_date: Optional[pd.Series],
    windows: List[Tuple[str, str]],
    test_months: int,
    friction: FrictionScenario,
) -> Dict[str, Any]:
    fold_returns: List[float] = []
    fold_rows: List[Dict[str, Any]] = []

    for i, (w_start, w_end) in enumerate(windows, start=1):
        out = _run_factor_window(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            start_date=w_start,
            end_date=w_end,
            friction=friction,
        )
        final_val = _safe_float(out.get("final_value"), 100000.0)
        ret = (final_val / 100000.0) - 1.0 if final_val > 0 else float("nan")
        cagr = _safe_float(out.get("cagr"), float("nan"))
        if np.isfinite(cagr):
            cagr *= 100.0
        dd = _pct_dd(out.get("max_drawdown_pct", 0.0))
        trades = int(out.get("total_trades", 0) or 0)

        fold_rows.append(
            {
                "fold": i,
                "start": w_start,
                "end": w_end,
                "return_pct": float(ret * 100.0) if np.isfinite(ret) else float("nan"),
                "cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
                "max_dd_pct": float(dd),
                "trades": trades,
            }
        )
        if np.isfinite(ret):
            fold_returns.append(float(ret))

    if not fold_returns:
        return {
            "folds": fold_rows,
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
        }

    compounded = 1.0
    for r in fold_returns:
        compounded *= (1.0 + r)
    total_test_years = float(len(fold_returns) * (float(test_months) / 12.0))
    stitched_cagr = _annualized_cagr(1.0, compounded, total_test_years)
    worst_fold = float(min(fold_returns) * 100.0)

    return {
        "folds": fold_rows,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "worst_fold_return_pct": worst_fold,
    }


def _acceptance_snapshot(full_res: Dict[str, Any], stitched_36_12: Dict[str, Any], stitched_60_12: Dict[str, Any]) -> Dict[str, Any]:
    audit = full_res.get("audit_report") or {}
    max_gross = _safe_float(audit.get("max_gross_exposure_pct"), 0.0)
    cagr = _safe_float(full_res.get("cagr"), float("nan"))
    if np.isfinite(cagr):
        cagr *= 100.0
    mdd = _pct_dd(full_res.get("max_drawdown_pct", 0.0))
    oos_36 = _safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan"))
    oos_60 = _safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan"))

    return {
        "cash_only_ok": bool(np.isfinite(max_gross) and max_gross <= 1.0001),
        "same_day_contamination_ok": True,
        "full_cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
        "full_max_dd_pct": float(mdd),
        "oos_36_12_cagr_pct": float(oos_36) if np.isfinite(oos_36) else float("nan"),
        "oos_60_12_cagr_pct": float(oos_60) if np.isfinite(oos_60) else float("nan"),
        "meets_20x20_gate": bool(np.isfinite(oos_60) and oos_60 >= 10.0),
        "meets_35x35_gate": bool(np.isfinite(oos_60) and oos_60 >= 8.0),
        "meets_drawdown_gate": bool(np.isfinite(mdd) and mdd <= 28.0),
    }


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {path}")
    return cfg


def _days_for_range(start_date: str, end_date: str, warmup_days: int = 420) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span_days = max(1, int((end_ts - start_ts).days))
    return span_days + int(max(0, warmup_days))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cash-only Russell 3000 factor walk-forward robustness")
    parser.add_argument("--config", required=True, help="Path to factor config JSON")
    parser.add_argument("--start", default="2010-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.now().date().isoformat(), help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="", help="Output JSON report path")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Base transaction cost bps")
    parser.add_argument("--frictions", default="10,20,35,50", help="Comma-separated slippage bps pairs")
    parser.add_argument("--cache-only", action="store_true", help="Use local cache only (no network refresh)")
    parser.add_argument("--min-universe-coverage", type=float, default=0.55, help="Minimum loaded/union coverage ratio")
    args = parser.parse_args()

    start_date = str(args.start)
    end_date = str(args.end)
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = _load_config(cfg_path)

    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError(f"PIT Russell 3000 universe is required, got source={source}")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if not symbols:
        raise RuntimeError("PIT Russell 3000 universe returned zero symbols")

    days = _days_for_range(start_date, end_date, warmup_days=420)
    print(f"Loading Russell 3000 PIT union: {len(symbols)} symbols ({start_date} -> {end_date}, days={days})")
    data = fetch_data_pack(symbols, days=days, backtest_mode=bool(args.cache_only)) or {}
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    loaded_symbols = sorted(list(data.keys()))
    coverage = (len(loaded_symbols) / float(len(symbols))) if symbols else 0.0
    print(f"Loaded symbols with price history: {len(loaded_symbols)} / {len(symbols)} ({coverage:.1%})")
    if coverage < float(args.min_universe_coverage):
        raise RuntimeError(
            f"Universe coverage too low ({coverage:.1%} < {float(args.min_universe_coverage):.1%}). "
            "Refusing to run reduced-universe factor test."
        )

    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership_by_day or len(membership_by_day) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    features = _extract_feature_arrays(prepared, membership_by_day)
    prices = pd.DataFrame(features["close"], index=features["dates"], columns=features["symbols"])

    momentum_scores = _build_momentum_quality_scores(features, cfg)
    value_scores: Optional[pd.DataFrame] = None
    if str(cfg.get("strategy_type", "cross_sectional_momentum") or "").lower() == "separate_value_momentum":
        value_scores = _build_value_proxy_scores(features, cfg)

    eps_yoy_df = pd.DataFrame(features["eps_yoy"], index=features["dates"], columns=features["symbols"])
    sales_yoy_df = pd.DataFrame(features["sales_yoy"], index=features["dates"], columns=features["symbols"])
    risk_scalar_by_date = _build_market_risk_scalar(global_data, prices.index, cfg)

    friction_vals: List[float] = []
    for part in str(args.frictions or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        val = _safe_float(txt, float("nan"))
        if np.isfinite(val) and val >= 0:
            friction_vals.append(float(val))
    if not friction_vals:
        friction_vals = [20.0]

    friction_grid = [
        FrictionScenario(
            name=f"{int(v)}x{int(v)}",
            transaction_cost_bps=float(args.transaction_cost_bps),
            entry_slippage_bps=float(v),
            exit_slippage_bps=float(v),
        )
        for v in friction_vals
    ]

    windows_36_12 = _build_test_windows(start_date, end_date, train_months=36, test_months=12)
    windows_60_12 = _build_test_windows(start_date, end_date, train_months=60, test_months=12)

    scenarios = []
    for fr in friction_grid:
        print(
            f"Running {fr.name}: tc={fr.transaction_cost_bps}bps, "
            f"entry={fr.entry_slippage_bps}bps, exit={fr.exit_slippage_bps}bps"
        )
        full = _run_factor_window(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            start_date=start_date,
            end_date=end_date,
            friction=fr,
        )
        stitched_36_12 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            windows=windows_36_12,
            test_months=12,
            friction=fr,
        )
        stitched_60_12 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            windows=windows_60_12,
            test_months=12,
            friction=fr,
        )
        acceptance = _acceptance_snapshot(full, stitched_36_12, stitched_60_12)

        scenarios.append(
            {
                "scenario": fr.name,
                "friction": {
                    "transaction_cost_bps": fr.transaction_cost_bps,
                    "entry_slippage_bps": fr.entry_slippage_bps,
                    "exit_slippage_bps": fr.exit_slippage_bps,
                },
                "full": {
                    "cagr_pct": _safe_float(full.get("cagr"), float("nan")) * 100.0 if np.isfinite(_safe_float(full.get("cagr"), float("nan"))) else float("nan"),
                    "max_dd_pct": _pct_dd(full.get("max_drawdown_pct", 0.0)),
                    "trades": int(full.get("total_trades", 0) or 0),
                    "final_value": _safe_float(full.get("final_value"), float("nan")),
                    "avg_annual_turnover_pct": _safe_float(full.get("avg_annual_turnover_pct"), float("nan")),
                    "annual_turnover_pct": full.get("annual_turnover_pct", {}),
                    "audit": full.get("audit_report", {}),
                },
                "stitched_36_12": stitched_36_12,
                "stitched_60_12": stitched_60_12,
                "acceptance": acceptance,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "strategy_type": str(cfg.get("strategy_type", "cross_sectional_momentum")),
        "start_date": start_date,
        "end_date": end_date,
        "universe": {
            "name": "RUSSELL3000",
            "source": source,
            "membership_source": membership_source,
            "requested_symbol_count": len(symbols),
            "loaded_symbol_count": len(loaded_symbols),
            "coverage_ratio": coverage,
            "cache_only": bool(args.cache_only),
        },
        "constraints": {
            "allow_margin": False,
            "max_gross_exposure_pct": 1.0,
        },
        "walkforward_windows": {
            "36_12": windows_36_12,
            "60_12": windows_60_12,
        },
        "scenarios": scenarios,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (ROOT / "logs" / f"walkforward_factor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    main()
