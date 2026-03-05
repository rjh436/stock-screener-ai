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

    # Optional overlays to de-risk earlier in credit/volatility stress regimes.
    overlay_w = float(regime_cfg.get("overlay_weight", 0.35) or 0.35)
    overlay_w = float(np.clip(overlay_w, 0.0, 1.0))

    if bool(regime_cfg.get("use_vix_overlay", False)):
        vix_frame = global_data.get("VIX") if isinstance(global_data, Mapping) else None
        vix_close = _close_col(vix_frame) if vix_frame is not None else None
        if vix_close is not None and not vix_close.empty:
            vix_ma = vix_close.rolling(63, min_periods=20).mean()
            vix_std = vix_close.rolling(63, min_periods=20).std()
            with np.errstate(invalid="ignore", divide="ignore"):
                z = (vix_close - vix_ma) / vix_std.replace(0.0, np.nan)
            # z <= 0 -> risk-on (1.0), z >= 2.5 -> risk-off (0.0).
            vix_scalar = 1.0 - np.clip((z - 0.0) / 2.5, 0.0, 1.0)
            vix_scalar = pd.Series(vix_scalar, index=vix_close.index, dtype=float).clip(lower=0.0, upper=1.0)
            regime = ((1.0 - overlay_w) * regime) + (overlay_w * vix_scalar.reindex(regime.index).ffill().fillna(1.0))

    if bool(regime_cfg.get("use_credit_overlay", False)):
        hyg_frame = global_data.get("HYG") if isinstance(global_data, Mapping) else None
        lqd_frame = global_data.get("LQD") if isinstance(global_data, Mapping) else None
        hyg_close = _close_col(hyg_frame) if hyg_frame is not None else None
        lqd_close = _close_col(lqd_frame) if lqd_frame is not None else None
        if hyg_close is not None and lqd_close is not None and not hyg_close.empty and not lqd_close.empty:
            aligned = pd.concat([hyg_close.rename("h"), lqd_close.rename("l")], axis=1).dropna()
            if not aligned.empty:
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = aligned["h"] / aligned["l"]
                    mom63 = (ratio / ratio.shift(63)) - 1.0
                # <= -10% -> risk-off (0), >= +10% -> risk-on (1)
                credit_scalar = np.clip((mom63 + 0.10) / 0.20, 0.0, 1.0)
                credit_scalar = pd.Series(credit_scalar, index=aligned.index, dtype=float)
                regime = ((1.0 - overlay_w) * regime) + (overlay_w * credit_scalar.reindex(regime.index).ffill().fillna(1.0))

    target_idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).tz_localize(None).normalize()
    out = regime.reindex(target_idx).ffill().fillna(1.0)
    out = out.clip(lower=risk_off_scalar, upper=1.0)
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
    eps_ttm = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    revenue_ttm = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    net_income_ttm = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    net_margin_ttm = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    eps_accel = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    revenue_accel = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    accruals_ratio = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    roe_ttm = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    roe_trend = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    net_issuance_12m = np.full((n_days, n_syms), np.nan, dtype=np.float32)
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
        if "eps_ttm" in df.columns:
            eps_ttm[idx, j] = pd.to_numeric(df["eps_ttm"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "revenue_ttm" in df.columns:
            revenue_ttm[idx, j] = pd.to_numeric(df["revenue_ttm"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "net_income_ttm" in df.columns:
            net_income_ttm[idx, j] = pd.to_numeric(df["net_income_ttm"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "net_margin_ttm" in df.columns:
            net_margin_ttm[idx, j] = pd.to_numeric(df["net_margin_ttm"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "eps_accel" in df.columns:
            eps_accel[idx, j] = pd.to_numeric(df["eps_accel"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "revenue_accel" in df.columns:
            revenue_accel[idx, j] = pd.to_numeric(df["revenue_accel"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "accruals_ratio" in df.columns:
            accruals_ratio[idx, j] = pd.to_numeric(df["accruals_ratio"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "roe_ttm" in df.columns:
            roe_ttm[idx, j] = pd.to_numeric(df["roe_ttm"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "roe_trend" in df.columns:
            roe_trend[idx, j] = pd.to_numeric(df["roe_trend"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        if "net_issuance_12m" in df.columns:
            net_issuance_12m[idx, j] = pd.to_numeric(df["net_issuance_12m"], errors="coerce").to_numpy(dtype=np.float32)[valid]

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
        "eps_ttm": eps_ttm,
        "revenue_ttm": revenue_ttm,
        "net_income_ttm": net_income_ttm,
        "net_margin_ttm": net_margin_ttm,
        "eps_accel": eps_accel,
        "revenue_accel": revenue_accel,
        "accruals_ratio": accruals_ratio,
        "roe_ttm": roe_ttm,
        "roe_trend": roe_trend,
        "net_issuance_12m": net_issuance_12m,
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


def _daily_membership_price_coverage(features: Dict[str, Any]) -> Dict[str, float]:
    membership = np.asarray(features.get("membership_mask"), dtype=bool)
    close = np.asarray(features.get("close"), dtype=np.float64)
    if membership.size == 0 or close.size == 0 or membership.shape != close.shape:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    with np.errstate(invalid="ignore"):
        available = membership & np.isfinite(close) & (close > 0.0)
    denom = membership.sum(axis=1).astype(np.float64)
    numer = available.sum(axis=1).astype(np.float64)

    ratios = np.full(denom.shape[0], np.nan, dtype=np.float64)
    valid_rows = denom > 0
    ratios[valid_rows] = numer[valid_rows] / denom[valid_rows]
    clean = ratios[np.isfinite(ratios)]
    if clean.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p10": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    return {
        "mean": float(np.nanmean(clean)),
        "median": float(np.nanmedian(clean)),
        "p10": float(np.nanpercentile(clean, 10.0)),
        "p25": float(np.nanpercentile(clean, 25.0)),
        "p75": float(np.nanpercentile(clean, 75.0)),
        "min": float(np.nanmin(clean)),
        "max": float(np.nanmax(clean)),
    }


def _group_indices_from_map(
    symbols: Sequence[str],
    symbol_group_map: Optional[Mapping[str, str]],
    *,
    min_group_size: int = 8,
) -> List[np.ndarray]:
    if not symbol_group_map:
        return []
    groups: Dict[str, List[int]] = {}
    for i, sym in enumerate(symbols):
        key = str(symbol_group_map.get(str(sym), "") or "").strip().upper()
        if not key:
            continue
        groups.setdefault(key, []).append(i)
    out: List[np.ndarray] = []
    for idxs in groups.values():
        if len(idxs) < int(max(2, min_group_size)):
            continue
        out.append(np.asarray(idxs, dtype=int))
    return out


def _demean_by_groups(
    values: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray],
    group_indices: Sequence[np.ndarray],
) -> np.ndarray:
    if values.size == 0 or not group_indices:
        return values
    out = values.astype(np.float64, copy=True)
    for idxs in group_indices:
        if idxs.size < 2:
            continue
        sub = out[:, idxs]
        if valid_mask is None:
            sub_valid = np.isfinite(sub)
        else:
            sub_valid = np.isfinite(sub) & valid_mask[:, idxs]
        if not np.any(sub_valid):
            continue
        numer = np.where(sub_valid, sub, 0.0).sum(axis=1)
        denom = sub_valid.sum(axis=1)
        means = np.divide(
            numer,
            denom,
            out=np.zeros_like(numer, dtype=np.float64),
            where=denom > 0,
        )
        out[:, idxs] = np.where(sub_valid, sub - means[:, None], sub)
    return out.astype(np.float32, copy=False)


def _build_momentum_quality_scores(
    features: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    symbol_group_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    close = features["close"]
    high_52w = features["high_52w"]
    eps_yoy = features["eps_yoy"]
    sales_yoy = features["sales_yoy"]
    inst = features["inst"]
    eps_accel = features.get("eps_accel")
    revenue_accel = features.get("revenue_accel")
    accruals_ratio = features.get("accruals_ratio")
    roe_trend = features.get("roe_trend")
    net_issuance_12m = features.get("net_issuance_12m")
    if eps_accel is None:
        eps_accel = np.full(close.shape, np.nan, dtype=np.float32)
    if revenue_accel is None:
        revenue_accel = np.full(close.shape, np.nan, dtype=np.float32)
    if accruals_ratio is None:
        accruals_ratio = np.full(close.shape, np.nan, dtype=np.float32)
    if roe_trend is None:
        roe_trend = np.full(close.shape, np.nan, dtype=np.float32)
    if net_issuance_12m is None:
        net_issuance_12m = np.full(close.shape, np.nan, dtype=np.float32)

    valid = _base_valid_mask(features, cfg)

    c21 = _lag_matrix(close, 21)
    c126 = _lag_matrix(close, 126)
    c252 = _lag_matrix(close, 252)

    with np.errstate(divide="ignore", invalid="ignore"):
        mom12_1 = (c21 / c252) - 1.0
        mom6_1 = (c21 / c126) - 1.0
        mom1_0 = (close / c21) - 1.0
        proximity = close / high_52w

    if bool(cfg.get("use_sector_relative_momentum", False)):
        min_group_size = int(cfg.get("sector_relative_min_group_size", 8) or 8)
        group_indices = _group_indices_from_map(
            symbols=features.get("symbols", []),
            symbol_group_map=symbol_group_map,
            min_group_size=min_group_size,
        )
        if group_indices:
            mom12_1 = _demean_by_groups(mom12_1, valid_mask=valid, group_indices=group_indices)
            mom6_1 = _demean_by_groups(mom6_1, valid_mask=valid, group_indices=group_indices)

    min_rank_names = int(cfg.get("min_rank_names", 40) or 40)
    r_m12 = _cross_section_rank_matrix(mom12_1, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_m61 = _cross_section_rank_matrix(mom6_1, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_m10_penalty = _cross_section_rank_matrix(mom1_0, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_prox = _cross_section_rank_matrix(proximity, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    q_entries: List[Tuple[np.ndarray, float]] = [
        (
            _cross_section_rank_matrix(eps_yoy, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_eps_yoy_weight", 1.0) or 1.0),
        ),
        (
            _cross_section_rank_matrix(sales_yoy, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_sales_yoy_weight", 1.0) or 1.0),
        ),
        (
            _cross_section_rank_matrix(inst, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_inst_weight", 1.0) or 1.0),
        ),
        (
            _cross_section_rank_matrix(eps_accel, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_eps_accel_weight", 1.0) or 1.0),
        ),
        (
            _cross_section_rank_matrix(revenue_accel, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_revenue_accel_weight", 1.0) or 1.0),
        ),
        (
            _cross_section_rank_matrix(accruals_ratio, valid_mask=valid, higher_is_better=False, min_names=min_rank_names),
            float(cfg.get("quality_accruals_weight", 0.0) or 0.0),
        ),
        (
            _cross_section_rank_matrix(roe_trend, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
            float(cfg.get("quality_roe_trend_weight", 0.0) or 0.0),
        ),
        (
            _cross_section_rank_matrix(net_issuance_12m, valid_mask=valid, higher_is_better=False, min_names=min_rank_names),
            float(cfg.get("quality_issuance_weight", 0.0) or 0.0),
        ),
    ]
    q_num = np.zeros(close.shape, dtype=np.float32)
    q_den = np.zeros(close.shape, dtype=np.float32)
    for mat, wt in q_entries:
        if not np.isfinite(wt) or wt <= 0.0:
            continue
        m = np.isfinite(mat)
        if not np.any(m):
            continue
        q_num[m] += (float(wt) * mat[m]).astype(np.float32)
        q_den[m] += float(wt)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_blend = q_num / q_den
    q_blend[q_den <= 0.0] = np.nan
    r_qual = _cross_section_rank_matrix(q_blend, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)

    w_m12 = float(cfg.get("mom_12_1_weight", 0.45) or 0.45)
    w_m61 = float(cfg.get("mom_6_1_weight", 0.20) or 0.20)
    w_m10 = float(cfg.get("mom_1_0_penalty_weight", 0.10) or 0.10)
    w_prox = float(cfg.get("proximity_52w_weight", 0.15) or 0.15)
    w_qual = float(cfg.get("quality_growth_weight", 0.15) or 0.15)

    score = (w_m12 * r_m12) + (w_m61 * r_m61) + (w_m10 * r_m10_penalty) + (w_prox * r_prox) + (w_qual * r_qual)
    score[~valid] = np.nan

    return pd.DataFrame(score, index=features["dates"], columns=features["symbols"])


def _build_value_proxy_scores(features: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    close = features["close"]
    natr = features["natr"]
    adr = features["adr"]
    eps_yoy = features["eps_yoy"]
    sales_yoy = features["sales_yoy"]
    inst = features["inst"]
    eps_ttm = features.get("eps_ttm")
    net_margin_ttm = features.get("net_margin_ttm")
    accruals_ratio = features.get("accruals_ratio")
    roe_ttm = features.get("roe_ttm")
    net_issuance_12m = features.get("net_issuance_12m")
    if accruals_ratio is None:
        accruals_ratio = np.full(close.shape, np.nan, dtype=np.float32)
    if roe_ttm is None:
        roe_ttm = np.full(close.shape, np.nan, dtype=np.float32)
    if net_issuance_12m is None:
        net_issuance_12m = np.full(close.shape, np.nan, dtype=np.float32)

    valid = _base_valid_mask(features, cfg)

    c21 = _lag_matrix(close, 21)
    c126 = _lag_matrix(close, 126)
    c252 = _lag_matrix(close, 252)

    with np.errstate(divide="ignore", invalid="ignore"):
        mom12_1 = (c21 / c252) - 1.0
        earnings_yield = (eps_ttm / close) if eps_ttm is not None else np.full(close.shape, np.nan, dtype=np.float32)

    min_rank_names = int(cfg.get("min_rank_names", 40) or 40)
    q_parts = [
        _cross_section_rank_matrix(eps_yoy, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
        _cross_section_rank_matrix(sales_yoy, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
        _cross_section_rank_matrix(inst, valid_mask=valid, higher_is_better=True, min_names=min_rank_names),
    ]
    q_stack = np.stack(q_parts, axis=0)
    q_sum = np.nansum(q_stack, axis=0)
    q_cnt = np.sum(np.isfinite(q_stack), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        q_blend = q_sum / q_cnt
    q_blend[q_cnt <= 0] = np.nan
    with np.errstate(invalid="ignore"):
        stability = -0.5 * natr - 0.5 * adr

    # Value-proxy sleeve: prefer laggards with improving fundamentals and lower volatility.
    r_rev = _cross_section_rank_matrix(mom12_1, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_qual = _cross_section_rank_matrix(q_blend, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_stab = _cross_section_rank_matrix(stability, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_ey = _cross_section_rank_matrix(earnings_yield, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_accrual = _cross_section_rank_matrix(accruals_ratio, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_roe = _cross_section_rank_matrix(roe_ttm, valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
    r_issuance = _cross_section_rank_matrix(net_issuance_12m, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    if net_margin_ttm is not None:
        r_margin = _cross_section_rank_matrix(
            net_margin_ttm,
            valid_mask=valid,
            higher_is_better=True,
            min_names=min_rank_names,
        )
    else:
        r_margin = np.full(close.shape, np.nan, dtype=np.float32)

    w_rev = float(cfg.get("value_rev_weight", 0.30) or 0.30)
    w_qual = float(cfg.get("value_quality_weight", 0.20) or 0.20)
    w_stab = float(cfg.get("value_stability_weight", 0.10) or 0.10)
    w_ey = float(cfg.get("value_earnings_yield_weight", 0.35) or 0.35)
    w_margin = float(cfg.get("value_margin_weight", 0.05) or 0.05)
    w_accrual = float(cfg.get("value_accruals_weight", 0.0) or 0.0)
    w_roe = float(cfg.get("value_roe_weight", 0.0) or 0.0)
    w_issuance = float(cfg.get("value_issuance_weight", 0.0) or 0.0)

    score_num = np.zeros(close.shape, dtype=np.float32)
    score_den = np.zeros(close.shape, dtype=np.float32)
    for mat, wt in (
        (r_rev, w_rev),
        (r_qual, w_qual),
        (r_stab, w_stab),
        (r_ey, w_ey),
        (r_margin, w_margin),
        (r_accrual, w_accrual),
        (r_roe, w_roe),
        (r_issuance, w_issuance),
    ):
        if wt <= 0:
            continue
        mask = np.isfinite(mat)
        if not np.any(mask):
            continue
        score_num[mask] += (float(wt) * mat[mask]).astype(np.float32)
        score_den[mask] += float(wt)

    with np.errstate(divide="ignore", invalid="ignore"):
        score = score_num / score_den
    score[score_den <= 0.0] = np.nan
    score[~valid] = np.nan

    # Avoid deep downtrends unless explicitly allowed.
    max_12_1_draw = float(cfg.get("value_max_negative_mom12_1", -0.55) or -0.55)
    score[mom12_1 < max_12_1_draw] = np.nan
    min_earnings_yield = float(cfg.get("value_min_earnings_yield", -0.10) or -0.10)
    score[earnings_yield < min_earnings_yield] = np.nan

    return pd.DataFrame(score, index=features["dates"], columns=features["symbols"])


def _build_low_vol_scores(features: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    close = features["close"]
    natr = features["natr"]
    adr = features["adr"]
    valid = _base_valid_mask(features, cfg)

    lookback = int(cfg.get("low_vol_lookback_days", 63) or 63)
    c1 = _lag_matrix(close, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret1 = (close / c1) - 1.0

    ret_df = pd.DataFrame(ret1, index=features["dates"], columns=features["symbols"])
    rolling_std = ret_df.rolling(window=lookback, min_periods=max(20, lookback // 2)).std()
    vol = rolling_std.to_numpy(dtype=np.float32)

    min_rank_names = int(cfg.get("min_rank_names", 40) or 40)
    r_vol = _cross_section_rank_matrix(vol, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_natr = _cross_section_rank_matrix(natr, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
    r_adr = _cross_section_rank_matrix(adr, valid_mask=valid, higher_is_better=False, min_names=min_rank_names)

    w_vol = float(cfg.get("low_volatility_weight", 0.70) or 0.70)
    w_natr = float(cfg.get("low_natr_weight", 0.20) or 0.20)
    w_adr = float(cfg.get("low_adr_weight", 0.10) or 0.10)

    score = (w_vol * r_vol) + (w_natr * r_natr) + (w_adr * r_adr)
    score[~valid] = np.nan
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


def _build_momentum_low_vol_schedule(
    momentum_scores: pd.DataFrame,
    low_vol_scores: pd.DataFrame,
    *,
    rebalance_freq: str,
    momentum_count: int,
    low_vol_count: int,
    hold_buffer_mult: float,
    momentum_weight: float,
    low_vol_weight: float,
    risk_scalar_by_date: Optional[pd.Series] = None,
    adaptive_low_vol_max_weight: Optional[float] = None,
) -> pd.DataFrame:
    reb_dates = _rebalance_dates(momentum_scores.index, rebalance_freq)
    out_rows: List[Dict[str, float]] = []
    out_idx: List[pd.Timestamp] = []
    prev_m: List[str] = []
    prev_lv: List[str] = []

    for dt in reb_dates:
        mom_row = pd.to_numeric(momentum_scores.loc[dt], errors="coerce").dropna().sort_values(ascending=False)
        lv_row = pd.to_numeric(low_vol_scores.loc[dt], errors="coerce").dropna().sort_values(ascending=False)

        m_sel = select_target_portfolio(
            mom_row,
            target_count=int(momentum_count),
            existing_symbols=prev_m,
            hold_buffer_mult=float(hold_buffer_mult),
        )
        lv_sel = select_target_portfolio(
            lv_row,
            target_count=int(low_vol_count),
            existing_symbols=prev_lv,
            hold_buffer_mult=float(hold_buffer_mult),
        )

        eff_momentum_weight = float(momentum_weight)
        eff_low_vol_weight = float(low_vol_weight)
        if risk_scalar_by_date is not None:
            hist = pd.to_numeric(risk_scalar_by_date.loc[risk_scalar_by_date.index <= dt], errors="coerce")
            risk_scalar = float(hist.iloc[-1]) if not hist.empty else 1.0
            if not np.isfinite(risk_scalar):
                risk_scalar = 1.0
            risk_scalar = float(np.clip(risk_scalar, 0.0, 1.0))
            lv_cap = float(
                adaptive_low_vol_max_weight
                if adaptive_low_vol_max_weight is not None
                else max(float(low_vol_weight), 0.60)
            )
            lv_cap = float(np.clip(lv_cap, 0.0, 1.0))
            eff_low_vol_weight = float(low_vol_weight) + (1.0 - risk_scalar) * (lv_cap - float(low_vol_weight))
            eff_low_vol_weight = float(np.clip(eff_low_vol_weight, 0.0, 1.0))
            eff_momentum_weight = float(np.clip(1.0 - eff_low_vol_weight, 0.0, 1.0))

        weights: Dict[str, float] = {}
        if m_sel:
            wm = float(eff_momentum_weight) / float(len(m_sel))
            for sym in m_sel:
                weights[sym] = weights.get(sym, 0.0) + wm
        if lv_sel:
            wv = float(eff_low_vol_weight) / float(len(lv_sel))
            for sym in lv_sel:
                weights[sym] = weights.get(sym, 0.0) + wv

        total = float(sum(weights.values()))
        if total > 0:
            weights = {k: (v / total) for k, v in weights.items()}

        out_idx.append(pd.Timestamp(dt))
        out_rows.append(weights)
        prev_m = list(m_sel)
        prev_lv = list(lv_sel)

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
    low_vol_scores: Optional[pd.DataFrame],
    eps_yoy_df: Optional[pd.DataFrame],
    sales_yoy_df: Optional[pd.DataFrame],
    risk_scalar_by_date: Optional[pd.Series],
    sector_map: Optional[Mapping[str, str]],
    industry_map: Optional[Mapping[str, str]],
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
    execution_lag_days = int(cfg.get("execution_lag_days", 1) or 1)
    apply_risk_scalar_to_gross = bool(cfg.get("apply_risk_scalar_to_gross", True))
    gross_risk_scalar = risk_scalar_by_date if apply_risk_scalar_to_gross else None

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
            sector_map=sector_map,
            sector_cap=float(cfg.get("sector_cap", 0.20) or 0.20),
            industry_map=industry_map,
            industry_cap=float(cfg.get("industry_cap", 0.15) or 0.15),
            turnover_budget=float(cfg.get("turnover_budget", 0.35) or 0.35),
            transaction_cost_bps=tx_bps,
            start_cash=100000.0,
            target_weights_by_date=schedule,
            risk_scalar_by_date=gross_risk_scalar,
            hard_stop_pct=hard_stop_pct,
            trend_ma_days=trend_ma_days,
            time_stop_days=time_stop_days,
            execution_lag_days=execution_lag_days,
            conviction_weighted=bool(cfg.get("conviction_weighted", False)),
            conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        )
        return run
    if strategy_type == "momentum_low_vol_blend":
        if low_vol_scores is None:
            raise ValueError("momentum_low_vol_blend requires low volatility score matrix")
        lv_win = _slice_frame(low_vol_scores, start_date, end_date)
        schedule = _build_momentum_low_vol_schedule(
            m_win,
            lv_win,
            rebalance_freq=rebalance_freq,
            momentum_count=int(cfg.get("momentum_target_count", 16) or 16),
            low_vol_count=int(cfg.get("low_vol_target_count", 8) or 8),
            hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
            momentum_weight=float(cfg.get("momentum_weight", 0.75) or 0.75),
            low_vol_weight=float(cfg.get("low_vol_weight", 0.25) or 0.25),
            risk_scalar_by_date=risk_scalar_by_date if bool(cfg.get("risk_adaptive_blend", True)) else None,
            adaptive_low_vol_max_weight=(
                None
                if cfg.get("adaptive_low_vol_max_weight") is None
                else float(cfg.get("adaptive_low_vol_max_weight"))
            ),
        )
        run = run_periodic_rebalance(
            prices=p_win,
            ranked_scores=m_win,
            rebalance_freq=rebalance_freq,
            target_count=int(cfg.get("momentum_target_count", 16) or 16),
            hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
            position_cap=float(cfg.get("max_position_weight", 0.06) or 0.06),
            sector_map=sector_map,
            sector_cap=float(cfg.get("sector_cap", 0.20) or 0.20),
            industry_map=industry_map,
            industry_cap=float(cfg.get("industry_cap", 0.15) or 0.15),
            turnover_budget=float(cfg.get("turnover_budget", 0.20) or 0.20),
            transaction_cost_bps=tx_bps,
            start_cash=100000.0,
            target_weights_by_date=schedule,
            risk_scalar_by_date=gross_risk_scalar,
            hard_stop_pct=hard_stop_pct,
            trend_ma_days=trend_ma_days,
            time_stop_days=time_stop_days,
            execution_lag_days=execution_lag_days,
            conviction_weighted=bool(cfg.get("conviction_weighted", False)),
            conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        )
        return run

    run = run_periodic_rebalance(
        prices=p_win,
        ranked_scores=m_win,
        rebalance_freq=rebalance_freq,
        target_count=int(cfg.get("target_count", 20) or 20),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.25) or 1.25),
        position_cap=float(cfg.get("max_position_weight", 0.075) or 0.075),
        sector_map=sector_map,
        sector_cap=float(cfg.get("sector_cap", 0.25) or 0.25),
        industry_map=industry_map,
        industry_cap=float(cfg.get("industry_cap", 0.15) or 0.15),
        turnover_budget=float(cfg.get("turnover_budget", 0.25) or 0.25),
        transaction_cost_bps=tx_bps,
        start_cash=100000.0,
        risk_scalar_by_date=gross_risk_scalar,
        hard_stop_pct=hard_stop_pct,
        trend_ma_days=trend_ma_days,
        time_stop_days=time_stop_days,
        execution_lag_days=execution_lag_days,
        conviction_weighted=bool(cfg.get("conviction_weighted", False)),
        conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
    )
    return run


def _stitch_test_windows(
    *,
    cfg: Dict[str, Any],
    prices: pd.DataFrame,
    momentum_scores: pd.DataFrame,
    value_scores: Optional[pd.DataFrame],
    low_vol_scores: Optional[pd.DataFrame],
    eps_yoy_df: Optional[pd.DataFrame],
    sales_yoy_df: Optional[pd.DataFrame],
    risk_scalar_by_date: Optional[pd.Series],
    sector_map: Optional[Mapping[str, str]],
    industry_map: Optional[Mapping[str, str]],
    windows: List[Tuple[str, str]],
    test_months: int,
    friction: FrictionScenario,
) -> Dict[str, Any]:
    fold_returns: List[float] = []
    fold_rows: List[Dict[str, Any]] = []
    total_test_years = 0.0

    for i, (w_start, w_end) in enumerate(windows, start=1):
        start_ts = pd.Timestamp(w_start)
        end_exclusive_ts = pd.Timestamp(w_end)
        end_inclusive_ts = end_exclusive_ts - pd.Timedelta(days=1)
        if end_inclusive_ts < start_ts:
            continue
        out = _run_factor_window(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            sector_map=sector_map,
            industry_map=industry_map,
            start_date=start_ts.date().isoformat(),
            end_date=end_inclusive_ts.date().isoformat(),
            friction=friction,
        )
        final_val = _safe_float(out.get("final_value"), 100000.0)
        ret = (final_val / 100000.0) - 1.0 if final_val > 0 else float("nan")
        cagr = _safe_float(out.get("cagr"), float("nan"))
        if np.isfinite(cagr):
            cagr *= 100.0
        dd = _pct_dd(out.get("max_drawdown_pct", 0.0))
        trades = int(out.get("total_trades", 0) or 0)
        turnover = _safe_float(out.get("avg_annual_turnover_pct"), float("nan"))

        fold_rows.append(
            {
                "fold": i,
                "start": start_ts.date().isoformat(),
                "end": end_inclusive_ts.date().isoformat(),
                "end_exclusive": end_exclusive_ts.date().isoformat(),
                "return_pct": float(ret * 100.0) if np.isfinite(ret) else float("nan"),
                "cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
                "max_dd_pct": float(dd),
                "trades": trades,
                "avg_annual_turnover_pct": float(turnover) if np.isfinite(turnover) else float("nan"),
            }
        )
        if np.isfinite(ret):
            fold_returns.append(float(ret))
            total_test_years += max(0.0, float((end_exclusive_ts - start_ts).days) / 365.25)

    if not fold_returns:
        return {
            "mode": "cold_start_independent",
            "folds": fold_rows,
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
            "compounded_return_pct": float("nan"),
            "total_test_years": float("nan"),
        }

    compounded = 1.0
    for r in fold_returns:
        compounded *= (1.0 + r)
    if total_test_years <= 0.0:
        total_test_years = float(len(fold_returns) * (float(test_months) / 12.0))
    stitched_cagr = _annualized_cagr(1.0, compounded, total_test_years)
    worst_fold = float(min(fold_returns) * 100.0)

    return {
        "mode": "cold_start_independent",
        "folds": fold_rows,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "worst_fold_return_pct": worst_fold,
        "compounded_return_pct": float((compounded - 1.0) * 100.0),
        "total_test_years": total_test_years,
    }


def _equity_series_from_run(run_out: Dict[str, Any]) -> pd.Series:
    eq_curve = run_out.get("equity_curve") or []
    if not isinstance(eq_curve, list) or not eq_curve:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(eq_curve)
    if frame.empty or "Date" not in frame.columns or "Equity" not in frame.columns:
        return pd.Series(dtype=float)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Equity"] = pd.to_numeric(frame["Equity"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Equity"]).sort_values("Date")
    if frame.empty:
        return pd.Series(dtype=float)
    series = pd.Series(frame["Equity"].to_numpy(dtype=np.float64), index=pd.DatetimeIndex(frame["Date"]))
    series = series[~series.index.duplicated(keep="last")]
    return series


def _stitch_test_windows_warm(
    *,
    full_run: Dict[str, Any],
    windows: List[Tuple[str, str]],
    test_months: int,
) -> Dict[str, Any]:
    eq = _equity_series_from_run(full_run)
    if eq.empty:
        return {
            "mode": "warm_restart_equity_slice",
            "folds": [],
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "oos_max_dd_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
            "compounded_return_pct": float("nan"),
            "total_test_years": float("nan"),
        }

    fold_rows: List[Dict[str, Any]] = []
    fold_returns: List[float] = []
    total_test_years = 0.0
    for i, (w_start, w_end) in enumerate(windows, start=1):
        start_ts = pd.Timestamp(w_start)
        end_exclusive_ts = pd.Timestamp(w_end)
        end_inclusive_ts = end_exclusive_ts - pd.Timedelta(days=1)
        seg = eq.loc[(eq.index >= start_ts) & (eq.index < end_exclusive_ts)]
        if seg.empty or len(seg) < 2:
            fold_rows.append(
                {
                    "fold": i,
                    "start": start_ts.date().isoformat(),
                    "end": end_inclusive_ts.date().isoformat(),
                    "end_exclusive": end_exclusive_ts.date().isoformat(),
                    "return_pct": float("nan"),
                    "cagr_pct": float("nan"),
                    "max_dd_pct": float("nan"),
                    "trades": int(0),
                    "avg_annual_turnover_pct": float("nan"),
                }
            )
            continue
        start_val = float(seg.iloc[0])
        end_val = float(seg.iloc[-1])
        ret = (end_val / start_val) - 1.0 if start_val > 0 else float("nan")
        years = max(1e-9, float((seg.index[-1] - seg.index[0]).days) / 365.25)
        cagr = _annualized_cagr(start_val, end_val, years)
        peaks = seg.cummax()
        dd = ((seg - peaks) / peaks).min() if len(seg) else float("nan")
        fold_rows.append(
            {
                "fold": i,
                "start": start_ts.date().isoformat(),
                "end": end_inclusive_ts.date().isoformat(),
                "end_exclusive": end_exclusive_ts.date().isoformat(),
                "return_pct": float(ret * 100.0) if np.isfinite(ret) else float("nan"),
                "cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
                "max_dd_pct": float(abs(dd) * 100.0) if np.isfinite(dd) else float("nan"),
                "trades": int(0),
                "avg_annual_turnover_pct": float("nan"),
            }
        )
        if np.isfinite(ret):
            fold_returns.append(float(ret))
            total_test_years += max(0.0, float((end_exclusive_ts - start_ts).days) / 365.25)

    if not fold_returns:
        return {
            "mode": "warm_restart_equity_slice",
            "folds": fold_rows,
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "oos_max_dd_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
            "compounded_return_pct": float("nan"),
            "total_test_years": float("nan"),
        }

    compounded = 1.0
    for r in fold_returns:
        compounded *= (1.0 + r)
    if total_test_years <= 0.0:
        total_test_years = float(len(fold_returns) * (float(test_months) / 12.0))
    stitched_cagr = _annualized_cagr(1.0, compounded, total_test_years)
    worst_fold = float(min(fold_returns) * 100.0)
    if windows:
        oos_start = pd.Timestamp(windows[0][0])
        oos_end_exclusive = pd.Timestamp(windows[-1][1])
        oos_seg = eq.loc[(eq.index >= oos_start) & (eq.index < oos_end_exclusive)]
        if len(oos_seg) >= 2:
            oos_peaks = oos_seg.cummax()
            oos_dd = float(abs(((oos_seg - oos_peaks) / oos_peaks).min()) * 100.0)
        else:
            oos_dd = float("nan")
    else:
        oos_dd = float("nan")
    return {
        "mode": "warm_restart_equity_slice",
        "folds": fold_rows,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "oos_max_dd_pct": float(oos_dd) if np.isfinite(oos_dd) else float("nan"),
        "worst_fold_return_pct": worst_fold,
        "compounded_return_pct": float((compounded - 1.0) * 100.0),
        "total_test_years": total_test_years,
    }


def _acceptance_snapshot(
    full_res: Dict[str, Any],
    stitched_36_12: Dict[str, Any],
    stitched_60_12: Dict[str, Any],
    stitched_24_12: Optional[Dict[str, Any]] = None,
    stitched_24_12_warm: Optional[Dict[str, Any]] = None,
    stitched_60_12_warm: Optional[Dict[str, Any]] = None,
    gate_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    audit = full_res.get("audit_report") or {}
    max_gross = _safe_float(audit.get("max_gross_exposure_pct"), 0.0)
    same_day = int(audit.get("same_day_open_entries", 0) or 0)
    cagr = _safe_float(full_res.get("cagr"), float("nan"))
    if np.isfinite(cagr):
        cagr *= 100.0
    mdd = _pct_dd(full_res.get("max_drawdown_pct", 0.0))
    oos_24_cold = _safe_float((stitched_24_12 or {}).get("stitched_cagr_pct"), float("nan"))
    oos_24_warm = _safe_float((stitched_24_12_warm or {}).get("stitched_cagr_pct"), float("nan"))
    oos_24_warm_dd = _safe_float((stitched_24_12_warm or {}).get("oos_max_dd_pct"), float("nan"))
    oos_36 = _safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan"))
    oos_60_cold = _safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan"))
    oos_60_warm = _safe_float((stitched_60_12_warm or {}).get("stitched_cagr_pct"), float("nan"))
    oos_60_warm_dd = _safe_float((stitched_60_12_warm or {}).get("oos_max_dd_pct"), float("nan"))

    gates = dict(gate_cfg or {})
    oos_gate_20 = _safe_float(gates.get("oos60_gate_20x20_cagr_pct"), 6.0)
    oos_gate_35 = _safe_float(gates.get("oos60_gate_35x35_cagr_pct"), 4.5)
    dd_gate = _safe_float(gates.get("max_dd_gate_pct"), 42.0)
    use_warm = bool(gates.get("use_warm_restart_for_oos_gate", True))
    primary_window = str(gates.get("oos_gate_primary_window", "24_12") or "24_12").strip().lower()
    if primary_window not in {"24_12", "60_12"}:
        primary_window = "24_12"
    oos_primary_cold = oos_24_cold if primary_window == "24_12" else oos_60_cold
    oos_primary_warm = oos_24_warm if primary_window == "24_12" else oos_60_warm
    oos_secondary_cold = oos_60_cold if primary_window == "24_12" else oos_24_cold
    oos_secondary_warm = oos_60_warm if primary_window == "24_12" else oos_24_warm

    oos_mode = "warm_restart" if (use_warm and np.isfinite(oos_primary_warm)) else "cold_start"
    oos_for_gate = oos_primary_warm if (use_warm and np.isfinite(oos_primary_warm)) else oos_primary_cold
    if not np.isfinite(oos_for_gate):
        if use_warm and np.isfinite(oos_secondary_warm):
            oos_for_gate = oos_secondary_warm
            oos_mode = "warm_restart_fallback_secondary"
        elif np.isfinite(oos_secondary_cold):
            oos_for_gate = oos_secondary_cold
            oos_mode = "cold_start_fallback_secondary"

    return {
        "cash_only_ok": bool(np.isfinite(max_gross) and max_gross <= 1.0001),
        "same_day_contamination_ok": bool(same_day == 0),
        "same_day_contamination_count": same_day,
        "full_cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
        "full_max_dd_pct": float(mdd),
        "oos_24_12_cagr_pct_cold": float(oos_24_cold) if np.isfinite(oos_24_cold) else float("nan"),
        "oos_24_12_cagr_pct_warm": float(oos_24_warm) if np.isfinite(oos_24_warm) else float("nan"),
        "oos_24_12_max_dd_pct_warm": float(oos_24_warm_dd) if np.isfinite(oos_24_warm_dd) else float("nan"),
        "oos_24_12_cagr_pct": float(oos_24_warm) if np.isfinite(oos_24_warm) else (float(oos_24_cold) if np.isfinite(oos_24_cold) else float("nan")),
        "oos_36_12_cagr_pct": float(oos_36) if np.isfinite(oos_36) else float("nan"),
        "oos_60_12_cagr_pct_cold": float(oos_60_cold) if np.isfinite(oos_60_cold) else float("nan"),
        "oos_60_12_cagr_pct_warm": float(oos_60_warm) if np.isfinite(oos_60_warm) else float("nan"),
        "oos_60_12_max_dd_pct_warm": float(oos_60_warm_dd) if np.isfinite(oos_60_warm_dd) else float("nan"),
        "oos_60_12_cagr_pct": float(oos_for_gate) if np.isfinite(oos_for_gate) else float("nan"),
        "oos_gate_primary_window": primary_window,
        "oos_gate_mode": oos_mode,
        "oos60_gate_20x20_cagr_pct": float(oos_gate_20),
        "oos60_gate_35x35_cagr_pct": float(oos_gate_35),
        "max_dd_gate_pct": float(dd_gate),
        "meets_20x20_gate": bool(np.isfinite(oos_for_gate) and oos_for_gate >= oos_gate_20),
        "meets_35x35_gate": bool(np.isfinite(oos_for_gate) and oos_for_gate >= oos_gate_35),
        "meets_drawdown_gate": bool(np.isfinite(mdd) and mdd <= dd_gate),
    }


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {path}")
    return cfg


def _load_symbol_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip().upper()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


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
    parser.add_argument("--min-universe-coverage", type=float, default=0.60, help="Minimum loaded/union coverage ratio")
    parser.add_argument(
        "--min-daily-membership-coverage",
        type=float,
        default=0.75,
        help="Minimum mean daily PIT membership coverage with available prices",
    )
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
    global_data = fetch_data_pack(["SPY", "VIX", "HYG", "LQD"], days=days, backtest_mode=True) or {}

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
    coverage_stats = _daily_membership_price_coverage(features)
    mean_daily_coverage = _safe_float(coverage_stats.get("mean"), 0.0)
    print(f"Daily PIT membership price coverage (mean): {mean_daily_coverage:.1%}")
    if mean_daily_coverage < float(args.min_daily_membership_coverage):
        raise RuntimeError(
            f"Daily PIT coverage too low ({mean_daily_coverage:.1%} < {float(args.min_daily_membership_coverage):.1%}). "
            "Refusing to run biased factor test."
        )

    prices = pd.DataFrame(features["close"], index=features["dates"], columns=features["symbols"])

    sector_map = _load_symbol_map(ROOT / "config" / "sectors.json")
    industry_map = _load_symbol_map(ROOT / "config" / "industries.json")
    if not industry_map:
        industry_map = _load_symbol_map(ROOT / "config" / "industry.json")

    momentum_scores = _build_momentum_quality_scores(features, cfg, symbol_group_map=sector_map)
    value_scores: Optional[pd.DataFrame] = None
    low_vol_scores: Optional[pd.DataFrame] = None
    strategy_type = str(cfg.get("strategy_type", "cross_sectional_momentum") or "").lower()
    if strategy_type == "separate_value_momentum":
        value_scores = _build_value_proxy_scores(features, cfg)
    elif strategy_type == "momentum_low_vol_blend":
        low_vol_scores = _build_low_vol_scores(features, cfg)

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

    windows_24_12 = _build_test_windows(start_date, end_date, train_months=24, test_months=12)
    windows_36_12 = _build_test_windows(start_date, end_date, train_months=36, test_months=12)
    windows_60_12 = _build_test_windows(start_date, end_date, train_months=60, test_months=12)
    acceptance_gate_cfg = cfg.get("acceptance_gates", {}) if isinstance(cfg.get("acceptance_gates"), Mapping) else {}

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
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            sector_map=sector_map,
            industry_map=industry_map,
            start_date=start_date,
            end_date=end_date,
            friction=fr,
        )
        stitched_24_12 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_24_12,
            test_months=12,
            friction=fr,
        )
        stitched_36_12 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_36_12,
            test_months=12,
            friction=fr,
        )
        stitched_60_12 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=value_scores,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar_by_date,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_60_12,
            test_months=12,
            friction=fr,
        )
        stitched_24_12_warm = _stitch_test_windows_warm(full_run=full, windows=windows_24_12, test_months=12)
        stitched_36_12_warm = _stitch_test_windows_warm(full_run=full, windows=windows_36_12, test_months=12)
        stitched_60_12_warm = _stitch_test_windows_warm(full_run=full, windows=windows_60_12, test_months=12)
        acceptance = _acceptance_snapshot(
            full,
            stitched_36_12,
            stitched_60_12,
            stitched_24_12=stitched_24_12,
            stitched_24_12_warm=stitched_24_12_warm,
            stitched_60_12_warm=stitched_60_12_warm,
            gate_cfg=acceptance_gate_cfg,
        )

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
                "stitched_24_12": stitched_24_12,
                "stitched_24_12_warm": stitched_24_12_warm,
                "stitched_36_12": stitched_36_12,
                "stitched_36_12_warm": stitched_36_12_warm,
                "stitched_60_12": stitched_60_12,
                "stitched_60_12_warm": stitched_60_12_warm,
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
            "daily_membership_price_coverage": coverage_stats,
            "min_daily_membership_coverage": float(args.min_daily_membership_coverage),
            "cache_only": bool(args.cache_only),
        },
        "constraints": {
            "allow_margin": False,
            "max_gross_exposure_pct": 1.0,
        },
        "walkforward_windows": {
            "24_12": windows_24_12,
            "36_12": windows_36_12,
            "60_12": windows_60_12,
            "mode": "rolling_oos_fixed_params",
            "stitched_modes": ["cold_start_independent", "warm_restart_equity_slice"],
        },
        "classification_maps": {
            "sector_map_size": len(sector_map),
            "industry_map_size": len(industry_map),
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
