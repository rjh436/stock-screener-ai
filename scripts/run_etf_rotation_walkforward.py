#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from execution.rebalance_engine import (
    _build_selected_target_weights,
    _sanitize_target_weights,
    run_periodic_rebalance,
    select_target_portfolio,
)
from scripts.run_factor_walkforward import (
    _build_market_risk_scalar,
    _build_test_windows,
    _cross_section_rank_matrix,
    _pct_dd,
    _safe_float,
    _stitch_test_windows_warm,
)

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional runtime dependency
    yf = None


DEFAULT_CONFIG = ROOT / "config" / "etf_rotation_3x_growth_v1.json"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return dict(cfg)


def _normalize_symbol_list(values: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values or []:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _yf_symbol(sym: str) -> str:
    sym = str(sym or "").strip().upper()
    if sym == "VIX":
        return "^VIX"
    return sym


def _maybe_patch_yfinance() -> None:
    if yf is None:
        return
    try:
        from yfinance.data import YfData
        from curl_cffi import requests as curl_requests
    except Exception:
        return

    if getattr(YfData, "_codex_cookie_patch", False):
        return

    def _get_cookie_basic(self, timeout=30):
        if getattr(self, "_cookie", None) is not None:
            return True
        try:
            if self._load_cookie_curlCffi():
                return True
        except Exception:
            pass
        try:
            self._session.get(
                url="https://guce.yahoo.com/consent",
                timeout=timeout,
                allow_redirects=False,
            )
        except curl_requests.exceptions.DNSError:
            return False
        except Exception:
            return False
        try:
            cookies = self._session.cookies.jar._cookies
            yahoo_domains = [d for d in cookies.keys() if "yahoo" in d]
            if len(yahoo_domains) > 1:
                yahoo_domains = [d for d in yahoo_domains if "consent" not in d]
            if not yahoo_domains:
                return False
            domain = yahoo_domains[0]
            cookie = cookies[domain]["/"].get("A3")
            if cookie is None:
                return False
            self._cookie = cookie
        except Exception:
            return False
        try:
            self._save_cookie_curlCffi()
        except Exception:
            pass
        return True

    YfData._get_cookie_basic = _get_cookie_basic
    YfData._codex_cookie_patch = True


def _download_yfinance_pack(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    if yf is None:
        return {}
    symbols = _normalize_symbol_list(symbols)
    if not symbols:
        return {}

    _maybe_patch_yfinance()
    start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=700)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=5)
    request_symbols = [_yf_symbol(sym) for sym in symbols]
    frame = yf.download(
        request_symbols,
        start=start_ts.date().isoformat(),
        end=end_ts.date().isoformat(),
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if frame is None or len(frame) == 0:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        ysym = _yf_symbol(sym)
        try:
            sub = frame[ysym].copy()
        except Exception:
            continue
        if sub is None or sub.empty:
            continue
        sub.columns = [str(c).strip().lower().replace(" ", "_") for c in sub.columns]
        rename_map = {"adj_close": "adj_close"}
        sub = sub.rename(columns=rename_map)
        sub.index = pd.DatetimeIndex(pd.to_datetime(sub.index, errors="coerce")).tz_localize(None)
        sub = sub[~sub.index.duplicated(keep="last")].sort_index()
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col in sub.columns:
                sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        out[sym] = sub
    return out


def _price_frame_from_data(
    data: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    column: str,
) -> pd.DataFrame:
    series_map: Dict[str, pd.Series] = {}
    for sym in symbols:
        frame = data.get(sym)
        if frame is None or frame.empty or column not in frame.columns:
            continue
        ser = pd.to_numeric(frame[column], errors="coerce")
        if ser.empty:
            continue
        idx = pd.to_datetime(ser.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = pd.DatetimeIndex(idx).tz_localize(None).normalize()
        ser.index = idx
        ser = ser[~ser.index.duplicated(keep="last")]
        series_map[str(sym)] = ser
    if not series_map:
        return pd.DataFrame()
    out = pd.concat(series_map, axis=1).sort_index()
    out.columns = [str(c) for c in out.columns]
    return out


def _lag_frame(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    return frame.shift(int(periods))


def _parse_lookbacks(cfg: Mapping[str, Any]) -> List[Tuple[int, float]]:
    raw = cfg.get("lookbacks", {})
    out: List[Tuple[int, float]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            try:
                days = int(key)
                weight = float(value)
            except Exception:
                continue
            if days <= 0 or not np.isfinite(weight) or weight <= 0:
                continue
            out.append((days, weight))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            try:
                days = int(item.get("days", 0) or 0)
                weight = float(item.get("weight", 0.0) or 0.0)
            except Exception:
                continue
            if days <= 0 or not np.isfinite(weight) or weight <= 0:
                continue
            out.append((days, weight))
    out.sort(key=lambda pair: pair[0])
    return out


def _build_rotation_scores(close_px: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    if close_px.empty:
        return pd.DataFrame()

    min_price = float(cfg.get("min_price", 1.0) or 1.0)
    min_rank_names = int(cfg.get("min_rank_names", 2) or 2)
    lookbacks = _parse_lookbacks(cfg)
    if not lookbacks:
        raise ValueError("Config must define at least one momentum lookback")

    close = close_px.astype(float)
    valid = np.isfinite(close.to_numpy(dtype=np.float64)) & (close.to_numpy(dtype=np.float64) >= min_price)

    score = np.zeros(close.shape, dtype=np.float32)
    score_den = np.zeros(close.shape, dtype=np.float32)

    for days, weight in lookbacks:
        lagged = _lag_frame(close, days)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (close / lagged) - 1.0
        rnk = _cross_section_rank_matrix(ret.to_numpy(dtype=np.float32), valid_mask=valid, higher_is_better=True, min_names=min_rank_names)
        mask = np.isfinite(rnk)
        if np.any(mask):
            score[mask] += (float(weight) * rnk[mask]).astype(np.float32)
            score_den[mask] += float(weight)

    vol_days = int(cfg.get("volatility_lookback_days", 0) or 0)
    vol_weight = float(cfg.get("volatility_weight", 0.0) or 0.0)
    if vol_days > 1 and vol_weight > 0.0:
        vol = close.pct_change().rolling(vol_days, min_periods=max(5, vol_days // 2)).std()
        vol_rank = _cross_section_rank_matrix(vol.to_numpy(dtype=np.float32), valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
        mask = np.isfinite(vol_rank)
        if np.any(mask):
            score[mask] += (float(vol_weight) * vol_rank[mask]).astype(np.float32)
            score_den[mask] += float(vol_weight)

    rev_days = int(cfg.get("short_term_reversal_days", 0) or 0)
    rev_weight = float(cfg.get("short_term_reversal_weight", 0.0) or 0.0)
    if rev_days > 0 and rev_weight > 0.0:
        lagged = _lag_frame(close, rev_days)
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = (close / lagged) - 1.0
        rev_rank = _cross_section_rank_matrix(ret.to_numpy(dtype=np.float32), valid_mask=valid, higher_is_better=False, min_names=min_rank_names)
        mask = np.isfinite(rev_rank)
        if np.any(mask):
            score[mask] += (float(rev_weight) * rev_rank[mask]).astype(np.float32)
            score_den[mask] += float(rev_weight)

    with np.errstate(invalid="ignore", divide="ignore"):
        score = score / score_den
    score[score_den <= 0.0] = np.nan

    abs_lb = int(cfg.get("absolute_momentum_lookback", 0) or 0)
    abs_min = float(cfg.get("absolute_momentum_min", 0.0) or 0.0)
    if abs_lb > 0:
        lagged = _lag_frame(close, abs_lb)
        with np.errstate(divide="ignore", invalid="ignore"):
            abs_ret = (close / lagged) - 1.0
        abs_ok = np.isfinite(abs_ret.to_numpy(dtype=np.float64)) & (abs_ret.to_numpy(dtype=np.float64) >= abs_min)
        score[~abs_ok] = np.nan

    trend_ma_days = int(cfg.get("trend_ma_days", 0) or 0)
    if trend_ma_days > 1:
        ma = close.rolling(trend_ma_days, min_periods=max(20, trend_ma_days // 2)).mean()
        ma_ok = np.isfinite(ma.to_numpy(dtype=np.float64)) & (close.to_numpy(dtype=np.float64) > ma.to_numpy(dtype=np.float64))
        score[~ma_ok] = np.nan

    fast_ma_days = int(cfg.get("fast_ma_days", 0) or 0)
    require_fast_above_trend = bool(cfg.get("require_fast_above_trend", False))
    if require_fast_above_trend and trend_ma_days > 1 and fast_ma_days > 1:
        fast_ma = close.rolling(fast_ma_days, min_periods=max(10, fast_ma_days // 2)).mean()
        slow_ma = close.rolling(trend_ma_days, min_periods=max(20, trend_ma_days // 2)).mean()
        stack_ok = np.isfinite(fast_ma.to_numpy(dtype=np.float64)) & np.isfinite(slow_ma.to_numpy(dtype=np.float64))
        stack_ok &= fast_ma.to_numpy(dtype=np.float64) > slow_ma.to_numpy(dtype=np.float64)
        score[~stack_ok] = np.nan

    min_score = cfg.get("min_score")
    if min_score is not None:
        try:
            threshold = float(min_score)
        except Exception:
            threshold = float("nan")
        if np.isfinite(threshold):
            score[score < threshold] = np.nan

    return pd.DataFrame(score, index=close.index, columns=close.columns)


def _aligned_close_series(
    close_px: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    symbol: str,
) -> pd.Series:
    sym = str(symbol or "").strip().upper()
    dates = pd.DatetimeIndex(close_px.index)
    if sym in close_px.columns:
        ser = pd.to_numeric(close_px[sym], errors="coerce")
        return ser.reindex(dates)
    frame = global_data.get(sym)
    if frame is None or frame.empty or "close" not in frame.columns:
        return pd.Series(index=dates, dtype=float)
    ser = pd.to_numeric(frame["close"], errors="coerce")
    idx = pd.DatetimeIndex(pd.to_datetime(ser.index, errors="coerce")).tz_localize(None).normalize()
    ser.index = idx
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser.reindex(dates)


def _build_proxy_close(
    close_px: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> pd.Series:
    exposure_cfg = dict(cfg.get("exposure_control") or {})
    dates = pd.DatetimeIndex(close_px.index)
    proxy_symbols = _normalize_symbol_list(exposure_cfg.get("proxy_symbols", []))
    proxy_symbol = str(exposure_cfg.get("proxy_symbol", "") or "").upper()
    if proxy_symbol:
        return _aligned_close_series(close_px, global_data, proxy_symbol)
    if proxy_symbols:
        cols = [sym for sym in proxy_symbols if sym in close_px.columns]
        if cols:
            proxy_close = close_px[cols].astype(float)
            proxy_ret = proxy_close.pct_change().replace([np.inf, -np.inf], np.nan)
            ew_ret = proxy_ret.mean(axis=1, skipna=True).fillna(0.0)
            proxy = (1.0 + ew_ret).cumprod()
            proxy.index = dates
            return proxy
    # Default to an equal-weight basket proxy for the scored ETF universe.
    proxy_close = close_px.astype(float)
    proxy_ret = proxy_close.pct_change().replace([np.inf, -np.inf], np.nan)
    ew_ret = proxy_ret.mean(axis=1, skipna=True).fillna(0.0)
    proxy = (1.0 + ew_ret).cumprod()
    proxy.index = dates
    return proxy


def _build_exposure_scalar(
    close_px: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    *,
    portfolio_proxy: Optional[pd.Series] = None,
) -> pd.Series:
    dates = pd.DatetimeIndex(close_px.index)
    exposure_cfg = dict(cfg.get("exposure_control") or {})
    base_gross = float(exposure_cfg.get("base_gross_exposure", cfg.get("base_gross_exposure", 1.0)) or 1.0)
    if not np.isfinite(base_gross):
        base_gross = 1.0
    min_gross = float(exposure_cfg.get("min_gross_exposure", 0.0) or 0.0)
    max_gross = float(exposure_cfg.get("max_gross_exposure", 1.0) or 1.0)
    min_gross = max(0.0, min(1.0, min_gross))
    max_gross = max(min_gross, min(1.0, max_gross))
    scalar = pd.Series(float(max(min_gross, min(max_gross, base_gross))), index=dates, dtype=float)

    if close_px.empty:
        return scalar

    use_portfolio_proxy = bool(exposure_cfg.get("use_portfolio_proxy", False))
    proxy = None
    if use_portfolio_proxy and isinstance(portfolio_proxy, pd.Series) and not portfolio_proxy.empty:
        proxy = pd.to_numeric(portfolio_proxy, errors="coerce").reindex(dates)
    if proxy is None:
        proxy = _build_proxy_close(close_px, global_data, cfg)
    proxy = pd.to_numeric(proxy, errors="coerce").reindex(dates)
    proxy_ret = proxy.pct_change().replace([np.inf, -np.inf], np.nan)

    vol_target = float(exposure_cfg.get("vol_target_annual", 0.0) or 0.0)
    vol_lookback = int(exposure_cfg.get("vol_lookback_days", 0) or 0)
    if vol_target > 0.0 and vol_lookback > 1:
        vol = proxy_ret.rolling(vol_lookback, min_periods=max(5, vol_lookback // 2)).std() * np.sqrt(252.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_scalar = vol_target / vol
        vol_scalar = pd.to_numeric(vol_scalar, errors="coerce").replace([np.inf, -np.inf], np.nan)
        vol_scalar = vol_scalar.clip(lower=min_gross, upper=max_gross).fillna(max_gross)
        scalar = np.minimum(scalar, vol_scalar)

    dd_start = float(exposure_cfg.get("drawdown_brake_start_pct", 0.0) or 0.0)
    dd_full = float(exposure_cfg.get("drawdown_brake_full_pct", 0.0) or 0.0)
    dd_min_gross = float(exposure_cfg.get("drawdown_min_gross_exposure", min_gross) or min_gross)
    dd_min_gross = max(0.0, min(max_gross, dd_min_gross))
    dd_lookback = int(exposure_cfg.get("drawdown_lookback_days", 0) or 0)
    if dd_full > dd_start >= 0.0:
        if dd_lookback > 1:
            peak = proxy.rolling(dd_lookback, min_periods=max(20, dd_lookback // 2)).max()
        else:
            peak = proxy.cummax()
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = 1.0 - (proxy / peak)
        dd = pd.to_numeric(dd, errors="coerce").fillna(0.0).clip(lower=0.0)
        dd_scalar = pd.Series(1.0, index=dates, dtype=float)
        full_mask = dd >= dd_full
        mid_mask = (dd > dd_start) & (dd < dd_full)
        dd_scalar.loc[full_mask] = dd_min_gross
        if bool(mid_mask.any()):
            frac = (dd.loc[mid_mask] - dd_start) / max(1e-9, (dd_full - dd_start))
            dd_scalar.loc[mid_mask] = 1.0 - frac * (1.0 - dd_min_gross)
        dd_scalar = dd_scalar.clip(lower=dd_min_gross, upper=1.0)
        scalar = np.minimum(scalar, dd_scalar)

    return pd.Series(pd.to_numeric(scalar, errors="coerce"), index=dates).fillna(base_gross).clip(lower=0.0, upper=max_gross)


def _build_combined_risk_scalar(
    close_px: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    *,
    portfolio_proxy: Optional[pd.Series] = None,
) -> pd.Series:
    market_scalar = _build_market_risk_scalar(global_data, close_px.index, cfg)
    exposure_scalar = _build_exposure_scalar(close_px, global_data, cfg, portfolio_proxy=portfolio_proxy)
    if isinstance(market_scalar, pd.Series):
        market_scalar = pd.to_numeric(market_scalar, errors="coerce").reindex(close_px.index).fillna(1.0)
    else:
        try:
            scalar_val = float(market_scalar)
        except Exception:
            scalar_val = 1.0
        if not np.isfinite(scalar_val):
            scalar_val = 1.0
        market_scalar = pd.Series(scalar_val, index=close_px.index, dtype=float)
    exposure_scalar = pd.to_numeric(exposure_scalar, errors="coerce").reindex(close_px.index).fillna(1.0)
    combined = (market_scalar * exposure_scalar).clip(lower=0.0, upper=1.0)
    return combined


def _build_allocator_state(
    close_px: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> pd.Series:
    allocator = dict(cfg.get("allocator") or {})
    if close_px.empty or not allocator:
        return pd.Series(dtype=object)

    dates = pd.DatetimeIndex(close_px.index)
    regime_symbol = str(allocator.get("regime_symbol", "SPY") or "SPY").upper()
    confirm_symbol = str(allocator.get("confirm_symbol", "QQQ") or "QQQ").upper()
    vix_symbol = str(allocator.get("vix_symbol", "VIX") or "VIX").upper()
    regime_ma_days = int(allocator.get("regime_ma_days", 200) or 200)
    breadth_ma_days = int(allocator.get("breadth_ma_days", regime_ma_days) or regime_ma_days)
    risk_on_breadth = float(allocator.get("risk_on_breadth", 0.6) or 0.6)
    risk_off_breadth = float(allocator.get("risk_off_breadth", 0.4) or 0.4)
    vix_risk_on_max = float(allocator.get("vix_risk_on_max", 20.0) or 20.0)
    vix_risk_off_min = float(allocator.get("vix_risk_off_min", 28.0) or 28.0)
    breadth_symbols = _normalize_symbol_list(
        allocator.get("breadth_symbols")
        or (cfg.get("sleeves", {}).get("aggressive", {}) if isinstance(cfg.get("sleeves"), Mapping) else {}).get("symbols", [])
    )

    regime_close = _aligned_close_series(close_px, global_data, regime_symbol)
    confirm_close = _aligned_close_series(close_px, global_data, confirm_symbol)
    vix_close = _aligned_close_series(close_px, global_data, vix_symbol)
    regime_ma = regime_close.rolling(regime_ma_days, min_periods=max(20, regime_ma_days // 2)).mean()
    confirm_ma = confirm_close.rolling(regime_ma_days, min_periods=max(20, regime_ma_days // 2)).mean()

    breadth = pd.Series(0.0, index=dates, dtype=float)
    if breadth_symbols:
        breadth_cols = [sym for sym in breadth_symbols if sym in close_px.columns]
        if breadth_cols:
            breadth_close = close_px[breadth_cols].astype(float)
            breadth_ma = breadth_close.rolling(breadth_ma_days, min_periods=max(20, breadth_ma_days // 2)).mean()
            with np.errstate(invalid="ignore"):
                breadth_mask = breadth_close.to_numpy(dtype=np.float64) > breadth_ma.to_numpy(dtype=np.float64)
            breadth = pd.Series(np.nanmean(breadth_mask.astype(float), axis=1), index=dates)

    state = pd.Series("neutral", index=dates, dtype=object)
    risk_on = (
        np.isfinite(regime_close.to_numpy(dtype=np.float64))
        & np.isfinite(regime_ma.to_numpy(dtype=np.float64))
        & (regime_close.to_numpy(dtype=np.float64) > regime_ma.to_numpy(dtype=np.float64))
        & np.isfinite(confirm_close.to_numpy(dtype=np.float64))
        & np.isfinite(confirm_ma.to_numpy(dtype=np.float64))
        & (confirm_close.to_numpy(dtype=np.float64) > confirm_ma.to_numpy(dtype=np.float64))
        & np.isfinite(breadth.to_numpy(dtype=np.float64))
        & (breadth.to_numpy(dtype=np.float64) >= risk_on_breadth)
    )
    if vix_close.notna().any():
        risk_on &= np.isfinite(vix_close.to_numpy(dtype=np.float64)) & (vix_close.to_numpy(dtype=np.float64) <= vix_risk_on_max)

    risk_off = (
        ~np.isfinite(regime_close.to_numpy(dtype=np.float64))
        | ~np.isfinite(regime_ma.to_numpy(dtype=np.float64))
        | (regime_close.to_numpy(dtype=np.float64) < regime_ma.to_numpy(dtype=np.float64))
        | ~np.isfinite(confirm_close.to_numpy(dtype=np.float64))
        | ~np.isfinite(confirm_ma.to_numpy(dtype=np.float64))
        | (confirm_close.to_numpy(dtype=np.float64) < confirm_ma.to_numpy(dtype=np.float64))
        | ~np.isfinite(breadth.to_numpy(dtype=np.float64))
        | (breadth.to_numpy(dtype=np.float64) <= risk_off_breadth)
    )
    if vix_close.notna().any():
        risk_off |= np.isfinite(vix_close.to_numpy(dtype=np.float64)) & (vix_close.to_numpy(dtype=np.float64) >= vix_risk_off_min)

    state.loc[risk_off] = "risk_off"
    state.loc[risk_on] = "risk_on"
    return state


def _build_weight_schedule_from_scores(
    scores: pd.DataFrame,
    *,
    target_count: int,
    gross_exposure: float,
    conviction_weighted: bool,
    conviction_power: float,
) -> pd.DataFrame:
    if scores is None or scores.empty:
        return pd.DataFrame()
    cols = [str(c) for c in scores.columns]
    out = pd.DataFrame(index=scores.index, columns=cols, dtype=float)
    gross = float(max(0.0, gross_exposure))
    if gross <= 0.0 or int(target_count) <= 0:
        return out
    for dt, row in scores.iterrows():
        ranked = pd.to_numeric(row, errors="coerce").dropna().sort_values(ascending=False)
        if ranked.empty:
            continue
        selected = list(ranked.head(int(target_count)).index)
        weights = _build_selected_target_weights(
            selected,
            ranked,
            conviction_weighted=bool(conviction_weighted),
            conviction_power=float(conviction_power),
        )
        if not weights:
            continue
        for sym, wt in weights.items():
            out.at[dt, str(sym)] = gross * float(wt)
    return out


def _build_signal_schedule_from_scores(
    scores: pd.DataFrame,
    *,
    rebalance_freq: str,
    target_count: int,
    hold_buffer_mult: float,
    conviction_weighted: bool,
    conviction_power: float,
    gross_exposure: float,
) -> pd.DataFrame:
    if scores is None or scores.empty:
        return pd.DataFrame()
    gross = float(max(0.0, gross_exposure))
    if gross <= 0.0 or int(target_count) <= 0:
        return pd.DataFrame(index=scores.index, columns=scores.columns, dtype=float)

    freq = str(rebalance_freq or "W").upper()
    if freq not in {"D", "W", "M", "Q"}:
        raise ValueError(f"Unsupported rebalance_freq '{rebalance_freq}'.")
    if freq == "D":
        signal_dates = list(pd.Index(scores.index).unique())
    else:
        period_freq = "W-FRI" if freq == "W" else freq
        signal_dates = list(scores.groupby(scores.index.to_period(period_freq)).tail(1).index)

    cols = [str(c) for c in scores.columns]
    out = pd.DataFrame(index=scores.index, columns=cols, dtype=float)
    existing_symbols: List[str] = []
    for dt in signal_dates:
        row = scores.loc[dt]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        ranked = pd.to_numeric(row, errors="coerce").dropna().sort_values(ascending=False)
        if ranked.empty:
            existing_symbols = []
            continue
        selected = select_target_portfolio(
            ranked,
            target_count=int(target_count),
            existing_symbols=existing_symbols,
            hold_buffer_mult=float(hold_buffer_mult),
        )
        existing_symbols = list(selected)
        if not selected:
            continue
        weights = _build_selected_target_weights(
            selected,
            ranked,
            conviction_weighted=bool(conviction_weighted),
            conviction_power=float(conviction_power),
        )
        if not weights:
            continue
        for sym, wt in weights.items():
            out.at[pd.Timestamp(dt), str(sym)] = gross * float(wt)
    return out


def _build_portfolio_proxy_series(
    close_px: pd.DataFrame,
    target_schedule: pd.DataFrame,
    *,
    execution_lag_days: int,
) -> pd.Series:
    dates = pd.DatetimeIndex(close_px.index)
    if close_px.empty or target_schedule is None or target_schedule.empty:
        return pd.Series(index=dates, dtype=float)
    weights = target_schedule.reindex(index=dates, columns=close_px.columns)
    weights = weights.ffill().fillna(0.0)
    lag_days = max(0, int(execution_lag_days))
    if lag_days > 0:
        weights = weights.shift(lag_days).fillna(0.0)
    returns = close_px.astype(float).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    proxy_ret = (weights * returns).sum(axis=1, skipna=True)
    proxy = (1.0 + proxy_ret).cumprod()
    proxy.index = dates
    return proxy


def _schedule_row_weights(frame: Optional[pd.DataFrame], dt: pd.Timestamp) -> Dict[str, float]:
    if frame is None or frame.empty or dt not in frame.index:
        return {}
    row = frame.loc[dt]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return _sanitize_target_weights(pd.to_numeric(row, errors="coerce").dropna().to_dict())


def _build_residual_defensive_target_schedule(
    data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
    global_data: Mapping[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    if not defensive_cfg or not bool(defensive_cfg.get("enabled", False)):
        raise ValueError("Residual defensive schedule requires enabled defensive_sleeve config")

    aggressive_symbols = _normalize_symbol_list(cfg.get("symbols", []))
    defensive_symbols = _normalize_symbol_list(defensive_cfg.get("symbols", []))
    if not aggressive_symbols or not defensive_symbols:
        raise ValueError("Residual defensive schedule requires aggressive and defensive symbols")

    aggressive_close = _price_frame_from_data(data, aggressive_symbols, "close")
    aggressive_open = _price_frame_from_data(data, aggressive_symbols, "open").reindex(aggressive_close.index)
    if aggressive_close.empty or aggressive_open.empty:
        raise RuntimeError("Failed to build aggressive ETF open/close matrices")

    aggressive_scores = _build_rotation_scores(aggressive_close, cfg)
    aggressive_schedule = _build_signal_schedule_from_scores(
        aggressive_scores,
        rebalance_freq=str(cfg.get("rebalance_freq", "W") or "W"),
        target_count=int(cfg.get("target_count", 1) or 1),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.0) or 1.0),
        conviction_weighted=bool(cfg.get("conviction_weighted", False)),
        conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        gross_exposure=1.0,
    )
    aggressive_proxy = _build_portfolio_proxy_series(
        aggressive_close,
        aggressive_schedule,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
    )
    aggressive_scalar = _build_combined_risk_scalar(
        aggressive_close,
        global_data,
        cfg,
        portfolio_proxy=aggressive_proxy,
    ).reindex(aggressive_close.index).ffill().fillna(1.0)

    defensive_merged_cfg = dict(cfg)
    defensive_merged_cfg.update(defensive_cfg)
    defensive_close = _price_frame_from_data(data, defensive_symbols, "close")
    defensive_open = _price_frame_from_data(data, defensive_symbols, "open").reindex(defensive_close.index)
    if defensive_close.empty or defensive_open.empty:
        raise RuntimeError("Failed to build defensive ETF open/close matrices")

    defensive_scores = _build_rotation_scores(defensive_close, defensive_merged_cfg)
    defensive_schedule = _build_signal_schedule_from_scores(
        defensive_scores,
        rebalance_freq=str(defensive_merged_cfg.get("rebalance_freq", cfg.get("rebalance_freq", "W")) or "W"),
        target_count=int(defensive_merged_cfg.get("target_count", 1) or 1),
        hold_buffer_mult=float(defensive_merged_cfg.get("hold_buffer_mult", cfg.get("hold_buffer_mult", 1.0)) or 1.0),
        conviction_weighted=bool(defensive_merged_cfg.get("conviction_weighted", False)),
        conviction_power=float(defensive_merged_cfg.get("conviction_power", 1.0) or 1.0),
        gross_exposure=1.0,
    )

    union_symbols = _normalize_symbol_list(aggressive_symbols + defensive_symbols)
    close_union = _price_frame_from_data(data, union_symbols, "close")
    open_union = _price_frame_from_data(data, union_symbols, "open").reindex(close_union.index)
    target_schedule = pd.DataFrame(index=close_union.index, columns=close_union.columns, dtype=float)

    signal_dates = sorted(
        set(aggressive_schedule.dropna(how="all").index).union(defensive_schedule.dropna(how="all").index)
    )
    for dt in signal_dates:
        hist = aggressive_scalar.loc[aggressive_scalar.index <= dt]
        scalar = float(hist.iloc[-1]) if not hist.empty else 1.0
        if not np.isfinite(scalar):
            scalar = 1.0
        scalar = float(np.clip(scalar, 0.0, 1.0))
        aggressive_row = _schedule_row_weights(aggressive_schedule, pd.Timestamp(dt))
        defensive_row = _schedule_row_weights(defensive_schedule, pd.Timestamp(dt))
        combined: Dict[str, float] = {}
        for sym in union_symbols:
            wt = scalar * float(aggressive_row.get(sym, 0.0))
            wt += (1.0 - scalar) * float(defensive_row.get(sym, 0.0))
            if wt > 0.0:
                combined[sym] = wt
        combined = _sanitize_target_weights(combined)
        for sym, wt in combined.items():
            target_schedule.at[pd.Timestamp(dt), str(sym)] = wt

    return close_union, open_union, target_schedule


def _build_allocator_target_schedule(
    data: Mapping[str, pd.DataFrame],
    cfg: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    sleeves = cfg.get("sleeves")
    if not isinstance(sleeves, Mapping) or not sleeves:
        raise ValueError("Allocator config requires non-empty 'sleeves'")

    all_symbols: List[str] = []
    sleeve_scores: Dict[str, pd.DataFrame] = {}
    sleeve_weights: Dict[str, pd.DataFrame] = {}
    close_frames: List[pd.DataFrame] = []
    open_frames: List[pd.DataFrame] = []

    for sleeve_name, sleeve_cfg_raw in sleeves.items():
        sleeve_cfg = dict(sleeve_cfg_raw or {})
        sleeve_symbols = _normalize_symbol_list(sleeve_cfg.get("symbols", []))
        if not sleeve_symbols:
            continue
        close_px = _price_frame_from_data(data, sleeve_symbols, "close")
        open_px = _price_frame_from_data(data, sleeve_symbols, "open")
        if close_px.empty or open_px.empty:
            continue
        close_frames.append(close_px)
        open_frames.append(open_px)
        all_symbols.extend(sleeve_symbols)

        merged_cfg = dict(cfg)
        merged_cfg.update(sleeve_cfg)
        scores = _build_rotation_scores(close_px, merged_cfg)
        sleeve_scores[str(sleeve_name)] = scores
        sleeve_weights[str(sleeve_name)] = _build_weight_schedule_from_scores(
            scores,
            target_count=int(sleeve_cfg.get("target_count", cfg.get("target_count", 1)) or 1),
            gross_exposure=float(sleeve_cfg.get("gross_exposure", 1.0) or 1.0),
            conviction_weighted=bool(sleeve_cfg.get("conviction_weighted", cfg.get("conviction_weighted", False))),
            conviction_power=float(sleeve_cfg.get("conviction_power", cfg.get("conviction_power", 1.0)) or 1.0),
        )

    if not close_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=object)

    union_symbols = _normalize_symbol_list(all_symbols)
    close_union = _price_frame_from_data(data, union_symbols, "close")
    open_union = _price_frame_from_data(data, union_symbols, "open").reindex(close_union.index)
    global_data = {sym: frame for sym, frame in data.items() if sym not in union_symbols}
    state = _build_allocator_state(close_union, global_data, cfg).reindex(close_union.index).ffill().fillna("neutral")

    target_schedule = pd.DataFrame(index=close_union.index, columns=close_union.columns, dtype=float)
    allocator = dict(cfg.get("allocator") or {})
    sleeve_map = {
        "risk_on": str(allocator.get("risk_on_sleeve", "aggressive") or "aggressive"),
        "neutral": str(allocator.get("neutral_sleeve", "neutral") or "neutral"),
        "risk_off": str(allocator.get("risk_off_sleeve", "defensive") or "defensive"),
    }

    for state_name, sleeve_name in sleeve_map.items():
        w = sleeve_weights.get(sleeve_name)
        if w is None or w.empty:
            continue
        mask = state == state_name
        if not bool(mask.any()):
            continue
        aligned = w.reindex(index=target_schedule.index, columns=target_schedule.columns)
        target_schedule.loc[mask] = aligned.loc[mask]

    return close_union, open_union, target_schedule, state


def _run_rotation_window(
    *,
    cfg: Mapping[str, Any],
    close_px: pd.DataFrame,
    open_px: pd.DataFrame,
    scores: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    win_close = close_px.loc[(close_px.index >= pd.Timestamp(start_date)) & (close_px.index <= pd.Timestamp(end_date))]
    win_open = open_px.reindex(win_close.index)
    win_scores = scores.reindex(win_close.index)
    base_target_schedule = _build_signal_schedule_from_scores(
        win_scores,
        rebalance_freq=str(cfg.get("rebalance_freq", "W") or "W"),
        target_count=int(cfg.get("target_count", 1) or 1),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.0) or 1.0),
        conviction_weighted=bool(cfg.get("conviction_weighted", False)),
        conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        gross_exposure=1.0,
    )
    portfolio_proxy = _build_portfolio_proxy_series(
        win_close,
        base_target_schedule,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
    )
    risk_scalar = _build_combined_risk_scalar(win_close, global_data, cfg, portfolio_proxy=portfolio_proxy)
    tx_bps = float(cfg.get("transaction_cost_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("entry_slippage_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("exit_slippage_bps", 0.0) or 0.0)

    return run_periodic_rebalance(
        prices=win_close,
        execution_prices=win_open,
        ranked_scores=win_scores,
        target_count=int(cfg.get("target_count", 1) or 1),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.0) or 1.0),
        transaction_cost_bps=tx_bps,
        start_cash=float(cfg.get("start_cash", 100000.0) or 100000.0),
        rebalance_freq=str(cfg.get("rebalance_freq", "W") or "W"),
        risk_scalar_by_date=risk_scalar,
        hard_stop_pct=cfg.get("hard_stop_pct"),
        trend_ma_days=cfg.get("trend_stop_ma_days"),
        time_stop_days=cfg.get("time_stop_days"),
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
        conviction_weighted=bool(cfg.get("conviction_weighted", False)),
        conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        position_cap=cfg.get("max_position_weight"),
        turnover_budget=cfg.get("turnover_budget"),
    )


def _run_rotation_window_with_targets(
    *,
    cfg: Mapping[str, Any],
    close_px: pd.DataFrame,
    open_px: pd.DataFrame,
    target_schedule: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    win_close = close_px.loc[(close_px.index >= pd.Timestamp(start_date)) & (close_px.index <= pd.Timestamp(end_date))]
    win_open = open_px.reindex(win_close.index)
    win_targets = target_schedule.reindex(win_close.index)
    portfolio_proxy = _build_portfolio_proxy_series(
        win_close,
        win_targets,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
    )
    risk_scalar = _build_combined_risk_scalar(win_close, global_data, cfg, portfolio_proxy=portfolio_proxy)
    tx_bps = float(cfg.get("transaction_cost_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("entry_slippage_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("exit_slippage_bps", 0.0) or 0.0)

    return run_periodic_rebalance(
        prices=win_close,
        execution_prices=win_open,
        ranked_scores=win_targets.fillna(0.0),
        target_weights_by_date=win_targets,
        target_count=max(1, int(cfg.get("target_count", 1) or 1)),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.0) or 1.0),
        transaction_cost_bps=tx_bps,
        start_cash=float(cfg.get("start_cash", 100000.0) or 100000.0),
        rebalance_freq=str(cfg.get("rebalance_freq", "W") or "W"),
        risk_scalar_by_date=risk_scalar,
        hard_stop_pct=cfg.get("hard_stop_pct"),
        trend_ma_days=cfg.get("trend_stop_ma_days"),
        time_stop_days=cfg.get("time_stop_days"),
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
        conviction_weighted=bool(cfg.get("conviction_weighted", False)),
        conviction_power=float(cfg.get("conviction_power", 1.0) or 1.0),
        position_cap=cfg.get("max_position_weight"),
        turnover_budget=cfg.get("turnover_budget"),
    )


def _result_payload(
    *,
    cfg: Mapping[str, Any],
    full: Dict[str, Any],
    oos24: Dict[str, Any],
    oos60: Dict[str, Any],
    symbols: Sequence[str],
    loaded_symbols: Sequence[str],
    start_date: str,
    end_date: str,
    allocator_state: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    audit = full.get("audit_report") or {}
    state_counts: Dict[str, int] = {}
    if isinstance(allocator_state, pd.Series) and not allocator_state.empty:
        counts = allocator_state.value_counts(dropna=False)
        state_counts = {str(k): int(v) for k, v in counts.items()}
    return {
        "strategy_name": str(cfg.get("name", "") or "ETF Rotation"),
        "symbols": list(symbols),
        "requested_symbols": int(len(symbols)),
        "loaded_symbols": int(len(loaded_symbols)),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "full_cagr_pct": float(_safe_float(full.get("cagr"), 0.0) * 100.0),
        "full_max_dd_pct": float(_pct_dd(full.get("max_drawdown_pct", 0.0))),
        "final_value": float(_safe_float(full.get("final_value"), 0.0)),
        "total_trades": int(full.get("total_trades", 0) or 0),
        "avg_annual_turnover_pct": float(_safe_float(full.get("avg_annual_turnover_pct"), float("nan"))),
        "oos_24_12_cagr_pct_warm": float(_safe_float(oos24.get("stitched_cagr_pct"), float("nan"))),
        "oos_24_12_max_dd_pct_warm": float(_safe_float(oos24.get("oos_max_dd_pct"), float("nan"))),
        "oos_60_12_cagr_pct_warm": float(_safe_float(oos60.get("stitched_cagr_pct"), float("nan"))),
        "oos_60_12_max_dd_pct_warm": float(_safe_float(oos60.get("oos_max_dd_pct"), float("nan"))),
        "audit_report": {
            "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
            "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
            "execution_lag_days": int(audit.get("execution_lag_days", cfg.get("execution_lag_days", 1)) or 1),
        },
        "allocator_state_counts": state_counts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-day ETF rotation walk-forward.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to ETF rotation JSON config")
    parser.add_argument("--start-date", default="2011-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=5000)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config).expanduser().resolve())
    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    if isinstance(sleeves, Mapping) and sleeves:
        sleeve_symbols: List[str] = []
        for sleeve_cfg in sleeves.values():
            if isinstance(sleeve_cfg, Mapping):
                sleeve_symbols.extend(_normalize_symbol_list(sleeve_cfg.get("symbols", [])))
        symbols = _normalize_symbol_list(sleeve_symbols)
    elif defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        symbols = _normalize_symbol_list(
            list(cfg.get("symbols", [])) + list(defensive_cfg.get("symbols", []))
        )
    else:
        symbols = _normalize_symbol_list(cfg.get("symbols", []))
    if not symbols:
        raise ValueError("Config must define a non-empty 'symbols' list or allocator sleeves")

    globals_cfg = _normalize_symbol_list(cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))
    all_requested = list(dict.fromkeys(symbols + globals_cfg))

    print(f"etf-rotation run: requested_symbols={len(symbols)} universe={','.join(symbols)}")
    prefer_yf = bool(cfg.get("prefer_yfinance", True))
    if prefer_yf:
        data = _download_yfinance_pack(all_requested, start_date=str(args.start_date), end_date=str(args.end_date))
    else:
        data = fetch_data_pack(all_requested, days=int(args.days), backtest_mode=True) or {}
    loaded_symbols = [sym for sym in symbols if sym in data]
    print(f"loaded_symbols={len(loaded_symbols)}")
    if len(loaded_symbols) < max(2, int(cfg.get("min_rank_names", 2) or 2)):
        raise RuntimeError("Insufficient ETF history loaded for ranking")
    global_data = {sym: data[sym] for sym in globals_cfg if sym in data}
    allocator_state = None

    if isinstance(sleeves, Mapping) and sleeves:
        close_px, open_px, target_schedule, allocator_state = _build_allocator_target_schedule(data, cfg)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError("Failed to build ETF allocator matrices")
        active = int(target_schedule.notna().any(axis=0).sum())
        print(f"active_scored_symbols={active}")
        full = _run_rotation_window_with_targets(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
    elif defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        close_px, open_px, target_schedule = _build_residual_defensive_target_schedule(data, cfg, global_data)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError("Failed to build ETF residual defensive matrices")
        active = int(target_schedule.notna().any(axis=0).sum())
        print(f"active_scored_symbols={active}")
        run_cfg = dict(cfg)
        run_cfg["market_regime"] = {"enabled": False}
        run_cfg["exposure_control"] = {}
        full = _run_rotation_window_with_targets(
            cfg=run_cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )
    else:
        close_px = _price_frame_from_data(data, loaded_symbols, "close")
        open_px = _price_frame_from_data(data, loaded_symbols, "open").reindex(close_px.index)
        if close_px.empty or open_px.empty:
            raise RuntimeError("Failed to build ETF open/close matrices")

        scores = _build_rotation_scores(close_px, cfg)
        active = int(scores.notna().any(axis=0).sum())
        print(f"active_scored_symbols={active}")
        if active == 0:
            raise RuntimeError("Rotation score builder produced no active ETFs")

        full = _run_rotation_window(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            scores=scores,
            global_data=global_data,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
        )

    windows_24 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=24, test_months=12)
    windows_60 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=60, test_months=12)
    oos24 = _stitch_test_windows_warm(full_run=full, windows=windows_24, test_months=12)
    oos60 = _stitch_test_windows_warm(full_run=full, windows=windows_60, test_months=12)

    payload = _result_payload(
        cfg=cfg,
        full=full,
        oos24=oos24,
        oos60=oos60,
        symbols=symbols,
        loaded_symbols=loaded_symbols,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        allocator_state=allocator_state,
    )
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
