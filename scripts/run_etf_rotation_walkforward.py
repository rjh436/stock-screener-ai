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
from execution.rebalance_engine import run_periodic_rebalance
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

    risk_scalar = _build_market_risk_scalar(global_data, win_close.index, cfg)
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
) -> Dict[str, Any]:
    audit = full.get("audit_report") or {}
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
    symbols = _normalize_symbol_list(cfg.get("symbols", []))
    if not symbols:
        raise ValueError("Config must define a non-empty 'symbols' list")

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

    close_px = _price_frame_from_data(data, loaded_symbols, "close")
    open_px = _price_frame_from_data(data, loaded_symbols, "open").reindex(close_px.index)
    if close_px.empty or open_px.empty:
        raise RuntimeError("Failed to build ETF open/close matrices")

    scores = _build_rotation_scores(close_px, cfg)
    active = int(scores.notna().any(axis=0).sum())
    print(f"active_scored_symbols={active}")
    if active == 0:
        raise RuntimeError("Rotation score builder produced no active ETFs")

    global_data = {sym: data[sym] for sym in globals_cfg if sym in data}
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
    )
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
