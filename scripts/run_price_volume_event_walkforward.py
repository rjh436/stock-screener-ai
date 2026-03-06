#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.custom_universe import list_cached_symbols, load_symbol_file
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data
from execution.rebalance_engine import run_periodic_rebalance
from scripts.run_factor_walkforward import (
    _build_market_risk_scalar,
    _build_test_windows,
    _cross_section_rank_matrix,
    _pct_dd,
    _safe_float,
    _stitch_test_windows_warm,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next-day price/volume event walkforward on a broad custom universe.")
    parser.add_argument("--config", default=str(ROOT / "config" / "price_volume_event_gap_momo_v1.json"))
    parser.add_argument("--universe", default="cache_all")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbol-file", default="")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3000)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {cfg_path}")
    return cfg


def _cfg_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    raw = cfg.get(key, default)
    if raw is None:
        raw = default
    return float(raw)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    if str(args.universe).lower() == "cache_all":
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        return list_cached_symbols(max_symbols=max_symbols)
    raise ValueError(f"Unsupported universe preset: {args.universe}")


def _lag_matrix(mat: np.ndarray, periods: int) -> np.ndarray:
    out = np.roll(mat, int(periods), axis=0)
    out[: int(periods), :] = np.nan
    return out


def _extract_price_volume_feature_arrays(prepared) -> Dict[str, Any]:
    all_dates = pd.to_datetime(list(getattr(prepared, "all_dates", [])))
    symbols = sorted(list(getattr(prepared, "enriched", {}).keys()))
    n_days = len(all_dates)
    n_syms = len(symbols)

    open_ = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    high = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    low = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    close = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    volume = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    vol_ma20 = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    sma20 = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    sma50 = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    sma200 = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    high_52w = np.full((n_days, n_syms), np.nan, dtype=np.float32)

    for j, sym in enumerate(symbols):
        sd = prepared.enriched.get(sym)
        if sd is None:
            continue
        valid = (sd.gidx >= 0) & (sd.gidx < n_days)
        if not np.any(valid):
            continue
        idx = sd.gidx[valid]
        df = sd.df

        open_[idx, j] = sd.open[valid].astype(np.float32)
        high[idx, j] = sd.high[valid].astype(np.float32)
        low[idx, j] = sd.low[valid].astype(np.float32)
        close[idx, j] = sd.close[valid].astype(np.float32)
        volume[idx, j] = sd.volume[valid].astype(np.float32)
        high_52w[idx, j] = sd.high52w[valid].astype(np.float32)
        sma20[idx, j] = sd.sma20[valid].astype(np.float32)
        sma50[idx, j] = sd.sma50[valid].astype(np.float32)
        sma200[idx, j] = sd.sma200[valid].astype(np.float32)

        if "vol_ma20" in df.columns:
            vol_ma20[idx, j] = pd.to_numeric(df["vol_ma20"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        elif "vol_ma30" in df.columns:
            vol_ma20[idx, j] = pd.to_numeric(df["vol_ma30"], errors="coerce").to_numpy(dtype=np.float32)[valid]
        else:
            series = pd.Series(sd.volume, index=sd.df.index)
            vol_ma20[idx, j] = series.rolling(20, min_periods=5).mean().to_numpy(dtype=np.float32)[valid]

    membership_mask = np.isfinite(close) & (close > 0.0)
    return {
        "dates": pd.DatetimeIndex(all_dates),
        "symbols": symbols,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vol_ma20": vol_ma20,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "high_52w": high_52w,
        "membership_mask": membership_mask,
    }


def _base_valid_mask(features: Dict[str, Any], cfg: Dict[str, Any]) -> np.ndarray:
    close = np.asarray(features["close"], dtype=np.float32)
    vol_ma20 = np.asarray(features["vol_ma20"], dtype=np.float32)
    membership_mask = np.asarray(features["membership_mask"], dtype=bool)

    min_price = _cfg_float(cfg, "min_price", 5.0)
    max_price = _cfg_float(cfg, "max_price", 0.0)
    min_adv20 = _cfg_float(cfg, "min_adv20", 2_000_000.0)
    max_adv20 = _cfg_float(cfg, "max_adv20", 0.0)
    min_history = int(cfg.get("min_history_bars", 252) or 252)

    dollar_vol = close * vol_ma20
    valid = membership_mask & np.isfinite(close) & np.isfinite(dollar_vol)
    valid &= close >= min_price
    if max_price > 0:
        valid &= close <= max_price
    valid &= dollar_vol >= min_adv20
    if max_adv20 > 0:
        valid &= dollar_vol <= max_adv20

    hist = np.cumsum(np.isfinite(close), axis=0)
    valid &= hist >= int(min_history)
    return valid


def _carry_event_scores(initial_scores: np.ndarray, hold_days: int, decay: float) -> np.ndarray:
    n_days, n_syms = initial_scores.shape
    out = np.full((n_days, n_syms), np.nan, dtype=np.float32)
    active_score = np.full(n_syms, np.nan, dtype=np.float32)
    active_age = np.full(n_syms, int(hold_days) + 1, dtype=np.int32)
    decay = float(np.clip(decay, 0.0, 1.0))

    for i in range(n_days):
        new_mask = np.isfinite(initial_scores[i, :])
        if np.any(new_mask):
            active_score[new_mask] = initial_scores[i, new_mask]
            active_age[new_mask] = 0

        active_mask = np.isfinite(active_score) & (active_age < int(hold_days))
        if np.any(active_mask):
            day_scores = active_score.copy()
            if decay < 0.999999:
                day_scores[active_mask] = day_scores[active_mask] * np.power(decay, active_age[active_mask])
            day_scores[~active_mask] = np.nan
            out[i, :] = day_scores

        active_age[active_mask] += 1
        expired = active_age >= int(hold_days)
        active_score[expired] = np.nan

    return out


def _build_price_volume_event_scores(features: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(features["dates"])
    symbols = list(features["symbols"])
    open_ = np.asarray(features["open"], dtype=np.float32)
    high = np.asarray(features["high"], dtype=np.float32)
    low = np.asarray(features["low"], dtype=np.float32)
    close = np.asarray(features["close"], dtype=np.float32)
    volume = np.asarray(features["volume"], dtype=np.float32)
    vol_ma20 = np.asarray(features["vol_ma20"], dtype=np.float32)
    sma20 = np.asarray(features["sma20"], dtype=np.float32)
    sma50 = np.asarray(features["sma50"], dtype=np.float32)
    sma200 = np.asarray(features["sma200"], dtype=np.float32)
    high_52w = np.asarray(features["high_52w"], dtype=np.float32)

    valid = _base_valid_mask(features, cfg)
    prev_close = _lag_matrix(close, 1)
    sma50_prev = _lag_matrix(sma50, int(cfg.get("trend_slope_lookback", 20) or 20))

    with np.errstate(invalid="ignore", divide="ignore"):
        gap_pct = (open_ / prev_close) - 1.0
        day_return = (close / prev_close) - 1.0
        intraday_return = (close / open_) - 1.0
        volume_surge = volume / vol_ma20
        close_strength = (close - low) / np.maximum(high - low, 1e-6)
        momentum_20 = (close / _lag_matrix(close, 20)) - 1.0
        momentum_63 = (close / _lag_matrix(close, 63)) - 1.0
        proximity_52w = close / high_52w

    trend_required = bool(cfg.get("trend_required", True))
    trend_mask = np.isfinite(close) & np.isfinite(sma50) & np.isfinite(sma200)
    trend_mask &= close > sma50
    trend_mask &= sma50 > sma200
    trend_mask &= sma50 > sma50_prev
    if not trend_required:
        trend_mask = np.isfinite(close)

    trigger_mode = str(cfg.get("trigger_mode", "gap_or_day_return") or "gap_or_day_return").lower()
    min_gap_pct = _cfg_float(cfg, "min_gap_pct", 0.03)
    max_gap_pct = _cfg_float(cfg, "max_gap_pct", 0.0)
    min_day_return = _cfg_float(cfg, "min_day_return", 0.05)
    max_day_return = _cfg_float(cfg, "max_day_return", 0.0)
    min_intraday_return = _cfg_float(cfg, "min_intraday_return", 0.0)
    min_volume_surge = _cfg_float(cfg, "min_volume_surge", 2.0)
    min_close_strength = _cfg_float(cfg, "min_close_strength", 0.65)
    min_momentum_20 = _cfg_float(cfg, "min_momentum_20", 0.0)
    min_momentum_63 = _cfg_float(cfg, "min_momentum_63", 0.10)
    min_proximity_52w = _cfg_float(cfg, "min_proximity_52w", 0.80)

    trigger = valid & trend_mask
    if trigger_mode == "gap_only":
        trigger &= np.isfinite(gap_pct) & (gap_pct >= min_gap_pct)
    elif trigger_mode == "day_return_only":
        trigger &= np.isfinite(day_return) & (day_return >= min_day_return)
    else:
        trigger &= (
            (np.isfinite(gap_pct) & (gap_pct >= min_gap_pct))
            | (np.isfinite(day_return) & (day_return >= min_day_return))
        )
    if max_gap_pct > 0:
        trigger &= np.isfinite(gap_pct) & (gap_pct <= max_gap_pct)
    if max_day_return > 0:
        trigger &= np.isfinite(day_return) & (day_return <= max_day_return)

    trigger &= np.isfinite(volume_surge) & (volume_surge >= min_volume_surge)
    trigger &= np.isfinite(close_strength) & (close_strength >= min_close_strength)
    if bool(cfg.get("require_green_day", True)):
        trigger &= np.isfinite(open_) & (close > open_)
    if min_intraday_return > 0:
        trigger &= np.isfinite(intraday_return) & (intraday_return >= min_intraday_return)
    if bool(cfg.get("require_above_sma20", True)):
        trigger &= np.isfinite(sma20) & (close >= sma20)
    if min_momentum_20 > 0:
        trigger &= np.isfinite(momentum_20) & (momentum_20 >= min_momentum_20)
    if min_momentum_63 > 0:
        trigger &= np.isfinite(momentum_63) & (momentum_63 >= min_momentum_63)
    if min_proximity_52w > 0:
        trigger &= np.isfinite(proximity_52w) & (proximity_52w >= min_proximity_52w)

    min_names = int(cfg.get("min_event_names", 1) or 1)
    components = [
        (_cross_section_rank_matrix(gap_pct, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("gap_pct_weight", 1.0) or 1.0)),
        (_cross_section_rank_matrix(day_return, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("day_return_weight", 0.9) or 0.9)),
        (_cross_section_rank_matrix(intraday_return, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("intraday_return_weight", 0.5) or 0.5)),
        (_cross_section_rank_matrix(volume_surge, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("volume_surge_weight", 0.8) or 0.8)),
        (_cross_section_rank_matrix(momentum_20, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("momentum_20_weight", 0.3) or 0.3)),
        (_cross_section_rank_matrix(momentum_63, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("momentum_63_weight", 0.6) or 0.6)),
        (_cross_section_rank_matrix(proximity_52w, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("proximity_52w_weight", 0.5) or 0.5)),
        (_cross_section_rank_matrix(close_strength, valid_mask=trigger, higher_is_better=True, min_names=min_names), float(cfg.get("close_strength_weight", 0.5) or 0.5)),
    ]

    numer = np.zeros(close.shape, dtype=np.float32)
    denom = np.zeros(close.shape, dtype=np.float32)
    for ranked, weight in components:
        if weight <= 0:
            continue
        mask = np.isfinite(ranked)
        numer[mask] += ranked[mask].astype(np.float32) * float(weight)
        denom[mask] += float(weight)

    with np.errstate(invalid="ignore", divide="ignore"):
        initial_scores = numer / denom
    initial_scores[~trigger] = np.nan

    hold_days = int(cfg.get("hold_days", 12) or 12)
    decay = float(cfg.get("score_decay", 0.985) or 0.985)
    scores = _carry_event_scores(initial_scores, hold_days=hold_days, decay=decay)

    active_cols = np.isfinite(scores).any(axis=0)
    if not np.any(active_cols):
        return pd.DataFrame(index=dates)
    return pd.DataFrame(scores[:, active_cols], index=dates, columns=np.array(symbols)[active_cols])


def _slice_prices(features: Dict[str, Any], columns: Sequence[str], active_columns: Sequence[str]) -> Dict[str, pd.DataFrame]:
    symbols = list(features["symbols"])
    dates = pd.DatetimeIndex(features["dates"])
    active_lookup = [symbols.index(sym) for sym in active_columns]
    out: Dict[str, pd.DataFrame] = {}
    for name in columns:
        arr = np.asarray(features[name], dtype=np.float32)[:, active_lookup]
        out[name] = pd.DataFrame(arr, index=dates, columns=active_columns)
    return out


def _run_price_volume_window(
    *,
    cfg: Dict[str, Any],
    prices_close: pd.DataFrame,
    prices_open: pd.DataFrame,
    scores: pd.DataFrame,
    global_data: Mapping[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    p_close = prices_close.loc[(prices_close.index >= pd.Timestamp(start_date)) & (prices_close.index <= pd.Timestamp(end_date))]
    p_open = prices_open.reindex(p_close.index)
    s_win = scores.loc[(scores.index >= pd.Timestamp(start_date)) & (scores.index <= pd.Timestamp(end_date))]
    risk_scalar = _build_market_risk_scalar(global_data, p_close.index, cfg)

    friction_cfg = cfg.get("friction", {}) or {}
    tx_bps = float(friction_cfg.get("transaction_cost_bps", 10.0) or 10.0)
    tx_bps += float(friction_cfg.get("entry_slippage_bps", 15.0) or 15.0)
    tx_bps += float(friction_cfg.get("exit_slippage_bps", 20.0) or 20.0)

    return run_periodic_rebalance(
        prices=p_close,
        ranked_scores=s_win,
        execution_prices=p_open,
        rebalance_freq=str(cfg.get("rebalance_freq", "D") or "D"),
        target_count=int(cfg.get("target_count", 8) or 8),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.10) or 1.10),
        position_cap=float(cfg.get("max_position_weight", 0.15) or 0.15),
        turnover_budget=float(cfg.get("turnover_budget", 1.0) or 1.0),
        transaction_cost_bps=tx_bps,
        start_cash=100000.0,
        risk_scalar_by_date=risk_scalar,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
        conviction_weighted=bool(cfg.get("conviction_weighted", True)),
        conviction_power=float(cfg.get("conviction_power", 1.25) or 1.25),
    )


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    symbols = _resolve_symbols(args)
    if not symbols:
        raise ValueError("No symbols resolved for price/volume event run.")

    global_symbols = ["SPY", "VIX", "$VIX", "HYG", "LQD"]
    print(f"price-volume-event run: requested_symbols={len(symbols)}")
    data = fetch_data_pack(symbols, days=int(args.days), backtest_mode=True) or {}
    loaded_symbols = sorted(data.keys())
    print(f"loaded_symbols={len(loaded_symbols)}")
    global_data = fetch_data_pack(global_symbols, days=int(args.days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, loaded_symbols, start_date=args.start_date, global_data=global_data)
    print(f"prepared_symbols={len(prepared.enriched)} prepared_dates={len(prepared.all_dates)}")

    features = _extract_price_volume_feature_arrays(prepared)
    scores = _build_price_volume_event_scores(features, cfg)
    active_columns = list(scores.columns)
    print(f"active_scored_symbols={len(active_columns)}")
    if not active_columns:
        raise RuntimeError("Price/volume event score builder produced no active symbols.")

    price_frames = _slice_prices(features, ["open", "close"], active_columns)
    full_run = _run_price_volume_window(
        cfg=cfg,
        prices_close=price_frames["close"],
        prices_open=price_frames["open"],
        scores=scores,
        global_data=global_data,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    windows_24_12 = _build_test_windows(args.start_date, args.end_date, train_months=24, test_months=12)
    windows_60_12 = _build_test_windows(args.start_date, args.end_date, train_months=60, test_months=12)
    stitched_24_12_warm = _stitch_test_windows_warm(full_run=full_run, windows=windows_24_12, test_months=12)
    stitched_60_12_warm = _stitch_test_windows_warm(full_run=full_run, windows=windows_60_12, test_months=12)
    audit = dict(full_run.get("audit_report") or {})

    payload = {
        "strategy_name": str(cfg.get("name", "Price Volume Event Momentum") or "Price Volume Event Momentum"),
        "requested_symbols": int(len(symbols)),
        "loaded_symbols": int(len(loaded_symbols)),
        "prepared_symbols": int(len(prepared.enriched)),
        "active_scored_symbols": int(len(active_columns)),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "full_cagr_pct": float(_safe_float(full_run.get("cagr"), 0.0) * 100.0),
        "full_max_dd_pct": float(_pct_dd(full_run.get("max_drawdown_pct", 0.0))),
        "total_trades": int(full_run.get("total_trades", 0) or 0),
        "final_value": float(full_run.get("final_value", 0.0) or 0.0),
        "avg_annual_turnover_pct": float(_safe_float(full_run.get("avg_annual_turnover_pct"), float("nan"))),
        "oos_24_12_cagr_pct_warm": float(_safe_float(stitched_24_12_warm.get("stitched_cagr_pct"), float("nan"))),
        "oos_24_12_max_dd_pct_warm": float(_safe_float(stitched_24_12_warm.get("oos_max_dd_pct"), float("nan"))),
        "oos_60_12_cagr_pct_warm": float(_safe_float(stitched_60_12_warm.get("stitched_cagr_pct"), float("nan"))),
        "oos_60_12_max_dd_pct_warm": float(_safe_float(stitched_60_12_warm.get("oos_max_dd_pct"), float("nan"))),
        "audit_report": {
            "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
            "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
            "execution_lag_days": int(audit.get("execution_lag_days", cfg.get("execution_lag_days", 1)) or 1),
        },
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
