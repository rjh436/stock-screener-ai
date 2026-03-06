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
from data.universe import build_russell3000_membership_by_day, get_universe_symbols_pit_window_with_meta
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


DEFAULT_CONFIG = ROOT / "config" / "smid_pullback_broad_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run broad small/mid-cap pullback momentum walkforward.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
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
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config in {cfg_path}")
    return dict(payload)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    if str(args.universe).lower() == "cache_all":
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        return list_cached_symbols(max_symbols=max_symbols)
    if str(args.universe).strip().upper() == "RUSSELL3000":
        symbols, _source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", args.start_date, args.end_date)
        return list(symbols)
    raise ValueError(f"Unsupported universe preset: {args.universe}")


def _build_membership_mask(
    dates: Sequence[pd.Timestamp],
    symbols: Sequence[str],
    membership_by_day: Sequence[Sequence[str] | frozenset[str] | None],
) -> np.ndarray:
    sym_to_idx = {str(sym): i for i, sym in enumerate(symbols)}
    mask = np.zeros((len(dates), len(symbols)), dtype=bool)
    for i, members in enumerate(membership_by_day):
        if members is None:
            continue
        for sym in members:
            j = sym_to_idx.get(str(sym))
            if j is not None:
                mask[i, j] = True
    return mask


def _extract_feature_arrays(prepared, membership_by_day: Sequence[Sequence[str] | frozenset[str] | None] | None = None) -> Dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(list(getattr(prepared, "all_dates", []))))
    symbols = sorted(list(getattr(prepared, "enriched", {}).keys()))
    n_days = len(dates)
    n_syms = len(symbols)
    fields = (
        "open",
        "close",
        "vol_ma20",
        "sma20",
        "sma50",
        "sma200",
        "high52w",
        "ret_3m",
        "ret_6m",
        "adr_pct",
        "rs_rating",
    )
    arrays = {name: np.full((n_days, n_syms), np.nan, dtype=np.float32) for name in fields}

    for j, sym in enumerate(symbols):
        sd = prepared.enriched.get(sym)
        if sd is None:
            continue
        valid = (sd.gidx >= 0) & (sd.gidx < n_days)
        if not np.any(valid):
            continue
        idx = sd.gidx[valid]
        df = sd.df
        arrays["open"][idx, j] = sd.open[valid].astype(np.float32)
        arrays["close"][idx, j] = sd.close[valid].astype(np.float32)

        for key, col, alt in (
            ("vol_ma20", "vol_ma20", "vol_ma30"),
            ("sma20", "sma20", None),
            ("sma50", "sma50", None),
            ("sma200", "sma200", None),
            ("high52w", "high_52w", "high52w"),
            ("ret_3m", "ret_3m", None),
            ("ret_6m", "ret_6m", None),
            ("adr_pct", "adr_pct", None),
            ("rs_rating", "rs_rating", None),
        ):
            name = col if col in df.columns else alt
            if name and name in df.columns:
                arrays[key][idx, j] = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float32)[valid]

    if membership_by_day is not None:
        membership_mask = _build_membership_mask(dates, symbols, membership_by_day)
    else:
        membership_mask = np.isfinite(arrays["close"]) & (arrays["close"] > 0.0)
    return {"dates": dates, "symbols": symbols, "membership_mask": membership_mask, **arrays}


def _base_valid_mask(features: Dict[str, Any], cfg: Mapping[str, Any]) -> np.ndarray:
    close = np.asarray(features["close"], dtype=np.float32)
    vol_ma20 = np.asarray(features["vol_ma20"], dtype=np.float32)
    membership_mask = np.asarray(features["membership_mask"], dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        adv20 = close * vol_ma20
    min_price = float(cfg.get("min_price", 2.0) or 2.0)
    max_price = float(cfg.get("max_price", 80.0) or 80.0)
    min_adv20 = float(cfg.get("min_adv20", 1_000_000.0) or 1_000_000.0)
    max_adv20 = float(cfg.get("max_adv20", 0.0) or 0.0)
    min_history = int(cfg.get("min_history_bars", 252) or 252)

    valid = membership_mask & np.isfinite(close) & np.isfinite(adv20)
    valid &= close >= min_price
    if max_price > 0:
        valid &= close <= max_price
    valid &= adv20 >= min_adv20
    if max_adv20 > 0:
        valid &= adv20 <= max_adv20

    history = np.cumsum(np.isfinite(close), axis=0)
    valid &= history >= int(min_history)
    return valid


def _build_smid_pullback_scores(features: Dict[str, Any], cfg: Mapping[str, Any]) -> pd.DataFrame:
    dates = pd.DatetimeIndex(features["dates"])
    symbols = list(features["symbols"])
    close = np.asarray(features["close"], dtype=np.float32)
    sma20 = np.asarray(features["sma20"], dtype=np.float32)
    sma50 = np.asarray(features["sma50"], dtype=np.float32)
    sma200 = np.asarray(features["sma200"], dtype=np.float32)
    high52w = np.asarray(features["high52w"], dtype=np.float32)
    ret_3m = np.asarray(features["ret_3m"], dtype=np.float32)
    ret_6m = np.asarray(features["ret_6m"], dtype=np.float32)
    adr_pct = np.asarray(features["adr_pct"], dtype=np.float32)
    rs_rating = np.asarray(features["rs_rating"], dtype=np.float32)

    valid = _base_valid_mask(features, cfg)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_off_high = (close / high52w) - 1.0
        dist_sma20 = (close / sma20) - 1.0

    trend_mask = (
        np.isfinite(sma20)
        & np.isfinite(sma50)
        & np.isfinite(sma200)
        & (close > sma20)
        & (sma20 > sma50)
        & (sma50 > sma200)
    )
    pullback_mask = valid & trend_mask
    pullback_mask &= np.isfinite(rs_rating) & (rs_rating >= float(cfg.get("min_rs_rating", 90.0) or 90.0))
    pullback_mask &= np.isfinite(ret_3m) & (ret_3m >= float(cfg.get("min_ret_3m", 35.0) or 35.0))
    pullback_mask &= np.isfinite(ret_6m) & (ret_6m >= float(cfg.get("min_ret_6m", 50.0) or 50.0))
    pullback_mask &= np.isfinite(pct_off_high)
    pullback_mask &= pct_off_high >= float(cfg.get("min_pct_off_high", -0.18) or -0.18)
    pullback_mask &= pct_off_high <= float(cfg.get("max_pct_off_high", -0.03) or -0.03)
    pullback_mask &= np.isfinite(dist_sma20)
    pullback_mask &= dist_sma20 >= float(cfg.get("min_dist_sma20", -0.04) or -0.04)
    pullback_mask &= dist_sma20 <= float(cfg.get("max_dist_sma20", 0.10) or 0.10)
    pullback_mask &= np.isfinite(adr_pct) & (adr_pct >= float(cfg.get("min_adr_pct", 3.0) or 3.0))

    min_names = int(cfg.get("min_rank_names", 4) or 4)
    components = [
        (_cross_section_rank_matrix(rs_rating, valid_mask=pullback_mask, higher_is_better=True, min_names=min_names), float(cfg.get("rs_weight", 1.0) or 1.0)),
        (_cross_section_rank_matrix(ret_3m, valid_mask=pullback_mask, higher_is_better=True, min_names=min_names), float(cfg.get("ret_3m_weight", 0.8) or 0.8)),
        (_cross_section_rank_matrix(ret_6m, valid_mask=pullback_mask, higher_is_better=True, min_names=min_names), float(cfg.get("ret_6m_weight", 0.5) or 0.5)),
        (_cross_section_rank_matrix(np.abs(dist_sma20), valid_mask=pullback_mask, higher_is_better=False, min_names=min_names), float(cfg.get("dist_sma20_weight", 0.7) or 0.7)),
        (_cross_section_rank_matrix(pct_off_high, valid_mask=pullback_mask, higher_is_better=True, min_names=min_names), float(cfg.get("pct_off_high_weight", 0.4) or 0.4)),
        (_cross_section_rank_matrix(adr_pct, valid_mask=pullback_mask, higher_is_better=True, min_names=min_names), float(cfg.get("adr_weight", 0.2) or 0.2)),
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
        scores = numer / denom
    scores[~pullback_mask] = np.nan
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


def _run_window(
    *,
    cfg: Mapping[str, Any],
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
    tx_bps = float(friction_cfg.get("transaction_cost_bps", 15.0) or 15.0)
    tx_bps += float(friction_cfg.get("entry_slippage_bps", 10.0) or 10.0)
    tx_bps += float(friction_cfg.get("exit_slippage_bps", 10.0) or 10.0)
    return run_periodic_rebalance(
        prices=p_close,
        ranked_scores=s_win,
        execution_prices=p_open,
        rebalance_freq=str(cfg.get("rebalance_freq", "D") or "D"),
        target_count=int(cfg.get("target_count", 5) or 5),
        hold_buffer_mult=float(cfg.get("hold_buffer_mult", 1.2) or 1.2),
        position_cap=float(cfg.get("max_position_weight", 0.22) or 0.22),
        turnover_budget=float(cfg.get("turnover_budget", 0.12) or 0.12),
        transaction_cost_bps=tx_bps,
        start_cash=float(cfg.get("start_cash", 100000.0) or 100000.0),
        risk_scalar_by_date=risk_scalar,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
        conviction_weighted=bool(cfg.get("conviction_weighted", True)),
        conviction_power=float(cfg.get("conviction_power", 1.25) or 1.25),
    )


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    universe_name = str(args.universe).strip().upper()
    universe_source = "cache_all"
    symbols = _resolve_symbols(args)
    if not symbols:
        raise ValueError("No symbols resolved for smid pullback run.")
    membership_by_day = None

    print(f"smid-pullback run: requested_symbols={len(symbols)}")
    data = fetch_data_pack(symbols, days=int(args.days), backtest_mode=True) or {}
    loaded_symbols = sorted(data.keys())
    print(f"loaded_symbols={len(loaded_symbols)}")
    global_data = fetch_data_pack(["SPY", "VIX", "HYG", "LQD"], days=int(args.days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, loaded_symbols, start_date=args.start_date, global_data=global_data)
    print(f"prepared_symbols={len(prepared.enriched)} prepared_dates={len(prepared.all_dates)}")

    if universe_name == "RUSSELL3000":
        membership_by_day, membership_source = build_russell3000_membership_by_day(list(prepared.all_dates), allow_missing_days=False)
        if not membership_by_day or len(membership_by_day) != len(prepared.all_dates):
            raise RuntimeError(f"Failed to build Russell 3000 PIT membership timeline (source={membership_source}).")
        universe_source = str(membership_source)
    features = _extract_feature_arrays(prepared, membership_by_day=membership_by_day)
    scores = _build_smid_pullback_scores(features, cfg)
    active_columns = list(scores.columns)
    print(f"active_scored_symbols={len(active_columns)}")
    if not active_columns:
        raise RuntimeError("SMID pullback score builder produced no active symbols.")

    price_frames = _slice_prices(features, ["open", "close"], active_columns)
    full_run = _run_window(
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
        "strategy_name": str(cfg.get("name", "SMID Pullback Momentum") or "SMID Pullback Momentum"),
        "requested_symbols": int(len(symbols)),
        "loaded_symbols": int(len(loaded_symbols)),
        "prepared_symbols": int(len(prepared.enriched)),
        "active_scored_symbols": int(len(active_columns)),
        "universe": str(args.universe),
        "universe_source": universe_source,
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
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
