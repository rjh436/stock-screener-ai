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

from data.custom_universe import list_cached_symbols, load_symbol_file
from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.rebalance_engine import (
    _build_selected_target_weights,
    run_periodic_rebalance,
    select_target_portfolio,
)
from scripts.run_etf_rotation_walkforward import (
    _build_combined_risk_scalar,
    _build_portfolio_proxy_series,
    _build_rotation_scores,
    _build_signal_schedule_from_scores,
    _download_yfinance_pack,
    _normalize_symbol_list,
    _price_frame_from_data,
    _schedule_row_weights,
)
from scripts.run_factor_walkforward import (
    _build_test_windows,
    _pct_dd,
    _safe_float,
    _stitch_test_windows_warm,
)


DEFAULT_CONFIG = ROOT / "config" / "sector_stock_momentum_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sector-filtered stock momentum walkforward.")
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
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {cfg_path}")
    return dict(cfg)


def _load_symbol_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(k).strip().upper(): str(v).strip()
        for k, v in payload.items()
        if str(k).strip() and str(v).strip()
    }


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    universe_key = str(args.universe).strip()
    if universe_key.lower() == "cache_all":
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        return list_cached_symbols(max_symbols=max_symbols)
    index_symbols = get_index_symbols(universe_key) or []
    if index_symbols:
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        index_symbols = _normalize_symbol_list(index_symbols)
        if max_symbols is not None:
            index_symbols = index_symbols[:max_symbols]
        return index_symbols
    raise ValueError(f"Unsupported universe preset: {args.universe}")


def _resolve_fetch_days(start_date: str, end_date: str, fallback_days: int) -> int:
    try:
        delta_days = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days)
        return max(int(fallback_days), max(365, delta_days + 420))
    except Exception:
        return int(max(365, fallback_days))


def _base_stock_valid_mask(
    close_px: pd.DataFrame,
    volume_px: pd.DataFrame,
    sector_map: Mapping[str, str],
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if close_px.empty or volume_px.empty:
        return pd.DataFrame(index=close_px.index, columns=close_px.columns, dtype=bool)

    close = close_px.astype(float)
    volume = volume_px.astype(float).reindex(close.index, columns=close.columns)
    adv20 = (close * volume).rolling(20, min_periods=5).mean()
    min_price = float(cfg.get("min_price", 10.0) or 10.0)
    max_price = float(cfg.get("max_price", 0.0) or 0.0)
    min_adv20 = float(cfg.get("min_adv20", 2_000_000.0) or 2_000_000.0)
    max_adv20 = float(cfg.get("max_adv20", 0.0) or 0.0)
    min_history = int(cfg.get("min_history_bars", 252) or 252)

    valid = close.notna() & volume.notna() & (close >= min_price)
    if max_price > 0:
        valid &= close <= max_price
    valid &= adv20 >= min_adv20
    if max_adv20 > 0:
        valid &= adv20 <= max_adv20
    history = close.notna().cumsum(axis=0)
    valid &= history >= int(min_history)

    sector_series = pd.Series({str(sym): str(sector_map.get(str(sym).upper(), "")).strip() for sym in close.columns})
    known_sector_cols = sector_series[sector_series != ""].index.tolist()
    valid = valid.reindex(columns=close.columns, fill_value=False)
    if known_sector_cols:
        valid.loc[:, [c for c in close.columns if c not in known_sector_cols]] = False
    else:
        valid.loc[:, :] = False
    return valid


def _signal_dates_from_schedule(schedule: pd.DataFrame) -> List[pd.Timestamp]:
    if schedule is None or schedule.empty:
        return []
    return [pd.Timestamp(dt).tz_localize(None) for dt in schedule.dropna(how="all").index]


def _build_sector_filtered_target_schedule(
    *,
    stock_scores: pd.DataFrame,
    stock_valid_mask: pd.DataFrame,
    sector_target_schedule: pd.DataFrame,
    sector_name_by_symbol: Mapping[str, str],
    sector_name_by_etf: Mapping[str, str],
    stock_cfg: Mapping[str, Any],
) -> pd.DataFrame:
    if stock_scores.empty or stock_valid_mask.empty or sector_target_schedule.empty:
        return pd.DataFrame(index=stock_scores.index, columns=stock_scores.columns, dtype=float)

    target_count = int(stock_cfg.get("target_count", 8) or 8)
    hold_buffer_mult = float(stock_cfg.get("hold_buffer_mult", 1.25) or 1.25)
    conviction_weighted = bool(stock_cfg.get("conviction_weighted", False))
    conviction_power = float(stock_cfg.get("conviction_power", 1.0) or 1.0)
    gross_exposure = float(stock_cfg.get("gross_exposure", 1.0) or 1.0)
    min_rank_names = int(stock_cfg.get("min_rank_names", 3) or 3)
    max_stocks_per_sector = int(stock_cfg.get("max_stocks_per_sector", 0) or 0)

    out = pd.DataFrame(index=stock_scores.index, columns=stock_scores.columns, dtype=float)
    existing_symbols: List[str] = []

    for dt in _signal_dates_from_schedule(sector_target_schedule):
        sector_row = _schedule_row_weights(sector_target_schedule, dt)
        if not sector_row:
            existing_symbols = []
            continue
        winning_sectors = {
            str(sector_name_by_etf.get(str(etf).upper(), "")).strip()
            for etf in sector_row.keys()
        }
        winning_sectors = {sec for sec in winning_sectors if sec}
        if not winning_sectors or dt not in stock_scores.index:
            existing_symbols = []
            continue

        score_row = pd.to_numeric(stock_scores.loc[dt], errors="coerce")
        valid_row = stock_valid_mask.loc[dt].astype(bool)
        candidate_sector_mask = pd.Series(
            {
                str(sym): str(sector_name_by_symbol.get(str(sym).upper(), "")).strip() in winning_sectors
                for sym in stock_scores.columns
            }
        )
        ranked = score_row[valid_row & candidate_sector_mask].dropna().sort_values(ascending=False)
        if ranked.empty or len(ranked) < min_rank_names:
            existing_symbols = []
            continue

        if max_stocks_per_sector > 0:
            keep: List[str] = []
            sector_counts: Dict[str, int] = {}
            for sym in ranked.index:
                sec = str(sector_name_by_symbol.get(str(sym).upper(), "")).strip()
                cnt = int(sector_counts.get(sec, 0))
                if cnt >= max_stocks_per_sector:
                    continue
                keep.append(str(sym))
                sector_counts[sec] = cnt + 1
            ranked = ranked.loc[keep]

        selected = select_target_portfolio(
            ranked,
            target_count=target_count,
            existing_symbols=existing_symbols,
            hold_buffer_mult=hold_buffer_mult,
        )
        existing_symbols = list(selected)
        if not selected:
            continue

        weights = _build_selected_target_weights(
            selected,
            ranked,
            conviction_weighted=conviction_weighted,
            conviction_power=conviction_power,
        )
        for sym, wt in weights.items():
            out.at[pd.Timestamp(dt), str(sym)] = gross_exposure * float(wt)
    return out


def _window_metrics(run: Mapping[str, Any]) -> Dict[str, float | int]:
    audit = dict(run.get("audit_report") or {})
    return {
        "cagr_pct": float(_safe_float(run.get("cagr"), 0.0) * 100.0),
        "max_dd_pct": float(_pct_dd(run.get("max_drawdown_pct", 0.0))),
        "avg_annual_turnover_pct": float(_safe_float(run.get("avg_annual_turnover_pct"), float("nan"))),
        "total_trades": int(run.get("total_trades", 0) or 0),
        "final_value": float(_safe_float(run.get("final_value"), 0.0)),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
    }


def _result_payload(
    *,
    cfg: Mapping[str, Any],
    full: Mapping[str, Any],
    oos24: Mapping[str, Any],
    oos60: Mapping[str, Any],
    requested_symbols: Sequence[str],
    loaded_stock_symbols: Sequence[str],
    sector_etf_symbols: Sequence[str],
    loaded_sector_etfs: Sequence[str],
    winning_sector_counts: Mapping[str, int],
    valid_stock_symbols: int,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    audit = dict(full.get("audit_report") or {})
    return {
        "strategy_name": str(cfg.get("name", "") or "Sector Stock Momentum"),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "requested_stock_symbols": int(len(requested_symbols)),
        "loaded_stock_symbols": int(len(loaded_stock_symbols)),
        "valid_stock_symbols": int(valid_stock_symbols),
        "sector_etf_symbols": list(sector_etf_symbols),
        "loaded_sector_etfs": list(loaded_sector_etfs),
        "winning_sector_counts": {str(k): int(v) for k, v in dict(winning_sector_counts).items()},
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


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    stock_symbols = _normalize_symbol_list(_resolve_symbols(args))
    if not stock_symbols:
        raise ValueError("No stock symbols resolved for sector-stock momentum run")

    sector_map = _load_symbol_map(ROOT / "config" / "sectors.json")
    industry_map = _load_symbol_map(ROOT / "config" / "industries.json")
    if not industry_map:
        industry_map = _load_symbol_map(ROOT / "config" / "industry.json")

    sector_etf_map = {
        str(k).strip().upper(): str(v).strip()
        for k, v in dict(cfg.get("sector_etf_map") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    sector_etf_symbols = _normalize_symbol_list(list(sector_etf_map.keys()))
    if not sector_etf_symbols:
        raise ValueError("Config must define non-empty sector_etf_map")
    global_symbols = _normalize_symbol_list(cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))

    fetch_days = _resolve_fetch_days(args.start_date, args.end_date, int(args.days))
    print(f"sector-stock run: requested_stock_symbols={len(stock_symbols)} sector_etfs={len(sector_etf_symbols)}")
    stock_data = fetch_data_pack(stock_symbols, days=fetch_days, backtest_mode=True) or {}
    etf_data = _download_yfinance_pack(
        _normalize_symbol_list(sector_etf_symbols + global_symbols),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
    )
    data: Dict[str, pd.DataFrame] = {}
    data.update(stock_data)
    data.update(etf_data)

    loaded_stock_symbols = [sym for sym in stock_symbols if sym in stock_data]
    loaded_sector_etfs = [sym for sym in sector_etf_symbols if sym in data]
    print(f"loaded_stock_symbols={len(loaded_stock_symbols)} loaded_sector_etfs={len(loaded_sector_etfs)}")
    if len(loaded_sector_etfs) < max(1, int(cfg.get("sector_target_count", 1) or 1)):
        raise RuntimeError("Insufficient sector ETF history loaded for sector selection")

    stock_close = _price_frame_from_data(stock_data, loaded_stock_symbols, "close")
    stock_open = _price_frame_from_data(stock_data, loaded_stock_symbols, "open").reindex(stock_close.index)
    stock_volume = _price_frame_from_data(stock_data, loaded_stock_symbols, "volume").reindex(stock_close.index)
    if stock_close.empty or stock_open.empty or stock_volume.empty:
        raise RuntimeError("Failed to build stock price matrices")

    sector_close = _price_frame_from_data(data, loaded_sector_etfs, "close")
    if sector_close.empty:
        raise RuntimeError("Failed to build sector ETF close matrix")
    global_data = {sym: data[sym] for sym in global_symbols if sym in data}

    stock_cfg = dict(cfg.get("stock_momentum") or {})
    sector_cfg = dict(cfg.get("sector_rotation") or {})
    if not stock_cfg or not sector_cfg:
        raise ValueError("Config must define stock_momentum and sector_rotation sections")

    stock_valid_mask = _base_stock_valid_mask(stock_close, stock_volume, sector_map, stock_cfg)
    valid_stock_symbols = int(stock_valid_mask.any(axis=0).sum())
    print(f"valid_stock_symbols={valid_stock_symbols}")
    if valid_stock_symbols == 0:
        raise RuntimeError("Base stock validity filter produced zero symbols")

    stock_scores = _build_rotation_scores(stock_close, stock_cfg).where(stock_valid_mask)
    sector_scores = _build_rotation_scores(sector_close, sector_cfg)
    sector_schedule = _build_signal_schedule_from_scores(
        sector_scores,
        rebalance_freq=str(cfg.get("rebalance_freq", sector_cfg.get("rebalance_freq", "W")) or "W"),
        target_count=int(cfg.get("sector_target_count", 2) or 2),
        hold_buffer_mult=float(cfg.get("sector_hold_buffer_mult", cfg.get("hold_buffer_mult", 1.1)) or 1.1),
        conviction_weighted=bool(cfg.get("sector_conviction_weighted", False)),
        conviction_power=float(cfg.get("sector_conviction_power", 1.0) or 1.0),
        gross_exposure=1.0,
    )
    if sector_schedule.dropna(how="all").empty:
        raise RuntimeError("Sector ETF schedule produced zero active signal dates")

    target_schedule = _build_sector_filtered_target_schedule(
        stock_scores=stock_scores,
        stock_valid_mask=stock_valid_mask,
        sector_target_schedule=sector_schedule,
        sector_name_by_symbol=sector_map,
        sector_name_by_etf=sector_etf_map,
        stock_cfg=stock_cfg,
    )
    active_stock_symbols = int(target_schedule.notna().any(axis=0).sum())
    print(f"active_scored_symbols={active_stock_symbols}")
    if active_stock_symbols == 0:
        raise RuntimeError("Sector-filtered stock schedule produced zero active symbols")

    sector_proxy = _build_portfolio_proxy_series(
        sector_close,
        sector_schedule,
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
    )
    risk_cfg = {
        "market_regime": dict(cfg.get("market_regime") or {}),
        "exposure_control": dict(cfg.get("exposure_control") or {}),
    }
    risk_scalar = _build_combined_risk_scalar(
        sector_close,
        global_data,
        risk_cfg,
        portfolio_proxy=sector_proxy,
    )

    tx_bps = float(cfg.get("transaction_cost_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("entry_slippage_bps", 0.0) or 0.0)
    tx_bps += float(cfg.get("exit_slippage_bps", 0.0) or 0.0)

    full = run_periodic_rebalance(
        prices=stock_close.loc[(stock_close.index >= pd.Timestamp(args.start_date)) & (stock_close.index <= pd.Timestamp(args.end_date))],
        execution_prices=stock_open.reindex(stock_close.index).loc[(stock_close.index >= pd.Timestamp(args.start_date)) & (stock_close.index <= pd.Timestamp(args.end_date))],
        ranked_scores=target_schedule.fillna(0.0).loc[(target_schedule.index >= pd.Timestamp(args.start_date)) & (target_schedule.index <= pd.Timestamp(args.end_date))],
        target_weights_by_date=target_schedule.loc[(target_schedule.index >= pd.Timestamp(args.start_date)) & (target_schedule.index <= pd.Timestamp(args.end_date))],
        target_count=max(1, int(stock_cfg.get("target_count", 8) or 8)),
        hold_buffer_mult=float(stock_cfg.get("hold_buffer_mult", 1.25) or 1.25),
        transaction_cost_bps=tx_bps,
        start_cash=float(cfg.get("start_cash", 100000.0) or 100000.0),
        rebalance_freq=str(cfg.get("rebalance_freq", sector_cfg.get("rebalance_freq", "W")) or "W"),
        risk_scalar_by_date=risk_scalar,
        hard_stop_pct=cfg.get("hard_stop_pct"),
        trend_ma_days=cfg.get("trend_stop_ma_days"),
        time_stop_days=cfg.get("time_stop_days"),
        execution_lag_days=int(cfg.get("execution_lag_days", 1) or 1),
        conviction_weighted=bool(stock_cfg.get("conviction_weighted", False)),
        conviction_power=float(stock_cfg.get("conviction_power", 1.0) or 1.0),
        position_cap=cfg.get("max_position_weight", stock_cfg.get("max_position_weight", 1.0)),
        sector_map=sector_map,
        sector_cap=float(cfg.get("sector_cap", 1.0) or 1.0),
        industry_map=industry_map,
        industry_cap=float(cfg.get("industry_cap", 1.0) or 1.0),
        turnover_budget=float(cfg.get("turnover_budget", 1.0) or 1.0),
    )

    windows_24 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=24, test_months=12)
    windows_60 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=60, test_months=12)
    oos24 = _stitch_test_windows_warm(full_run=full, windows=windows_24, test_months=12)
    oos60 = _stitch_test_windows_warm(full_run=full, windows=windows_60, test_months=12)

    winning_sector_counts: Dict[str, int] = {}
    for dt in _signal_dates_from_schedule(sector_schedule):
        sector_row = _schedule_row_weights(sector_schedule, dt)
        for etf in sector_row.keys():
            sec = str(sector_etf_map.get(str(etf).upper(), "")).strip()
            if not sec:
                continue
            winning_sector_counts[sec] = int(winning_sector_counts.get(sec, 0) + 1)

    payload = _result_payload(
        cfg=cfg,
        full=full,
        oos24=oos24,
        oos60=oos60,
        requested_symbols=stock_symbols,
        loaded_stock_symbols=loaded_stock_symbols,
        sector_etf_symbols=sector_etf_symbols,
        loaded_sector_etfs=loaded_sector_etfs,
        winning_sector_counts=winning_sector_counts,
        valid_stock_symbols=valid_stock_symbols,
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
