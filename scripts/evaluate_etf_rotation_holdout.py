#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_etf_rotation_walkforward import (
    _build_allocator_target_schedule,
    _build_residual_defensive_target_schedule,
    _build_rotation_scores,
    _download_yfinance_pack,
    _load_config,
    _normalize_symbol_list,
    _price_frame_from_data,
    _run_rotation_window,
    _run_rotation_window_with_targets,
)
from scripts.run_factor_walkforward import _pct_dd, _safe_float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ETF configs on a train/holdout split.")
    parser.add_argument(
        "--configs",
        default="",
        help="Comma-separated ETF config paths. Defaults to the current deployable leaders.",
    )
    parser.add_argument("--start-date", default="2011-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _default_config_paths() -> List[Path]:
    return [
        ROOT / "config" / "etf_rotation_growth_core5_usable_top2_v1.json",
        ROOT / "config" / "etf_rotation_growth_core5_residual_defensive_calmar_v1.json",
        ROOT / "config" / "etf_rotation_growth_core5_residual_defensive_top1_v1.json",
    ]


def _resolve_config_paths(raw: str) -> List[Path]:
    if str(raw or "").strip():
        return [Path(part.strip()).expanduser().resolve() for part in str(raw).split(",") if part.strip()]
    return [path.resolve() for path in _default_config_paths()]


def _config_symbols(cfg: Mapping[str, Any]) -> List[str]:
    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    if isinstance(sleeves, Mapping) and sleeves:
        sleeve_symbols: List[str] = []
        for sleeve_cfg in sleeves.values():
            if isinstance(sleeve_cfg, Mapping):
                sleeve_symbols.extend(_normalize_symbol_list(sleeve_cfg.get("symbols", [])))
        return _normalize_symbol_list(sleeve_symbols)
    if defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        return _normalize_symbol_list(list(cfg.get("symbols", [])) + list(defensive_cfg.get("symbols", [])))
    return _normalize_symbol_list(cfg.get("symbols", []))


def _requested_symbols(cfgs: Sequence[Mapping[str, Any]]) -> List[str]:
    requested: List[str] = []
    for cfg in cfgs:
        requested.extend(_config_symbols(cfg))
        requested.extend(_normalize_symbol_list(cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"])))
    return _normalize_symbol_list(requested)


def _window_metrics(run: Mapping[str, Any]) -> Dict[str, float | int]:
    cagr_pct = float(_safe_float(run.get("cagr"), 0.0) * 100.0)
    dd_pct = float(_pct_dd(run.get("max_drawdown_pct", 0.0)))
    audit = dict(run.get("audit_report") or {})
    return {
        "cagr_pct": cagr_pct,
        "max_dd_pct": dd_pct,
        "calmar": float(cagr_pct / dd_pct) if dd_pct > 0 else float("nan"),
        "avg_annual_turnover_pct": float(_safe_float(run.get("avg_annual_turnover_pct"), float("nan"))),
        "total_trades": int(run.get("total_trades", 0) or 0),
        "final_value": float(_safe_float(run.get("final_value"), 0.0)),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
    }


def _evaluate_config(
    cfg: Mapping[str, Any],
    data: Mapping[str, Any],
    *,
    start_date: str,
    train_end_date: str,
    holdout_start_date: str,
    holdout_end_date: str,
) -> Dict[str, Any]:
    symbols = _config_symbols(cfg)
    global_symbols = _normalize_symbol_list(cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))
    global_data = {sym: data[sym] for sym in global_symbols if sym in data}
    loaded_symbols = [sym for sym in symbols if sym in data]
    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})

    if isinstance(sleeves, Mapping) and sleeves:
        close_px, open_px, target_schedule, _ = _build_allocator_target_schedule(data, cfg)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError(f"Failed to build allocator matrices for {cfg.get('name')}")
        train_run = _run_rotation_window_with_targets(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=start_date,
            end_date=train_end_date,
        )
        holdout_run = _run_rotation_window_with_targets(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=holdout_start_date,
            end_date=holdout_end_date,
        )
    elif defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        close_px, open_px, target_schedule = _build_residual_defensive_target_schedule(data, cfg, global_data)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError(f"Failed to build residual defensive matrices for {cfg.get('name')}")
        run_cfg = dict(cfg)
        run_cfg["market_regime"] = {"enabled": False}
        run_cfg["exposure_control"] = {}
        train_run = _run_rotation_window_with_targets(
            cfg=run_cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=start_date,
            end_date=train_end_date,
        )
        holdout_run = _run_rotation_window_with_targets(
            cfg=run_cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=holdout_start_date,
            end_date=holdout_end_date,
        )
    else:
        close_px = _price_frame_from_data(data, loaded_symbols, "close")
        open_px = _price_frame_from_data(data, loaded_symbols, "open").reindex(close_px.index)
        if close_px.empty or open_px.empty:
            raise RuntimeError(f"Failed to build ETF matrices for {cfg.get('name')}")
        scores = _build_rotation_scores(close_px, cfg)
        train_run = _run_rotation_window(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            scores=scores,
            global_data=global_data,
            start_date=start_date,
            end_date=train_end_date,
        )
        holdout_run = _run_rotation_window(
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            scores=scores,
            global_data=global_data,
            start_date=holdout_start_date,
            end_date=holdout_end_date,
        )

    train_metrics = _window_metrics(train_run)
    holdout_metrics = _window_metrics(holdout_run)
    return {
        "config_name": str(cfg.get("name", "") or Path(str(cfg.get("_config_path", "config"))).stem),
        "config_path": str(cfg.get("_config_path", "")),
        "symbols": symbols,
        "loaded_symbols": loaded_symbols,
        "train_start_date": str(start_date),
        "train_end_date": str(train_end_date),
        "holdout_start_date": str(holdout_start_date),
        "holdout_end_date": str(holdout_end_date),
        "train": train_metrics,
        "holdout": holdout_metrics,
    }


def main() -> None:
    args = _parse_args()
    config_paths = _resolve_config_paths(args.configs)
    cfgs: List[Dict[str, Any]] = []
    for path in config_paths:
        cfg = _load_config(path)
        cfg["_config_path"] = str(path)
        cfgs.append(cfg)

    requested = _requested_symbols(cfgs)
    print(f"holdout-eval: configs={len(cfgs)} requested_symbols={len(requested)}")
    data = _download_yfinance_pack(
        requested,
        start_date=str(args.start_date),
        end_date=str(args.holdout_end_date),
    )

    rows = [
        _evaluate_config(
            cfg,
            data,
            start_date=str(args.start_date),
            train_end_date=str(args.train_end_date),
            holdout_start_date=str(args.holdout_start_date),
            holdout_end_date=str(args.holdout_end_date),
        )
        for cfg in cfgs
    ]

    payload = {
        "start_date": str(args.start_date),
        "train_end_date": str(args.train_end_date),
        "holdout_start_date": str(args.holdout_start_date),
        "holdout_end_date": str(args.holdout_end_date),
        "results": rows,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
