#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day, get_universe_symbols_pit_window_with_meta
from execution.engine import prepare_backtest_data
from scripts.run_factor_walkforward import _daily_membership_price_coverage, _pct_dd, _safe_float
from scripts.run_smid_pullback_walkforward import (
    _build_smid_pullback_scores,
    _extract_feature_arrays,
    _load_config,
    _run_window,
    _slice_prices,
)


DEFAULT_CONFIGS = [
    ROOT / "config" / "smid_pullback_broad_v1.json",
    ROOT / "config" / "smid_pullback_r3000_tb006_v1.json",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMID pullback configs on a train/holdout split.")
    parser.add_argument("--configs", nargs="*", default=[str(p) for p in DEFAULT_CONFIGS])
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--train-start-date", default="2016-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3200)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _window_metrics(run: Dict[str, Any]) -> Dict[str, float | int]:
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


def _resolve_symbols(universe: str, start_date: str, end_date: str) -> Tuple[List[str], str]:
    name = str(universe).strip().upper()
    if name != "RUSSELL3000":
        raise ValueError("Holdout evaluator currently supports only RUSSELL3000.")
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    return list(symbols), str(source)


def main() -> None:
    args = _parse_args()
    requested_symbols, symbol_source = _resolve_symbols(args.universe, args.train_start_date, args.holdout_end_date)
    print(f"smid holdout eval: requested_symbols={len(requested_symbols)} universe={args.universe} source={symbol_source}")

    quality_report: Dict[str, Any] = {}
    data = fetch_data_pack(
        requested_symbols,
        days=int(args.days),
        backtest_mode=True,
        quality_report=quality_report,
    ) or {}
    loaded_symbols = sorted(data.keys())
    print(f"loaded_symbols={len(loaded_symbols)}")
    global_data = fetch_data_pack(["SPY", "VIX", "HYG", "LQD"], days=int(args.days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, loaded_symbols, start_date=args.train_start_date, global_data=global_data)
    membership_by_day, membership_source = build_russell3000_membership_by_day(list(prepared.all_dates), allow_missing_days=False)
    if not membership_by_day or len(membership_by_day) != len(prepared.all_dates):
        raise RuntimeError(f"Failed to build Russell 3000 PIT membership timeline (source={membership_source}).")
    features = _extract_feature_arrays(prepared, membership_by_day=membership_by_day)
    coverage_stats = _daily_membership_price_coverage(features)
    if coverage_stats.get("mean") == coverage_stats.get("mean"):
        print(f"daily_pit_price_coverage_mean={float(coverage_stats['mean']):.1%}")

    reports: List[Dict[str, Any]] = []
    for cfg_path in args.configs:
        cfg = _load_config(cfg_path)
        scores = _build_smid_pullback_scores(features, cfg)
        if scores.empty:
            reports.append(
                {
                    "config": str(cfg_path),
                    "strategy_name": str(cfg.get("name", "") or Path(cfg_path).stem),
                    "error": "no_scores",
                }
            )
            continue

        price_frames = _slice_prices(features, ["open", "close"], list(scores.columns))
        train_run = _run_window(
            cfg=cfg,
            prices_close=price_frames["close"],
            prices_open=price_frames["open"],
            scores=scores,
            global_data=global_data,
            start_date=args.train_start_date,
            end_date=args.train_end_date,
        )
        holdout_run = _run_window(
            cfg=cfg,
            prices_close=price_frames["close"],
            prices_open=price_frames["open"],
            scores=scores,
            global_data=global_data,
            start_date=args.holdout_start_date,
            end_date=args.holdout_end_date,
        )
        reports.append(
            {
                "config": str(cfg_path),
                "strategy_name": str(cfg.get("name", "") or Path(cfg_path).stem),
                "requested_symbols": int(len(requested_symbols)),
                "loaded_symbols": int(len(loaded_symbols)),
                "prepared_symbols": int(len(prepared.enriched)),
                "active_scored_symbols": int(len(scores.columns)),
                "universe": str(args.universe),
                "universe_source": symbol_source,
                "membership_source": membership_source,
                "data_quality": {
                    "missing_symbols": int(quality_report.get("missing", 0) or 0),
                    "incomplete_history": int(quality_report.get("incomplete_history", 0) or 0),
                    "stale": int(quality_report.get("stale", 0) or 0),
                },
                "daily_membership_price_coverage": coverage_stats,
                "train": _window_metrics(train_run),
                "holdout": _window_metrics(holdout_run),
            }
        )

    payload = {
        "train_start_date": args.train_start_date,
        "train_end_date": args.train_end_date,
        "holdout_start_date": args.holdout_start_date,
        "holdout_end_date": args.holdout_end_date,
        "reports": reports,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
