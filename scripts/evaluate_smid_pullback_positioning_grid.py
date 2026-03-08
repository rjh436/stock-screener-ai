#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_smid_pullback_holdout import (
    _coverage_gate_failures,
    _coverage_ratio,
    _safe_float,
    _window_metrics,
    build_smid_pullback_context,
)
from scripts.run_smid_pullback_walkforward import (
    _build_smid_pullback_scores,
    _load_config,
    _run_window,
    _slice_prices,
)


DEFAULT_CONFIG = ROOT / "config" / "smid_pullback_r3000_tb006_v1.json"


def _parse_csv_numbers(raw: str, cast):
    out = []
    for item in str(raw or "").split(","):
        item = str(item).strip()
        if not item:
            continue
        out.append(cast(item))
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep target_count / turnover_budget / conviction floors for the SMID pullback sleeve."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--train-start-date", default="2016-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3200)
    parser.add_argument("--target-counts", default="3,5,8,12")
    parser.add_argument("--turnover-budgets", default="0.03,0.06,0.09,0.12")
    parser.add_argument("--min-entry-scores", default="0,70,75,80")
    parser.add_argument("--prune-weight-floors", default="0.0,0.005,0.01")
    parser.add_argument("--min-universe-coverage", type=float, default=0.60)
    parser.add_argument("--min-daily-membership-coverage", type=float, default=0.90)
    parser.add_argument("--min-daily-membership-coverage-p10", type=float, default=0.85)
    parser.add_argument("--friction-multiplier", type=float, default=1.0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _position_stats(run: Mapping[str, Any], *, meaningful_weight_floor: float = 0.005) -> Dict[str, float | int]:
    counts: List[int] = []
    meaningful_counts: List[int] = []
    rebalance_log = list((run or {}).get("rebalance_log") or [])
    for entry in rebalance_log:
        weights = {
            str(sym): float(val)
            for sym, val in dict(entry.get("weights") or {}).items()
            if float(_safe_float(val, 0.0)) > 0.0
        }
        if not weights:
            continue
        counts.append(len(weights))
        meaningful_counts.append(sum(1 for val in weights.values() if float(val) >= float(meaningful_weight_floor)))
    if not counts:
        return {
            "avg_positions": 0.0,
            "median_positions": 0.0,
            "p90_positions": 0.0,
            "max_positions": 0,
            "avg_meaningful_positions": 0.0,
            "latest_positions": 0,
            "latest_meaningful_positions": 0,
        }
    counts_arr = np.asarray(counts, dtype=float)
    meaningful_arr = np.asarray(meaningful_counts, dtype=float)
    latest_weights = {
        str(sym): float(val)
        for sym, val in dict(rebalance_log[-1].get("weights") or {}).items()
        if float(_safe_float(val, 0.0)) > 0.0
    }
    return {
        "avg_positions": float(counts_arr.mean()),
        "median_positions": float(np.median(counts_arr)),
        "p90_positions": float(np.percentile(counts_arr, 90)),
        "max_positions": int(counts_arr.max()),
        "avg_meaningful_positions": float(meaningful_arr.mean()),
        "latest_positions": int(len(latest_weights)),
        "latest_meaningful_positions": int(sum(1 for val in latest_weights.values() if float(val) >= float(meaningful_weight_floor))),
    }


def _mutate_cfg(
    base_cfg: Mapping[str, Any],
    *,
    target_count: int,
    turnover_budget: float,
    min_entry_score: float,
    prune_weight_floor: float,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_cfg))
    cfg["target_count"] = int(target_count)
    cfg["turnover_budget"] = float(turnover_budget)
    cfg["max_position_weight"] = float(min(1.0, 1.1 / float(max(1, int(target_count)))))
    cfg["min_entry_score"] = float(min_entry_score)
    cfg["prune_weight_floor"] = float(prune_weight_floor)
    cfg["name"] = (
        f"{cfg.get('name', 'SMID Pullback')} "
        f"TC{int(target_count)} "
        f"TB{int(round(float(turnover_budget) * 1000)):03d} "
        f"MS{int(round(float(min_entry_score))):02d} "
        f"PF{int(round(float(prune_weight_floor) * 1000)):03d}"
    )
    if "friction" in cfg and isinstance(cfg["friction"], Mapping):
        friction = dict(cfg["friction"])
        mult = float(max(0.0, 1.0 * 1.0))
        for key in ("transaction_cost_bps", "entry_slippage_bps", "exit_slippage_bps"):
            friction[key] = float(friction.get(key, 0.0) or 0.0)
        cfg["friction"] = friction
    return cfg


def main() -> None:
    args = _parse_args()
    base_cfg = _load_config(args.config)
    target_counts = [int(x) for x in _parse_csv_numbers(args.target_counts, int)]
    turnover_budgets = [float(x) for x in _parse_csv_numbers(args.turnover_budgets, float)]
    min_entry_scores = [float(x) for x in _parse_csv_numbers(args.min_entry_scores, float)]
    prune_weight_floors = [float(x) for x in _parse_csv_numbers(args.prune_weight_floors, float)]
    if not target_counts or not turnover_budgets or not min_entry_scores or not prune_weight_floors:
        raise ValueError("No target_counts, turnover_budgets, min_entry_scores, or prune_weight_floors provided.")

    context = build_smid_pullback_context(
        universe=str(args.universe),
        start_date=str(args.train_start_date),
        end_date=str(args.holdout_end_date),
        days=int(args.days),
        config_paths=[args.config],
    )

    requested_symbols = list(context.get("requested_symbols") or [])
    loaded_symbols = list(context.get("loaded_symbols") or [])
    coverage_stats = dict(context.get("coverage_stats") or {})
    coverage_ratio = _coverage_ratio(requested_symbols, loaded_symbols)
    coverage_failures = _coverage_gate_failures(
        coverage_ratio=coverage_ratio,
        coverage_stats=coverage_stats,
        min_universe_coverage=float(args.min_universe_coverage),
        min_daily_membership_coverage=float(args.min_daily_membership_coverage),
        min_daily_membership_coverage_p10=float(args.min_daily_membership_coverage_p10),
    )
    if coverage_failures:
        raise RuntimeError("Coverage gate failed for positioning sweep: " + "; ".join(coverage_failures))

    features = context["features"]
    global_data = dict(context.get("global_data") or {})
    reports: List[Dict[str, Any]] = []

    for target_count in target_counts:
        for turnover_budget in turnover_budgets:
            for min_entry_score in min_entry_scores:
                for prune_weight_floor in prune_weight_floors:
                    cfg = _mutate_cfg(
                        base_cfg,
                        target_count=target_count,
                        turnover_budget=turnover_budget,
                        min_entry_score=min_entry_score,
                        prune_weight_floor=prune_weight_floor,
                    )
                    scores = _build_smid_pullback_scores(features, cfg)
                    if scores.empty:
                        reports.append(
                            {
                                "target_count": int(target_count),
                                "turnover_budget": float(turnover_budget),
                                "min_entry_score": float(min_entry_score),
                                "prune_weight_floor": float(prune_weight_floor),
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
                        start_date=str(args.train_start_date),
                        end_date=str(args.train_end_date),
                    )
                    holdout_run = _run_window(
                        cfg=cfg,
                        prices_close=price_frames["close"],
                        prices_open=price_frames["open"],
                        scores=scores,
                        global_data=global_data,
                        start_date=str(args.holdout_start_date),
                        end_date=str(args.holdout_end_date),
                    )
                    reports.append(
                        {
                            "strategy_name": str(cfg.get("name", "")),
                            "target_count": int(target_count),
                            "turnover_budget": float(turnover_budget),
                            "min_entry_score": float(min_entry_score),
                            "prune_weight_floor": float(prune_weight_floor),
                            "max_position_weight": float(cfg.get("max_position_weight", 0.0) or 0.0),
                            "train": _window_metrics(train_run),
                            "holdout": _window_metrics(holdout_run),
                            "holdout_position_stats": _position_stats(holdout_run),
                            "active_scored_symbols": int(len(scores.columns)),
                        }
                    )

    reports.sort(
        key=lambda r: (
            float((r.get("holdout") or {}).get("calmar", float("-inf"))),
            float((r.get("holdout") or {}).get("cagr_pct", float("-inf"))),
        ),
        reverse=True,
    )

    payload = {
        "base_config": str(Path(args.config).resolve()),
        "train_start_date": str(args.train_start_date),
        "train_end_date": str(args.train_end_date),
        "holdout_start_date": str(args.holdout_start_date),
        "holdout_end_date": str(args.holdout_end_date),
        "coverage_ratio": float(coverage_ratio),
        "coverage_stats": coverage_stats,
        "reports": reports,
        "best_by_holdout_calmar": reports[0] if reports else None,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
