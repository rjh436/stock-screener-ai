#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_smid_pullback_holdout import (
    DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
    DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
    DEFAULT_MIN_UNIVERSE_COVERAGE,
    _coverage_gate_failures,
    _coverage_ratio,
    build_smid_pullback_context,
)
from scripts.run_smid_pullback_walkforward import (
    _build_smid_pullback_scores,
    _load_config,
    _run_window,
    _slice_prices,
)


DEFAULT_CONFIG = ROOT / "config" / "smid_pullback_r3000_tc4_tb003_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-turnover discrete SMID pullback variants.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-start-date", default="2016-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3200)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _variant_name(cfg: Dict[str, Any]) -> str:
    return (
        f"{cfg['rebalance_freq']}_tc{cfg['target_count']}"
        f"_hb{str(cfg['hold_buffer_mult']).replace('.', '')}"
        f"_hold{cfg['min_hold_days']}"
        f"_ms{int(cfg['min_entry_score'])}"
    )


def _build_variants(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants: List[Dict[str, Any]] = []
    for freq in ("W", "M"):
        for target_count in (4, 5, 6):
            for hold_buffer in (1.2, 1.4, 1.6):
                for min_hold in (5, 10, 15):
                    for min_score in (0.0, 55.0, 65.0):
                        cfg = copy.deepcopy(base_cfg)
                        cfg["rebalance_freq"] = freq
                        cfg["target_count"] = int(target_count)
                        cfg["hold_buffer_mult"] = float(hold_buffer)
                        cfg["turnover_mode"] = "hard_rotate"
                        cfg["turnover_budget"] = 1.0
                        cfg["min_hold_days"] = int(min_hold)
                        cfg["min_entry_score"] = float(min_score)
                        cfg["name"] = _variant_name(cfg)
                        cap = min(0.34, max(float(cfg.get("max_position_weight", 0.275) or 0.275), (1.0 / float(target_count)) + 0.02))
                        cfg["max_position_weight"] = float(cap)
                        variants.append(cfg)
    return variants


def _run_variant(
    *,
    cfg: Dict[str, Any],
    context: Dict[str, Any],
    train_start_date: str,
    train_end_date: str,
    holdout_start_date: str,
    holdout_end_date: str,
) -> Dict[str, Any]:
    scores = _build_smid_pullback_scores(context["features"], cfg)
    if scores.empty:
        return {"name": cfg["name"], "status": "no_scores"}

    active_columns = list(scores.columns)
    price_frames = _slice_prices(context["features"], ["open", "close"], active_columns)
    train_run = _run_window(
        cfg=cfg,
        prices_close=price_frames["close"],
        prices_open=price_frames["open"],
        scores=scores,
        global_data=context["global_data"],
        start_date=train_start_date,
        end_date=train_end_date,
    )
    holdout_run = _run_window(
        cfg=cfg,
        prices_close=price_frames["close"],
        prices_open=price_frames["open"],
        scores=scores,
        global_data=context["global_data"],
        start_date=holdout_start_date,
        end_date=holdout_end_date,
    )
    years = max(1e-9, (np.datetime64(holdout_end_date) - np.datetime64(holdout_start_date)).astype("timedelta64[D]").astype(int) / 365.25)
    trades = int(holdout_run.get("total_trades", 0) or 0)
    cagr_pct = float((holdout_run.get("cagr") or 0.0) * 100.0)
    dd_pct = float(abs(float(holdout_run.get("max_drawdown_pct") or 0.0)) * 100.0)
    return {
        "name": cfg["name"],
        "status": "ok",
        "rebalance_freq": cfg["rebalance_freq"],
        "target_count": int(cfg["target_count"]),
        "hold_buffer_mult": float(cfg["hold_buffer_mult"]),
        "min_hold_days": int(cfg["min_hold_days"]),
        "min_entry_score": float(cfg["min_entry_score"]),
        "max_position_weight": float(cfg["max_position_weight"]),
        "active_scored_symbols": int(len(active_columns)),
        "train_cagr_pct": float((train_run.get("cagr") or 0.0) * 100.0),
        "train_max_dd_pct": float(abs(float(train_run.get("max_drawdown_pct") or 0.0)) * 100.0),
        "holdout_cagr_pct": cagr_pct,
        "holdout_max_dd_pct": dd_pct,
        "holdout_calmar": float(cagr_pct / dd_pct) if dd_pct > 0.0 else float("nan"),
        "holdout_avg_annual_turnover_pct": float(holdout_run.get("avg_annual_turnover_pct") or float("nan")),
        "holdout_total_trades": trades,
        "holdout_trades_per_year": float(trades / years),
    }


def main() -> None:
    args = _parse_args()
    base_cfg = _load_config(args.config)
    context = build_smid_pullback_context(
        universe="RUSSELL3000",
        start_date=args.train_start_date,
        end_date=args.holdout_end_date,
        days=int(args.days),
        config_paths=[args.config],
    )
    coverage_ratio = _coverage_ratio(
        list(context.get("requested_symbols") or []),
        list(context.get("loaded_symbols") or []),
    )
    coverage_failures = _coverage_gate_failures(
        coverage_ratio=float(coverage_ratio),
        coverage_stats=dict(context.get("coverage_stats") or {}),
        min_universe_coverage=DEFAULT_MIN_UNIVERSE_COVERAGE,
        min_daily_membership_coverage=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
        min_daily_membership_coverage_p10=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
    )
    if coverage_failures:
        raise RuntimeError("Coverage gate failed: " + "; ".join(coverage_failures))

    reports = [
        _run_variant(
            cfg=cfg,
            context=context,
            train_start_date=args.train_start_date,
            train_end_date=args.train_end_date,
            holdout_start_date=args.holdout_start_date,
            holdout_end_date=args.holdout_end_date,
        )
        for cfg in _build_variants(base_cfg)
    ]
    ok_reports = [row for row in reports if row.get("status") == "ok"]
    ok_reports.sort(
        key=lambda row: (
            float(row.get("holdout_trades_per_year", float("inf"))) > 150.0,
            -(float(row.get("holdout_cagr_pct", float("-inf")))),
            float(row.get("holdout_max_dd_pct", float("inf"))),
        )
    )
    payload = {
        "base_config": str(Path(args.config).resolve()),
        "coverage_ratio": float(coverage_ratio),
        "coverage_stats": dict(context.get("coverage_stats") or {}),
        "reports": reports,
        "top_candidates": ok_reports[:10],
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
