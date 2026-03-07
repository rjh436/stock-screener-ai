#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_smid_pullback_holdout import (
    DEFAULT_CONFIGS,
    DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
    DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
    DEFAULT_MIN_UNIVERSE_COVERAGE,
    build_smid_pullback_context,
    evaluate_smid_pullback_configs_on_context,
)


DEFAULT_SPLITS: List[Dict[str, str]] = [
    {
        "label": "pre_covid_holdout_2019_2021",
        "train_start_date": "2016-01-01",
        "train_end_date": "2018-12-31",
        "holdout_start_date": "2019-01-01",
        "holdout_end_date": "2021-12-31",
    },
    {
        "label": "rolling_5y_2021_2023",
        "train_start_date": "2016-01-01",
        "train_end_date": "2020-12-31",
        "holdout_start_date": "2021-01-01",
        "holdout_end_date": "2023-12-31",
    },
    {
        "label": "rolling_5y_2022_2024",
        "train_start_date": "2017-01-01",
        "train_end_date": "2021-12-31",
        "holdout_start_date": "2022-01-01",
        "holdout_end_date": "2024-12-31",
    },
    {
        "label": "rolling_5y_2023_2025",
        "train_start_date": "2018-01-01",
        "train_end_date": "2022-12-31",
        "holdout_start_date": "2023-01-01",
        "holdout_end_date": "2025-12-31",
    },
    {
        "label": "stress_2022_only",
        "train_start_date": "2016-01-01",
        "train_end_date": "2021-12-31",
        "holdout_start_date": "2022-01-01",
        "holdout_end_date": "2022-12-31",
    },
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a broader validation matrix for SMID pullback benchmark candidates."
    )
    parser.add_argument("--configs", nargs="*", default=[str(p) for p in DEFAULT_CONFIGS])
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--days", type=int, default=4200)
    parser.add_argument("--min-universe-coverage", type=float, default=DEFAULT_MIN_UNIVERSE_COVERAGE)
    parser.add_argument("--min-daily-membership-coverage", type=float, default=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE)
    parser.add_argument(
        "--min-daily-membership-coverage-p10",
        type=float,
        default=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
    )
    parser.add_argument(
        "--stress-slippage-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to the config friction fields for the stressed scenario.",
    )
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _scenario_bounds(scenarios: Sequence[Dict[str, str]]) -> tuple[str, str]:
    starts = [str(item["train_start_date"]) for item in scenarios]
    ends = [str(item["holdout_end_date"]) for item in scenarios]
    return min(starts), max(ends)


def main() -> None:
    args = _parse_args()
    start_date, end_date = _scenario_bounds(DEFAULT_SPLITS)
    context = build_smid_pullback_context(
        universe=str(args.universe),
        start_date=start_date,
        end_date=end_date,
        days=int(args.days),
        config_paths=args.configs,
    )

    results: List[Dict[str, Any]] = []
    for split in DEFAULT_SPLITS:
        base_result = evaluate_smid_pullback_configs_on_context(
            context,
            config_paths=args.configs,
            universe=str(args.universe),
            train_start_date=str(split["train_start_date"]),
            train_end_date=str(split["train_end_date"]),
            holdout_start_date=str(split["holdout_start_date"]),
            holdout_end_date=str(split["holdout_end_date"]),
            min_universe_coverage=float(args.min_universe_coverage),
            min_daily_membership_coverage=float(args.min_daily_membership_coverage),
            min_daily_membership_coverage_p10=float(args.min_daily_membership_coverage_p10),
            raise_on_coverage_fail=False,
        )
        stress_result = evaluate_smid_pullback_configs_on_context(
            context,
            config_paths=args.configs,
            universe=str(args.universe),
            train_start_date=str(split["train_start_date"]),
            train_end_date=str(split["train_end_date"]),
            holdout_start_date=str(split["holdout_start_date"]),
            holdout_end_date=str(split["holdout_end_date"]),
            min_universe_coverage=float(args.min_universe_coverage),
            min_daily_membership_coverage=float(args.min_daily_membership_coverage),
            min_daily_membership_coverage_p10=float(args.min_daily_membership_coverage_p10),
            friction_multiplier=float(args.stress_slippage_multiplier),
            raise_on_coverage_fail=False,
        )
        results.append(
            {
                "label": str(split["label"]),
                "train_start_date": str(split["train_start_date"]),
                "train_end_date": str(split["train_end_date"]),
                "holdout_start_date": str(split["holdout_start_date"]),
                "holdout_end_date": str(split["holdout_end_date"]),
                "base": base_result,
                "stressed": stress_result,
            }
        )

    payload = {
        "universe": str(args.universe),
        "days": int(args.days),
        "min_universe_coverage": float(args.min_universe_coverage),
        "min_daily_membership_coverage": float(args.min_daily_membership_coverage),
        "min_daily_membership_coverage_p10": float(args.min_daily_membership_coverage_p10),
        "stress_slippage_multiplier": float(args.stress_slippage_multiplier),
        "splits": results,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
