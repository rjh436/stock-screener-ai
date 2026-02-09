#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from optimize_weights import build_superperformance_config
from run_phase3_validation import _run_known_winner_suite

RESULTS_FILE = os.path.join(ROOT, "superperformance_results.csv")
WINNER_FILE = os.path.join(ROOT, "config", "superperformance_winner.json")
CHAMPION_FILE = os.path.join(ROOT, "config", "superperformance_champion.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and promote best hybrid genome using known-winner diagnostics."
    )
    parser.add_argument("--lookback", type=int, default=20, help="Rows to inspect from results tail.")
    parser.add_argument("--max-candidates", type=int, default=8, help="Max unique genomes to evaluate.")
    parser.add_argument("--tech-weight", type=float, default=0.60)
    parser.add_argument("--fund-weight", type=float, default=0.40)
    return parser.parse_args()


def _read_recent_candidates(lookback: int, max_candidates: int) -> List[Dict[str, Any]]:
    if not os.path.exists(RESULTS_FILE):
        return []

    rows: List[Dict[str, Any]] = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                metrics = {
                    "cagr": float(row[1]),
                    "dd": float(row[2]),
                    "calmar": float(row[3]),
                    "trades": int(float(row[4])),
                }
                genome = ast.literal_eval(row[5])
                if not isinstance(genome, dict):
                    continue
            except Exception:
                continue
            rows.append({"genome": genome, "metrics": metrics})

    if not rows:
        return []

    recent = rows[-max(int(lookback), 1) :]
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for item in reversed(recent):
        key = json.dumps(item["genome"], sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= max(int(max_candidates), 1):
            break
    return list(reversed(unique))


def _evaluate_candidate(
    label: str,
    genome: Dict[str, Any],
    metrics: Dict[str, Any],
    tech_weight: float,
    fund_weight: float,
) -> Dict[str, Any]:
    results = _run_known_winner_suite(
        technical_weight=float(tech_weight),
        fundamental_weight=float(fund_weight),
        strategy_override=genome,
    )
    entries = sum(1 for r in results if bool(r.get("entry_detected")))
    pfs: List[float] = []
    pf_pass = 0
    for r in results:
        if not bool(r.get("entry_detected")):
            continue
        pf = float(r.get("profit_factor", 0.0) or 0.0)
        if not math.isfinite(pf):
            pf = 99.0
        pfs.append(pf)
        if pf > 2.0:
            pf_pass += 1
    min_pf = min(pfs) if pfs else 0.0
    mean_pf = (sum(pfs) / len(pfs)) if pfs else 0.0

    # Rank for "hybrid winners first": entries -> quality -> backtest fitness.
    rank_key = (
        entries,
        pf_pass,
        min(min_pf, 99.0),
        float(metrics.get("calmar", 0.0) or 0.0),
        float(metrics.get("cagr", 0.0) or 0.0),
        -float(metrics.get("dd", 999.0) or 999.0),
    )
    return {
        "label": label,
        "genome": genome,
        "metrics": metrics,
        "entries": entries,
        "pf_pass": pf_pass,
        "min_pf": min_pf,
        "mean_pf": mean_pf,
        "rank_key": rank_key,
        "results": results,
    }


def _write_promoted_winner(best: Dict[str, Any], tech_weight: float, fund_weight: float) -> None:
    genome = dict(best["genome"])
    genome["technical_weight"] = float(tech_weight)
    genome["fundamental_weight"] = float(fund_weight)
    with open(WINNER_FILE, "w") as f:
        json.dump(genome, f, indent=4)

    raw_metrics = {
        "cagr": float(best["metrics"].get("cagr", 0.0) or 0.0),
        "dd": float(best["metrics"].get("dd", 0.0) or 0.0),
        "calmar": float(best["metrics"].get("calmar", 0.0) or 0.0),
        "trades": int(best["metrics"].get("trades", 0) or 0),
    }
    metrics_unavailable = raw_metrics["dd"] >= 900.0 or raw_metrics["cagr"] < 0.0
    if metrics_unavailable:
        raw_metrics = {"cagr": 0.0, "dd": 0.0, "calmar": 0.0, "trades": 0}

    champion = {
        "genome": genome,
        "metrics": raw_metrics,
        "known_winners": {
            "entries": int(best["entries"]),
            "pf_pass": int(best["pf_pass"]),
            "min_pf": float(best["min_pf"]),
            "mean_pf": float(best["mean_pf"]),
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": f"select_hybrid_winner:{best['label']}",
    }
    if metrics_unavailable:
        champion["metrics_note"] = "Backtest metrics unavailable for selected candidate source."
    with open(CHAMPION_FILE, "w") as f:
        json.dump(champion, f, indent=4)


def main() -> None:
    args = _parse_args()
    candidates = _read_recent_candidates(args.lookback, args.max_candidates)
    if not candidates:
        print("No recent candidates found in results CSV.")
        sys.exit(1)

    # Include baseline strategy config as fallback candidate.
    baseline_cfg = build_superperformance_config(args.tech_weight, args.fund_weight)
    candidates.append(
        {
            "genome": baseline_cfg,
            "metrics": {"cagr": -1.0, "dd": 999.0, "calmar": -1.0, "trades": 0},
        }
    )

    evaluated: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates, start=1):
        label = f"candidate_{idx}"
        print(f"\n=== Evaluating {label} ===")
        evaluated.append(
            _evaluate_candidate(
                label,
                cand["genome"],
                cand["metrics"],
                args.tech_weight,
                args.fund_weight,
            )
        )

    evaluated.sort(key=lambda x: x["rank_key"], reverse=True)
    best = evaluated[0]
    _write_promoted_winner(best, args.tech_weight, args.fund_weight)

    print("\n=== PROMOTED HYBRID WINNER ===")
    print(
        f"{best['label']} | entries={best['entries']}/3 | "
        f"pf_pass={best['pf_pass']} | min_pf={best['min_pf']:.3f} | mean_pf={best['mean_pf']:.3f}"
    )
    print(
        f"Backtest metrics | CAGR={float(best['metrics'].get('cagr', 0.0) or 0.0):.2f}% | "
        f"DD={float(best['metrics'].get('dd', 0.0) or 0.0):.2f}% | "
        f"Calmar={float(best['metrics'].get('calmar', 0.0) or 0.0):.3f}"
    )
    print(f"Wrote winner: {WINNER_FILE}")
    print(f"Wrote champion: {CHAMPION_FILE}")


if __name__ == "__main__":
    main()
