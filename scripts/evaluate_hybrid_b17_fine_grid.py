#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_hybrid_b17_walkforward import (
    DEFAULT_B17_CONFIG,
    DEFAULT_ETF_CONFIG,
    DEFAULT_PREPARED_CACHE,
    DEFAULT_SMID_CONFIG,
    _blend_equity_curves,
    _build_b17_runner,
    _build_etf_runner,
    _build_smid_runner,
    _equity_curve_to_series,
    _metrics_from_equity,
    _resolve_end_date,
)


DEFAULT_FINE_GRID = [
    {"label": "FG1", "etf": 0.05, "smid": 0.75, "b17": 0.20},
    {"label": "FG2", "etf": 0.05, "smid": 0.70, "b17": 0.25},
    {"label": "FG3", "etf": 0.05, "smid": 0.65, "b17": 0.30},
    {"label": "FG4", "etf": 0.10, "smid": 0.75, "b17": 0.15},
    {"label": "FG5", "etf": 0.10, "smid": 0.65, "b17": 0.25},
    {"label": "FG6", "etf": 0.10, "smid": 0.72, "b17": 0.18},
    {"label": "FG7", "etf": 0.15, "smid": 0.70, "b17": 0.15},
    {"label": "FG8", "etf": 0.15, "smid": 0.65, "b17": 0.20},
    {"label": "FG9", "etf": 0.07, "smid": 0.73, "b17": 0.20}
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine full-window ETF/SMID/B17 allocation grid.")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--etf-config", default=str(DEFAULT_ETF_CONFIG))
    parser.add_argument("--smid-config", default=str(DEFAULT_SMID_CONFIG))
    parser.add_argument("--b17-config", default=str(DEFAULT_B17_CONFIG))
    parser.add_argument("--prepared-cache", default=str(DEFAULT_PREPARED_CACHE))
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--out-csv", default="")
    return parser.parse_args()


def _rows_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    start_date = pd.Timestamp(args.start_date).date().isoformat()
    end_date = _resolve_end_date(args.end_date, args.prepared_cache)

    etf_runner, etf_meta = _build_etf_runner(Path(args.etf_config).expanduser().resolve(), start_date=start_date, end_date=end_date)
    smid_runner, smid_meta = _build_smid_runner(Path(args.smid_config).expanduser().resolve(), start_date=start_date, end_date=end_date)
    b17_runner, b17_meta, _ = _build_b17_runner(
        Path(args.b17_config).expanduser().resolve(),
        start_date=start_date,
        end_date=end_date,
        prepared_cache_path=Path(args.prepared_cache).expanduser().resolve(),
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
    )

    full_runs = {
        "etf": etf_runner(start_date, end_date),
        "smid": smid_runner(start_date, end_date),
        "b17": b17_runner(start_date, end_date),
    }
    full_curves = {name: _equity_curve_to_series(run) for name, run in full_runs.items()}

    rows: List[Dict[str, Any]] = []
    for spec in DEFAULT_FINE_GRID:
        weights = {"etf": float(spec["etf"]), "smid": float(spec["smid"]), "b17": float(spec["b17"])}
        equity = _blend_equity_curves(full_curves, weights)
        metrics = _metrics_from_equity(equity)
        dd = float(metrics["max_dd_pct"])
        cagr = float(metrics["cagr_pct"])
        calmar = float(cagr / dd) if dd > 0 else float("nan")
        rows.append(
            {
                "label": str(spec["label"]),
                "etf_pct": int(round(weights["etf"] * 100.0)),
                "smid_pct": int(round(weights["smid"] * 100.0)),
                "b17_pct": int(round(weights["b17"] * 100.0)),
                "full_cagr_pct": cagr,
                "full_max_dd_pct": dd,
                "full_calmar": calmar,
                "beats_29pct_cagr": bool(cagr > 29.0),
                "meets_29pct_gate": bool(cagr > 29.0 and dd <= 14.0 and calmar >= 2.0),
            }
        )

    rows.sort(key=lambda row: float(row["full_cagr_pct"]), reverse=True)
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "component_sources": {
            "etf": etf_meta,
            "smid": smid_meta,
            "b17": b17_meta,
        },
        "promotion_probe_gate": {
            "min_full_cagr_pct": 29.0,
            "max_full_dd_pct": 14.0,
            "min_full_calmar": 2.0,
        },
        "rows": rows,
        "best_row_by_full_cagr": rows[0] if rows else None,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        _rows_to_csv(Path(args.out_csv).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
