#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd


DEFAULT_GRID = [
    {"label": "H0", "etf": 0.25, "smid": 0.75, "b17": 0.00, "purpose": "Current benchmark (control)"},
    {"label": "H1", "etf": 0.00, "smid": 0.80, "b17": 0.20, "purpose": "Pure 2-strategy, light B17"},
    {"label": "H2", "etf": 0.00, "smid": 0.70, "b17": 0.30, "purpose": "Pure 2-strategy, moderate B17"},
    {"label": "H3", "etf": 0.10, "smid": 0.70, "b17": 0.20, "purpose": "Light ETF anchor + B17"},
    {"label": "H4", "etf": 0.10, "smid": 0.60, "b17": 0.30, "purpose": "Light ETF + moderate B17"},
    {"label": "H5", "etf": 0.20, "smid": 0.60, "b17": 0.20, "purpose": "Moderate ETF + light B17"},
    {"label": "H6", "etf": 0.20, "smid": 0.55, "b17": 0.25, "purpose": "Balanced three-way"},
    {"label": "H7", "etf": 0.15, "smid": 0.50, "b17": 0.35, "purpose": "Max B17 stress test"},
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fixed arithmetic allocation grid over ETF/SMID/B17 monthly equity curves.")
    parser.add_argument("--etf-csv", required=True, help="CSV containing month_end and etf_equity columns.")
    parser.add_argument("--pair-csv", required=True, help="CSV containing month_end plus b17_equity and smid_equity columns.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--out-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--start-cash", type=float, default=100000.0)
    parser.add_argument("--control-label", default="H0")
    parser.add_argument("--control-target-cagr", type=float, default=33.71)
    parser.add_argument("--control-target-dd", type=float, default=14.92)
    parser.add_argument("--control-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--min-promote-cagr", type=float, default=25.0)
    return parser.parse_args()


def _load_equity_csv(path: Path, *, expected_columns: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "month_end" not in df.columns:
        raise ValueError(f"{path} is missing required column: month_end")
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = df.loc[:, ["month_end", *expected_columns]].copy()
    out["month_end"] = pd.to_datetime(out["month_end"], errors="coerce")
    out = out.dropna(subset=["month_end"]).sort_values("month_end")
    for col in expected_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _equity_to_monthly_returns(equity: pd.Series, *, start_cash: float) -> pd.Series:
    ser = pd.to_numeric(equity, errors="coerce").dropna().astype("float64")
    if ser.empty:
        return pd.Series(dtype="float64")
    prev = ser.shift(1)
    returns = (ser / prev) - 1.0
    returns.iloc[0] = (float(ser.iloc[0]) / float(start_cash)) - 1.0
    return returns.astype("float64")


def _metrics_from_monthly_returns(monthly_returns: pd.Series, *, start_cash: float) -> Dict[str, float]:
    ser = monthly_returns.dropna().astype("float64")
    if ser.empty:
        return {
            "cagr_pct": float("nan"),
            "max_dd_pct": float("nan"),
            "calmar": float("nan"),
            "final_value": float("nan"),
            "months": 0,
        }
    equity = (1.0 + ser).cumprod() * float(start_cash)
    years = float(len(ser)) / 12.0
    cagr = (float(equity.iloc[-1]) / float(start_cash)) ** (1.0 / years) - 1.0 if years > 0.0 else float("nan")
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd_pct = abs(float(drawdown.min()) * 100.0)
    cagr_pct = float(cagr * 100.0)
    calmar = float(cagr_pct / max_dd_pct) if max_dd_pct > 0.0 and not math.isnan(cagr_pct) else float("nan")
    return {
        "cagr_pct": cagr_pct,
        "max_dd_pct": max_dd_pct,
        "calmar": calmar,
        "final_value": float(equity.iloc[-1]),
        "months": int(len(ser)),
        "start_month": str(ser.index[0].strftime("%Y-%m")),
        "end_month": str(ser.index[-1].strftime("%Y-%m")),
    }


def _control_validation(row: Mapping[str, Any], *, target_cagr: float, target_dd: float, tol: float) -> Dict[str, Any]:
    cagr = float(row.get("cagr_pct", float("nan")))
    dd = float(row.get("max_dd_pct", float("nan")))
    cagr_diff = cagr - float(target_cagr)
    dd_diff = dd - float(target_dd)
    return {
        "target_cagr_pct": float(target_cagr),
        "target_max_dd_pct": float(target_dd),
        "tolerance_pp": float(tol),
        "cagr_diff_pp": cagr_diff,
        "max_dd_diff_pp": dd_diff,
        "pass": abs(cagr_diff) <= float(tol) and abs(dd_diff) <= float(tol),
    }


def _rows_to_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "etf_weight",
        "smid_weight",
        "b17_weight",
        "purpose",
        "cagr_pct",
        "max_dd_pct",
        "calmar",
        "final_value",
        "months",
        "beats_control_calmar",
        "qualifies_min_cagr",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = _parse_args()
    etf_df = _load_equity_csv(Path(args.etf_csv).expanduser().resolve(), expected_columns=["etf_equity"])
    pair_df = _load_equity_csv(Path(args.pair_csv).expanduser().resolve(), expected_columns=["b17_equity", "smid_equity"])
    merged = etf_df.merge(pair_df, on="month_end", how="inner").sort_values("month_end")
    if merged.empty:
        raise ValueError("No overlapping month_end rows found across the supplied CSVs.")

    merged = merged.set_index("month_end")
    monthly_returns = {
        "etf": _equity_to_monthly_returns(merged["etf_equity"], start_cash=float(args.start_cash)),
        "smid": _equity_to_monthly_returns(merged["smid_equity"], start_cash=float(args.start_cash)),
        "b17": _equity_to_monthly_returns(merged["b17_equity"], start_cash=float(args.start_cash)),
    }

    rows: List[Dict[str, Any]] = []
    for spec in DEFAULT_GRID:
        weights = {
            "etf": float(spec["etf"]),
            "smid": float(spec["smid"]),
            "b17": float(spec["b17"]),
        }
        total = sum(weights.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{spec['label']} weights do not sum to 1.0: {weights}")
        blended = sum(monthly_returns[name] * weight for name, weight in weights.items())
        metrics = _metrics_from_monthly_returns(blended, start_cash=float(args.start_cash))
        rows.append(
            {
                "label": str(spec["label"]),
                "etf_weight": weights["etf"],
                "smid_weight": weights["smid"],
                "b17_weight": weights["b17"],
                "purpose": str(spec["purpose"]),
                **metrics,
            }
        )

    control_row = next((row for row in rows if str(row["label"]) == str(args.control_label)), None)
    if control_row is None:
        raise ValueError(f"Control label {args.control_label} was not found in the grid.")
    control_validation = _control_validation(
        control_row,
        target_cagr=float(args.control_target_cagr),
        target_dd=float(args.control_target_dd),
        tol=float(args.control_tolerance_pp),
    )
    control_calmar = float(control_row.get("calmar", float("nan")))
    for row in rows:
        row["beats_control_calmar"] = bool(float(row.get("calmar", float("-inf"))) > control_calmar)
        row["qualifies_min_cagr"] = bool(float(row.get("cagr_pct", float("-inf"))) >= float(args.min_promote_cagr))

    winners = [
        row
        for row in rows
        if bool(row["beats_control_calmar"]) and bool(row["qualifies_min_cagr"])
    ]
    winners.sort(key=lambda row: float(row.get("calmar", float("-inf"))), reverse=True)

    payload = {
        "window": {
            "start_month": str(merged.index.min().strftime("%Y-%m")),
            "end_month": str(merged.index.max().strftime("%Y-%m")),
            "months": int(len(merged.index)),
        },
        "inputs": {
            "etf_csv": str(Path(args.etf_csv).expanduser().resolve()),
            "pair_csv": str(Path(args.pair_csv).expanduser().resolve()),
            "start_cash": float(args.start_cash),
        },
        "control_validation": control_validation,
        "rows": rows,
        "best_row_by_calmar": winners[0] if winners else None,
        "promotion_rule": {
            "min_promote_cagr_pct": float(args.min_promote_cagr),
            "must_beat_control_calmar": True,
        },
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
