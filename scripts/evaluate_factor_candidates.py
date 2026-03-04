#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data
from scripts.run_factor_walkforward import (
    FrictionScenario,
    _build_market_risk_scalar,
    _build_momentum_quality_scores,
    _build_test_windows,
    _extract_feature_arrays,
    _pct_dd,
    _run_factor_window,
    _safe_float,
    _stitch_test_windows,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return cfg


def _days_for_range(start_date: str, end_date: str, warmup_days: int = 420) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return max(1, int((end_ts - start_ts).days)) + int(max(0, warmup_days))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple factor configs with one PIT data load")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--frictions", default="20,35")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--output", default="")
    parser.add_argument("--min-universe-coverage", type=float, default=0.55)
    args = parser.parse_args()

    config_paths = [Path(p).expanduser().resolve() for p in args.configs]
    configs: List[Dict[str, Any]] = []
    for path in config_paths:
        cfg = _load_config(path)
        cfg["_path"] = str(path)
        cfg["_label"] = str(cfg.get("name") or path.stem)
        configs.append(cfg)

    start_date = str(args.start)
    end_date = str(args.end)
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError(f"PIT Russell 3000 universe is required, got source={source}")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})

    days = _days_for_range(start_date, end_date)
    print(f"Loading Russell 3000 PIT union: {len(symbols)} symbols ({start_date} -> {end_date}, days={days})")
    data = fetch_data_pack(symbols, days=days, backtest_mode=bool(args.cache_only)) or {}
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    loaded_symbols = sorted(list(data.keys()))
    coverage = (len(loaded_symbols) / float(len(symbols))) if symbols else 0.0
    print(f"Loaded symbols with price history: {len(loaded_symbols)} / {len(symbols)} ({coverage:.1%})")
    if coverage < float(args.min_universe_coverage):
        raise RuntimeError(
            f"Universe coverage too low ({coverage:.1%} < {float(args.min_universe_coverage):.1%}). "
            "Refusing to run reduced-universe evaluation."
        )

    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership_by_day or len(membership_by_day) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    features = _extract_feature_arrays(prepared, membership_by_day)
    prices = pd.DataFrame(features["close"], index=features["dates"], columns=features["symbols"])
    eps_yoy_df = pd.DataFrame(features["eps_yoy"], index=features["dates"], columns=features["symbols"])
    sales_yoy_df = pd.DataFrame(features["sales_yoy"], index=features["dates"], columns=features["symbols"])

    friction_vals: List[float] = []
    for part in str(args.frictions or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        val = _safe_float(txt, float("nan"))
        if np.isfinite(val) and val >= 0:
            friction_vals.append(float(val))
    if not friction_vals:
        friction_vals = [20.0, 35.0]

    frictions = [
        FrictionScenario(
            name=f"{int(v)}x{int(v)}",
            transaction_cost_bps=float(args.transaction_cost_bps),
            entry_slippage_bps=float(v),
            exit_slippage_bps=float(v),
        )
        for v in friction_vals
    ]
    windows_60_12 = _build_test_windows(start_date, end_date, train_months=60, test_months=12)

    rows: List[Dict[str, Any]] = []
    for cfg in configs:
        mom_scores = _build_momentum_quality_scores(features, cfg)
        risk_scalar = _build_market_risk_scalar(global_data, prices.index, cfg)
        cfg_row: Dict[str, Any] = {"name": cfg["_label"], "config_path": cfg["_path"], "scenarios": {}}
        print(f"Evaluating: {cfg['_label']}")

        for fr in frictions:
            full = _run_factor_window(
                cfg=cfg,
                prices=prices,
                momentum_scores=mom_scores,
                value_scores=None,
                eps_yoy_df=eps_yoy_df,
                sales_yoy_df=sales_yoy_df,
                risk_scalar_by_date=risk_scalar,
                start_date=start_date,
                end_date=end_date,
                friction=fr,
            )
            stitched = _stitch_test_windows(
                cfg=cfg,
                prices=prices,
                momentum_scores=mom_scores,
                value_scores=None,
                eps_yoy_df=eps_yoy_df,
                sales_yoy_df=sales_yoy_df,
                risk_scalar_by_date=risk_scalar,
                windows=windows_60_12,
                test_months=12,
                friction=fr,
            )
            full_cagr = _safe_float(full.get("cagr"), float("nan"))
            if np.isfinite(full_cagr):
                full_cagr *= 100.0
            dd = _pct_dd(full.get("max_drawdown_pct", 0.0))
            turnover = _safe_float(full.get("avg_annual_turnover_pct"), float("nan"))
            oos60 = _safe_float(stitched.get("stitched_cagr_pct"), float("nan"))

            cfg_row["scenarios"][fr.name] = {
                "full_cagr_pct": full_cagr,
                "full_max_dd_pct": dd,
                "full_trades": int(full.get("total_trades", 0) or 0),
                "avg_annual_turnover_pct": turnover,
                "oos60_cagr_pct": oos60,
            }

        s20 = cfg_row["scenarios"].get("20x20", {})
        s35 = cfg_row["scenarios"].get("35x35", {})
        gates = {
            "oos60_20_ge_10": bool(np.isfinite(s20.get("oos60_cagr_pct", float("nan"))) and s20.get("oos60_cagr_pct", 0.0) >= 10.0),
            "oos60_35_ge_8": bool(np.isfinite(s35.get("oos60_cagr_pct", float("nan"))) and s35.get("oos60_cagr_pct", 0.0) >= 8.0),
            "dd20_le_28": bool(np.isfinite(s20.get("full_max_dd_pct", float("nan"))) and s20.get("full_max_dd_pct", 999.0) <= 28.0),
            "turnover20_le_200": bool(np.isfinite(s20.get("avg_annual_turnover_pct", float("nan"))) and s20.get("avg_annual_turnover_pct", 999.0) <= 200.0),
        }
        gate_count = sum(1 for v in gates.values() if v)
        score = (
            gate_count * 1000.0
            + (s20.get("oos60_cagr_pct") or -999.0) * 10.0
            + (s35.get("oos60_cagr_pct") or -999.0) * 8.0
            - max(0.0, (s20.get("full_max_dd_pct") or 999.0) - 28.0) * 3.0
            - max(0.0, (s20.get("avg_annual_turnover_pct") or 999.0) - 200.0) * 0.5
        )
        cfg_row["gates"] = gates
        cfg_row["gate_count"] = gate_count
        cfg_row["score"] = score
        rows.append(cfg_row)

    rows.sort(key=lambda x: float(x.get("score", -999999.0)), reverse=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "universe": {
            "name": "RUSSELL3000",
            "source": source,
            "membership_source": membership_source,
            "requested_symbol_count": len(symbols),
            "loaded_symbol_count": len(loaded_symbols),
            "coverage_ratio": coverage,
            "cache_only": bool(args.cache_only),
        },
        "results": rows,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (
        ROOT / "tmp" / f"factor_candidate_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {output_path}")
    if rows:
        print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
