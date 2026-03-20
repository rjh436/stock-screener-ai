#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day
from execution.engine import run_backtest
from optimization.walkforward import enforce_cash_only
from strategies.strategy_loader import load_strategies
import optimize_superperformance as opt


def _dd_pct(value: Any) -> float:
    try:
        dd = float(value or 0.0)
    except Exception:
        return 0.0
    if dd <= 1.0:
        dd *= 100.0
    return dd


def _run_period(
    cfg: Dict[str, Any],
    prepared: Any,
    membership: list[set],
    global_data: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    strategies = load_strategies([cfg])
    if not strategies:
        raise RuntimeError("Failed to instantiate strategy from config.")
    result = run_backtest(
        strategies,
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership,
    )
    return result[0] if isinstance(result, list) else result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT Russell 3000 backtest for a strategy config.")
    parser.add_argument("--config", required=True, help="Path to strategy config JSON.")
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache_indicators.pkl"), help="Prepared data cache path.")
    parser.add_argument("--start-5y", default="2021-02-26")
    parser.add_argument("--start-10y", default="2016-02-26")
    parser.add_argument("--end-date", default="2026-02-26")
    parser.add_argument("--trading-days", type=int, default=3000)
    parser.add_argument("--no-compress", action="store_true")
    parser.add_argument("--out", default="", help="Optional output JSON file.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg = enforce_cash_only(dict(cfg or {}))

    cache_path = Path(args.cache).resolve()
    with open(cache_path, "rb") as f:
        prepared = pickle.load(f)
    if not args.no_compress:
        prepared = opt.compress_data(prepared)

    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}

    print(
        f"run config={config_path.name} prepared={len(prepared.enriched)} "
        f"dates={len(all_dates)} membership={membership_source}"
    )

    out: Dict[str, Any] = {
        "config_path": str(config_path),
        "strategy_name": str(cfg.get("name", "") or ""),
        "membership_source": membership_source,
    }

    print("running 5Y...")
    res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
    gc.collect()
    print("running 10Y...")
    res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)

    def _pack_metrics(obj: Dict[str, Any]) -> Dict[str, Any]:
        audit = obj.get("audit_report") or {}
        return {
            "cagr_pct": float(obj.get("cagr", 0.0) or 0.0) * 100.0,
            "max_dd_pct": _dd_pct(obj.get("max_drawdown_pct", 0.0)),
            "total_trades": int(obj.get("total_trades", 0) or 0),
            "final_value": float(obj.get("final_value", 0.0) or 0.0),
            "max_gross_exposure_pct": float(audit.get("max_gross_exposure_pct", 0.0) or 0.0),
            "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        }

    out["metrics_5y"] = _pack_metrics(res5)
    out["metrics_10y"] = _pack_metrics(res10)

    print(json.dumps(out, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
