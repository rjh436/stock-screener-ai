#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.custom_universe import list_cached_symbols, load_symbol_file
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from strategies.strategy_loader import load_strategies


def _load_strategy_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.config:
        path = Path(str(args.config)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Expected single strategy config object in {path}")
        return dict(cfg)

    generated_path = ROOT / "config" / "generated_strategies.json"
    with generated_path.open("r", encoding="utf-8") as f:
        configs = json.load(f)
    cfg = next((item for item in configs if str(item.get("name", "")) == str(args.strategy_name)), None)
    if not isinstance(cfg, dict):
        raise ValueError(f"Strategy '{args.strategy_name}' not found in {generated_path}")
    return dict(cfg)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    if str(args.universe).lower() == "cache_all":
        return list_cached_symbols(max_symbols=args.max_symbols)
    raise ValueError(f"Unsupported universe preset: {args.universe}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a backtest on a custom cached-symbol universe.")
    parser.add_argument("--config", default="", help="Path to a single-strategy JSON config.")
    parser.add_argument("--strategy-name", default="Superperformance Alpha B4", help="Name inside config/generated_strategies.json.")
    parser.add_argument("--universe", default="cache_all", help="Universe preset. Currently supported: cache_all.")
    parser.add_argument("--symbols", default="", help="Comma-separated explicit symbol list.")
    parser.add_argument("--symbol-file", default="", help="Optional file containing symbols.")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3000)
    parser.add_argument("--start-cash", type=float, default=100000.0)
    parser.add_argument("--max-symbols", type=int, default=0, help="Optional cap on symbol count for debugging.")
    parser.add_argument("--global-symbols", default="SPY,VIX,$VIX", help="Comma-separated global symbols.")
    parser.add_argument("--out", default="", help="Optional output JSON file.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_strategy_config(args)
    cfg["allow_margin"] = bool(cfg.get("allow_margin", False))
    symbols = _resolve_symbols(args)
    if not symbols:
        raise ValueError("No symbols resolved for custom-universe run.")

    global_symbols = [str(x).strip().upper() for x in str(args.global_symbols).split(",") if str(x).strip()]

    print(f"custom-universe run: strategy={cfg.get('name','')} requested_symbols={len(symbols)}")
    data = fetch_data_pack(symbols, days=int(args.days), backtest_mode=True) or {}
    loaded_symbols = sorted(data.keys())
    print(f"loaded_symbols={len(loaded_symbols)}")
    global_data = fetch_data_pack(global_symbols, days=int(args.days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, loaded_symbols, start_date=args.start_date, global_data=global_data)
    print(f"prepared_symbols={len(prepared.enriched)} prepared_dates={len(prepared.all_dates)}")

    strategies = load_strategies([cfg])
    if not strategies:
        raise RuntimeError("Failed to instantiate strategy from config.")

    result = run_backtest(
        strategies,
        prepared,
        start_cash=float(args.start_cash),
        start_date=args.start_date,
        end_date=args.end_date,
        global_data=global_data,
    )
    if isinstance(result, list):
        result = result[0] if result else {}

    audit = result.get("audit_report") or {}
    payload = {
        "strategy_name": str(cfg.get("name", "") or ""),
        "requested_symbols": int(len(symbols)),
        "loaded_symbols": int(len(loaded_symbols)),
        "prepared_symbols": int(len(prepared.enriched)),
        "prepared_dates": int(len(prepared.all_dates)),
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "cagr_pct": float(result.get("cagr", 0.0) or 0.0) * 100.0,
        "max_dd_pct": float(result.get("max_drawdown_pct", 0.0) or 0.0) * 100.0,
        "final_value": float(result.get("final_value", 0.0) or 0.0),
        "total_trades": int(result.get("total_trades", 0) or 0),
        "hit_rate_pct": float(result.get("hit_rate", 0.0) or 0.0),
        "max_gross_exposure_pct": float(audit.get("max_gross_exposure_pct", 0.0) or 0.0),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
