#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single config and export the traded symbol set plus result metadata."
    )
    parser.add_argument("--config", required=True, help="Path to the JSON config to replay.")
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache_indicators.pkl"))
    parser.add_argument("--start-date", default="2016-03-02")
    parser.add_argument("--end-date", default="2026-03-02")
    parser.add_argument("--trading-days", type=int, default=3000)
    parser.add_argument("--start-cash", type=float, default=100000.0)
    parser.add_argument("--out-symbols", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def _normalize_drawdown_pct(value: Any) -> float:
    try:
        dd = float(value or 0.0)
    except Exception:
        dd = 0.0
    return dd * 100.0 if dd <= 1.0 else dd


def main() -> None:
    args = _parse_args()
    config_path = (ROOT / args.config).resolve()
    cache_path = (ROOT / args.cache).resolve()
    out_symbols = (ROOT / args.out_symbols).resolve()
    out_json = (ROOT / args.out_json).resolve()

    cfg = enforce_cash_only(json.loads(config_path.read_text(encoding="utf-8")))

    with cache_path.open("rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}

    strategies = load_strategies([cfg])
    if not strategies:
        raise RuntimeError(f"Failed to load strategy from {config_path}")

    result = run_backtest(
        strategies,
        prepared,
        start_cash=float(args.start_cash),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        global_data=global_data,
        universe_membership_by_day=membership,
    )
    row: Dict[str, Any] = result[0] if isinstance(result, list) else result

    trades = row.get("trades", row.get("trades_list", [])) or []
    entries = row.get("entries", row.get("entries_list", [])) or []
    symbols = sorted(
        {
            str(item.get("Symbol", "")).upper()
            for item in [*trades, *entries]
            if str(item.get("Symbol", "")).strip()
        }
    )

    entry_gate_summary: Dict[str, Dict[str, float | int]] = {}
    for trade in trades:
        gate = str(trade.get("EntryGateMode", "unknown") or "unknown")
        bucket = entry_gate_summary.setdefault(gate, {"trades": 0, "avg_return_pct": 0.0})
        bucket["trades"] = int(bucket["trades"]) + 1
        bucket["avg_return_pct"] = float(bucket["avg_return_pct"]) + float(trade.get("Return %", 0.0) or 0.0)
    for bucket in entry_gate_summary.values():
        trades_count = int(bucket["trades"] or 0)
        bucket["avg_return_pct"] = (float(bucket["avg_return_pct"]) / trades_count) if trades_count > 0 else 0.0

    entry_gate_entries: Dict[str, int] = {}
    for entry in entries:
        gate = str(entry.get("EntryGateMode", "unknown") or "unknown")
        entry_gate_entries[gate] = int(entry_gate_entries.get(gate, 0) or 0) + 1

    out_symbols.parent.mkdir(parents=True, exist_ok=True)
    out_symbols.write_text("\n".join(symbols) + "\n", encoding="utf-8")

    payload = {
        "config_path": str(config_path),
        "membership_source": membership_source,
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "constraints": {
            "allow_margin": bool(cfg.get("allow_margin", False)),
            "max_total_exposure_pct_bull": float(cfg.get("max_total_exposure_pct_bull", 1.0) or 1.0),
            "max_total_exposure_pct_bear": float(cfg.get("max_total_exposure_pct_bear", 1.0) or 1.0),
        },
        "metrics": {
            "cagr_pct": float(row.get("cagr", 0.0) or 0.0) * 100.0,
            "max_dd_pct": _normalize_drawdown_pct(row.get("max_drawdown_pct", 0.0)),
            "total_trades": int(row.get("total_trades", 0) or 0),
            "final_value": float(row.get("final_value", 0.0) or 0.0),
        },
        "traded_symbol_count": len(symbols),
        "traded_symbols_path": str(out_symbols),
        "entry_gate_summary": entry_gate_summary,
        "entry_gate_entries": entry_gate_entries,
        "trades": trades,
        "entries": entries,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))
    print(f"traded_symbol_count={len(symbols)}")
    print(f"wrote_symbols={out_symbols}")
    print(f"wrote_json={out_json}")


if __name__ == "__main__":
    main()
