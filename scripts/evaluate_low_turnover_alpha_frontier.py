#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day
from execution.engine import run_backtest
from strategies.strategy_loader import load_strategies
import optimize_superperformance as opt


DEFAULT_CONFIGS = [
    "config/superperformance_alpha_b4.json",
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b10.json",
    "config/superperformance_alpha_b20.json",
]


def _dd_pct(value: Any) -> float:
    try:
        out = float(value or 0.0)
    except Exception:
        return 0.0
    if out <= 1.0:
        out *= 100.0
    return out


def _years_between(start_date: str, end_date: str) -> float:
    start = Path(start_date).name if "/" not in start_date else start_date
    end = Path(end_date).name if "/" not in end_date else end_date
    from pandas import Timestamp

    span_days = max(1, int((Timestamp(end) - Timestamp(start)).days))
    return span_days / 365.25


def _pack_metrics(result: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
    audit = result.get("audit_report") if isinstance(result.get("audit_report"), dict) else {}
    total_trades = int(result.get("total_trades", 0) or 0)
    years = max(0.01, _years_between(start_date, end_date))
    return {
        "cagr_pct": float(result.get("cagr", 0.0) or 0.0) * 100.0,
        "max_dd_pct": _dd_pct(result.get("max_drawdown_pct", 0.0)),
        "total_trades": total_trades,
        "trades_per_year": total_trades / years,
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(audit.get("max_gross_exposure_pct", 0.0) or 0.0),
        "final_value": float(result.get("final_value", 0.0) or 0.0),
    }


def _score(row: Dict[str, Any], max_trades_per_year: float) -> float:
    m5 = row.get("metrics_5y", {})
    m10 = row.get("metrics_10y", {})
    cagr5 = float(m5.get("cagr_pct", 0.0) or 0.0)
    cagr10 = float(m10.get("cagr_pct", 0.0) or 0.0)
    dd10 = abs(float(m10.get("max_dd_pct", 0.0) or 0.0))
    trades10 = float(m10.get("trades_per_year", 0.0) or 0.0)
    same_day = int(m10.get("same_day_open_entries", 0) or 0) + int(m5.get("same_day_open_entries", 0) or 0)
    if same_day > 0:
        return -1_000_000.0 - same_day
    score = cagr10 * 1.4 + cagr5 * 0.8 - dd10 * 0.45
    if trades10 > max_trades_per_year:
        score -= (trades10 - max_trades_per_year) * 0.8
    return float(score)


def _load_frontier_rows(frontier_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with frontier_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cfg = item.get("config")
            if isinstance(cfg, dict):
                rows.append(cfg)
    return rows


def _candidate_configs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _append_cfg(cfg: Dict[str, Any]) -> None:
        raw = json.dumps(cfg, sort_keys=True)
        if raw in seen:
            return
        seen.add(raw)
        out.append(cfg)

    for rel_path in args.config:
        path = (ROOT / rel_path).resolve()
        cfg = json.loads(path.read_text(encoding="utf-8"))
        _append_cfg(cfg)

    if args.frontier_log:
        rows = _load_frontier_rows((ROOT / args.frontier_log).resolve())
        for cfg in rows[: args.frontier_limit]:
            _append_cfg(cfg)

    return out


def _apply_realistic_costs(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    out["transaction_cost_bps"] = float(args.transaction_cost_bps)
    out["entry_slippage_bps"] = float(args.entry_slippage_bps)
    out["exit_slippage_bps"] = float(args.exit_slippage_bps)
    out["slippage_bps"] = 0.0
    out["allow_margin"] = False
    out["log_regime_skips"] = False
    return out


def _run_period(
    cfg: Dict[str, Any],
    prepared: Any,
    membership: List[set],
    global_data: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    strategies = load_strategies([cfg])
    if not strategies:
        raise RuntimeError(f"Failed to instantiate strategy: {cfg.get('name')}")
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate low-turnover Alpha candidates on cache-backed PIT Russell 3000 windows."
    )
    parser.add_argument("--config", action="append", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--frontier-log", default="")
    parser.add_argument("--frontier-limit", type=int, default=5)
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache_indicators.pkl"))
    parser.add_argument("--start-5y", default="2021-03-02")
    parser.add_argument("--start-10y", default="2016-03-02")
    parser.add_argument("--end-date", default="2026-03-02")
    parser.add_argument("--trading-days", type=int, default=3000)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=10.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=10.0)
    parser.add_argument("--max-trades-per-year", type=float, default=150.0)
    parser.add_argument("--skip-10y", action="store_true")
    parser.add_argument("--skip-5y", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open((ROOT / args.cache).resolve(), "rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}

    rows: List[Dict[str, Any]] = []
    for raw_cfg in _candidate_configs(args):
        cfg = _apply_realistic_costs(raw_cfg, args)
        name = str(cfg.get("name", "Unnamed"))
        print(f"[run] {name}", flush=True)
        res5 = None
        res10 = None
        if not args.skip_5y:
            print(f"  [5y] {args.start_5y} -> {args.end_date}", flush=True)
            res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
        if not args.skip_10y:
            print(f"  [10y] {args.start_10y} -> {args.end_date}", flush=True)
            res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)
        row = {
            "name": name,
            "metrics_5y": _pack_metrics(res5, args.start_5y, args.end_date) if res5 is not None else {},
            "metrics_10y": _pack_metrics(res10, args.start_10y, args.end_date) if res10 is not None else {},
            "config": cfg,
        }
        row["score"] = _score(row, args.max_trades_per_year)
        rows.append(row)

    rows.sort(key=lambda item: item.get("score", -1_000_000.0), reverse=True)
    out = {
        "membership_source": membership_source,
        "realistic_costs": {
            "transaction_cost_bps": args.transaction_cost_bps,
            "entry_slippage_bps": args.entry_slippage_bps,
            "exit_slippage_bps": args.exit_slippage_bps,
        },
        "rows": rows,
    }
    print(json.dumps(out, indent=2))
    if args.out:
        out_path = (ROOT / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
