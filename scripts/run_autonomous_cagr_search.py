#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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

from scripts.evaluate_low_turnover_alpha_frontier import _pack_metrics, _score


DEFAULT_BASE_CONFIGS = [
    "config/superperformance_alpha_b4.json",
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b10.json",
    "config/superperformance_alpha_b20.json",
    "config/qullamaggie_breakout_practical_v1.json",
]

DEFAULT_FRONTIER_LOGS = [
    "logs/superperformance_frontier_20260228_233006.jsonl",
]


def _resolve_args(values: list[str] | None, defaults: list[str]) -> list[str]:
    out = [str(v) for v in (values or []) if str(v).strip()]
    return out if out else list(defaults)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous frontier search for realistic high-CAGR moderate-turnover strategies."
    )
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--frontier-log", action="append", default=[])
    parser.add_argument("--frontier-limit", type=int, default=12)
    parser.add_argument("--cache", default=str(ROOT / "data" / "cache_indicators.pkl"))
    parser.add_argument("--start-5y", default="2021-03-02")
    parser.add_argument("--start-10y", default="2016-03-02")
    parser.add_argument("--end-date", default="2026-03-02")
    parser.add_argument("--trading-days", type=int, default=3000)
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--entry-slippage-bps", type=float, default=10.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=10.0)
    parser.add_argument("--soft-trades-per-year", type=float, default=150.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--progress", default="tmp/autonomous_cagr_search_progress.jsonl")
    parser.add_argument("--out", default="tmp/autonomous_cagr_search_latest.json")
    parser.add_argument("--skip-10y", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dedupe_configs(configs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for cfg in configs:
        raw = json.dumps(cfg, sort_keys=True)
        if raw in seen:
            continue
        seen.add(raw)
        out.append(cfg)
    return out


def _load_frontier_candidates(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        cfg = item.get("config")
        if isinstance(cfg, dict):
            rows.append(
                {
                    "score": float(item.get("score", 0.0) or 0.0),
                    "config": cfg,
                }
            )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return [dict(row["config"]) for row in rows[:limit]]


def _inject_fast_pyr_candidate() -> Dict[str, Any]:
    cfgs = _load_json(ROOT / "config" / "generated_strategies.json")
    base = next(item for item in cfgs if isinstance(item, dict) and item.get("name") == "Superperformance Alpha B4")
    out = copy.deepcopy(base)
    out.update(
        {
            "name": "Superperformance Alpha B4 FAST_PYR_EP_ON",
            "pyramid_threshold": 0.05,
            "pyramid_fraction": 1.0,
            "pyramid_max_adds": 2,
            "pyramid_ep_enabled": True,
            "min_entry_score": 20,
        }
    )
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


def _candidate_pool(args: argparse.Namespace) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for rel in _resolve_args(args.config, DEFAULT_BASE_CONFIGS):
        configs.append(_load_json((ROOT / rel).resolve()))
    configs.append(_inject_fast_pyr_candidate())
    for rel in _resolve_args(args.frontier_log, DEFAULT_FRONTIER_LOGS):
        path = (ROOT / rel).resolve()
        if path.exists():
            configs.extend(_load_frontier_candidates(path, args.frontier_limit))
    return _dedupe_configs(configs)[: args.max_candidates]


def _fetch_global_data(days: int) -> Dict[str, Any]:
    raw = fetch_data_pack(["SPY", "$VIX", "VIX"], days=int(days), backtest_mode=True) or {}
    return {
        "SPY": raw.get("SPY"),
        "VIX": raw.get("$VIX") if raw.get("$VIX") is not None else raw.get("VIX"),
    }


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


def _write_progress(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = _parse_args()
    progress_path = (ROOT / args.progress).resolve()
    out_path = (ROOT / args.out).resolve()

    with open((ROOT / args.cache).resolve(), "rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = _fetch_global_data(args.trading_days)

    rows: List[Dict[str, Any]] = []
    for idx, raw_cfg in enumerate(_candidate_pool(args), start=1):
        cfg = _apply_realistic_costs(raw_cfg, args)
        name = str(cfg.get("name", f"Candidate {idx}"))
        print(f"[{idx}] {name}", flush=True)
        row: Dict[str, Any] = {
            "name": name,
            "config": cfg,
        }
        res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
        row["metrics_5y"] = _pack_metrics(res5, args.start_5y, args.end_date)
        if not args.skip_10y:
            res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)
            row["metrics_10y"] = _pack_metrics(res10, args.start_10y, args.end_date)
        else:
            row["metrics_10y"] = {}
        row["score"] = _score(row, args.soft_trades_per_year)
        rows.append(row)
        _write_progress(progress_path, row)
        ranked = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)
        payload = {
            "membership_source": membership_source,
            "soft_trades_per_year": float(args.soft_trades_per_year),
            "realistic_costs": {
                "transaction_cost_bps": args.transaction_cost_bps,
                "entry_slippage_bps": args.entry_slippage_bps,
                "exit_slippage_bps": args.exit_slippage_bps,
            },
            "rows": ranked,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(out_path)


if __name__ == "__main__":
    main()
