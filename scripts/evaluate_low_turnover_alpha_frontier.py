#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.custom_universe import load_symbol_file
from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day
from execution.engine import run_backtest
from optimization.robustness import pack_result_metrics, score_candidate_row
from optimization.walkforward import (
    DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT,
    DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT,
    DEFAULT_WALKFORWARD_START,
    enforce_cash_only,
    parse_friction_values,
    replay_ranked_candidates,
)
from strategies.strategy_loader import load_strategies
import optimize_superperformance as opt


DEFAULT_CONFIGS = [
    "config/superperformance_alpha_b4.json",
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b10.json",
    "config/superperformance_alpha_b20.json",
]


def _pack_metrics(result: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
    return pack_result_metrics(result, start_date=start_date, end_date=end_date)


def _resolve_config_args(values: list[str] | None) -> list[str]:
    out = [str(v) for v in (values or []) if str(v).strip()]
    return out if out else list(DEFAULT_CONFIGS)


def _score(row: Dict[str, Any], max_trades_per_year: float) -> float:
    return score_candidate_row(
        row.get("metrics_5y", {}),
        row.get("metrics_10y", {}),
        max_trades_per_year,
    )


def _write_partial_out(
    out_path: Path | None,
    membership_source: str,
    universe_info: Dict[str, Any],
    transaction_cost_bps: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    rows: List[Dict[str, Any]],
) -> None:
    if out_path is None:
        return
    payload = {
        "membership_source": membership_source,
        "universe": universe_info,
        "realistic_costs": {
            "transaction_cost_bps": transaction_cost_bps,
            "entry_slippage_bps": entry_slippage_bps,
            "exit_slippage_bps": exit_slippage_bps,
        },
        "rows": sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_frontier_rows(frontier_path: Path) -> List[Dict[str, Any]]:
    raw = frontier_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    rows: List[Dict[str, Any]] = []
    if raw[:1] in {"{", "["}:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            items = payload.get("rows", [])
        elif isinstance(payload, list):
            items = payload
        else:
            items = []
        for item in items:
            cfg = item.get("config") if isinstance(item, dict) else None
            if isinstance(cfg, dict):
                rows.append(cfg)
        return rows

    for line in raw.splitlines():
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

    config_args = [str(v) for v in (args.config or []) if str(v).strip()]
    if config_args:
        for rel_path in config_args:
            path = (ROOT / rel_path).resolve()
            cfg = json.loads(path.read_text(encoding="utf-8"))
            _append_cfg(cfg)
    elif not args.frontier_log:
        for rel_path in list(DEFAULT_CONFIGS):
            path = (ROOT / rel_path).resolve()
            cfg = json.loads(path.read_text(encoding="utf-8"))
            _append_cfg(cfg)

    if args.frontier_log:
        rows = _load_frontier_rows((ROOT / args.frontier_log).resolve())
        for cfg in rows[: args.frontier_limit]:
            _append_cfg(cfg)

    return out


def _apply_realistic_costs(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = enforce_cash_only(cfg)
    out["transaction_cost_bps"] = float(args.transaction_cost_bps)
    out["entry_slippage_bps"] = float(args.entry_slippage_bps)
    out["exit_slippage_bps"] = float(args.exit_slippage_bps)
    out["slippage_bps"] = 0.0
    out["log_regime_skips"] = False
    return out


def _limit_prepared_universe(
    prepared: Any,
    limit: int,
    mode: str,
    sample_seed: int,
    explicit_symbols: List[str] | None = None,
    symbol_file: str = "",
) -> tuple[Any, Dict[str, Any]]:
    enriched = getattr(prepared, "enriched", {}) or {}
    total = len(enriched)
    explicit = sorted({str(sym).upper() for sym in (explicit_symbols or []) if str(sym).strip()})
    info = {
        "limit": int(limit),
        "mode": str(mode),
        "sample_seed": int(sample_seed),
        "prepared_symbols": int(total),
        "selected_symbols": int(total),
        "symbol_file": str(symbol_file or ""),
        "explicit_symbol_count": int(len(explicit)),
    }
    symbols = sorted(str(sym) for sym in enriched.keys())
    if explicit:
        wanted = set(explicit)
        symbols = [sym for sym in symbols if sym.upper() in wanted]

    if total <= 0 or not symbols:
        info["selected_symbols"] = 0
        limited = type(prepared)(enriched={}, all_dates=getattr(prepared, "all_dates"))
        return limited, info

    if limit > 0 and limit < len(symbols):
        if str(mode).lower() == "sample":
            chosen = sorted(random.Random(int(sample_seed)).sample(symbols, k=int(limit)))
        else:
            chosen = symbols[: int(limit)]
    else:
        chosen = symbols

    lookup = {str(sym): sym for sym in enriched.keys()}
    filtered = {lookup[sym]: enriched[lookup[sym]] for sym in chosen if sym in lookup}
    info["selected_symbols"] = int(len(filtered))
    limited = type(prepared)(enriched=filtered, all_dates=getattr(prepared, "all_dates"))
    return limited, info


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
    parser.add_argument("--config", action="append", default=[])
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
    parser.add_argument("--symbol-file", default="")
    parser.add_argument("--universe-limit", type=int, default=0)
    parser.add_argument("--universe-limit-mode", choices=["sample", "first"], default="sample")
    parser.add_argument("--universe-sample-seed", type=int, default=42)
    parser.add_argument("--skip-10y", action="store_true")
    parser.add_argument("--skip-5y", action="store_true")
    parser.add_argument("--walkforward-top-n", type=int, default=0)
    parser.add_argument("--walkforward-start", default=DEFAULT_WALKFORWARD_START)
    parser.add_argument("--walkforward-frictions", default="10")
    parser.add_argument("--walkforward-min-oos-cagr-pct", type=float, default=DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT)
    parser.add_argument("--walkforward-max-full-dd-pct", type=float, default=DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT)
    parser.add_argument("--walkforward-universe-limit", type=int, default=0)
    parser.add_argument("--walkforward-universe-limit-mode", choices=["sample", "first"], default="sample")
    parser.add_argument("--walkforward-universe-sample-seed", type=int, default=42)
    parser.add_argument("--walkforward-prepared-cache", default="")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with open((ROOT / args.cache).resolve(), "rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    symbol_file = str(args.symbol_file or "").strip()
    explicit_symbols = load_symbol_file(symbol_file) if symbol_file else []
    prepared, universe_info = _limit_prepared_universe(
        prepared,
        int(args.universe_limit),
        str(args.universe_limit_mode),
        int(args.universe_sample_seed),
        explicit_symbols,
        symbol_file,
    )
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}
    print(
        f"[prep] prepared_symbols={len(getattr(prepared, 'enriched', {}) or {})} "
        f"membership={membership_source} universe_limit={universe_info['limit']} mode={universe_info['mode']}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    out_path = (ROOT / args.out).resolve() if args.out else None
    for raw_cfg in _candidate_configs(args):
        cfg = _apply_realistic_costs(raw_cfg, args)
        name = str(cfg.get("name", "Unnamed"))
        print(f"[run] {name}", flush=True)
        raw_allow_margin = bool(raw_cfg.get("allow_margin", False))
        raw_bull_exposure = float(raw_cfg.get("max_total_exposure_pct_bull", 1.0) or 1.0)
        raw_bear_exposure = float(raw_cfg.get("max_total_exposure_pct_bear", 0.0) or 0.0)
        if raw_allow_margin or raw_bull_exposure > 1.0 or raw_bear_exposure > 1.0:
            print(
                "  [cash-only clamp] "
                f"allow_margin={raw_allow_margin} "
                f"bull={raw_bull_exposure:.2f} bear={raw_bear_exposure:.2f}",
                flush=True,
            )
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
        m5 = row["metrics_5y"]
        m10 = row["metrics_10y"]
        print(
            f"  => 5Y CAGR={float(m5.get('cagr_pct', 0.0) or 0.0):.2f}% "
            f"DD={float(m5.get('max_dd_pct', 0.0) or 0.0):.2f}% "
            f"TPY={float(m5.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"10Y CAGR={float(m10.get('cagr_pct', 0.0) or 0.0):.2f}% "
            f"DD={float(m10.get('max_dd_pct', 0.0) or 0.0):.2f}% "
            f"TPY={float(m10.get('trades_per_year', 0.0) or 0.0):.1f} "
            f"W12M={float(m10.get('worst_12m_return_pct', 0.0) or 0.0):.2f}% | "
            f"Score={row['score']:.2f}",
            flush=True,
        )
        rows.append(row)
        _write_partial_out(
            out_path,
            membership_source,
            universe_info,
            args.transaction_cost_bps,
            args.entry_slippage_bps,
            args.exit_slippage_bps,
            rows,
        )

    if int(args.walkforward_top_n) > 0:
        report_dir = None
        if out_path is not None:
            report_dir = out_path.parent / f"{out_path.stem}_walkforward"
        rows = replay_ranked_candidates(
            rows,
            top_n=int(args.walkforward_top_n),
            start_date=str(args.walkforward_start),
            end_date=str(args.end_date),
            transaction_cost_bps=float(args.transaction_cost_bps),
            friction_values=parse_friction_values(args.walkforward_frictions),
            min_oos_cagr_pct=float(args.walkforward_min_oos_cagr_pct),
            max_full_drawdown_pct=float(args.walkforward_max_full_dd_pct),
            universe_limit=int(args.walkforward_universe_limit),
            universe_limit_mode=str(args.walkforward_universe_limit_mode),
            universe_sample_seed=int(args.walkforward_universe_sample_seed),
            prepared_cache_path=str(args.walkforward_prepared_cache),
            report_dir=report_dir,
            report_prefix="frontier",
        )
    else:
        rows.sort(key=lambda item: item.get("score", -1_000_000.0), reverse=True)
    out = {
        "membership_source": membership_source,
        "universe": universe_info,
        "realistic_costs": {
            "transaction_cost_bps": args.transaction_cost_bps,
            "entry_slippage_bps": args.entry_slippage_bps,
            "exit_slippage_bps": args.exit_slippage_bps,
        },
        "walkforward_gate": {
            "top_n": int(args.walkforward_top_n),
            "start_date": str(args.walkforward_start),
            "frictions": list(parse_friction_values(args.walkforward_frictions)),
            "min_oos_cagr_pct": float(args.walkforward_min_oos_cagr_pct),
            "max_full_dd_pct": float(args.walkforward_max_full_dd_pct),
            "universe_limit": int(args.walkforward_universe_limit),
            "universe_limit_mode": str(args.walkforward_universe_limit_mode),
            "universe_sample_seed": int(args.walkforward_universe_sample_seed),
            "prepared_cache": str(args.walkforward_prepared_cache or ""),
        },
        "rows": rows,
    }
    print(json.dumps(out, indent=2))
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
