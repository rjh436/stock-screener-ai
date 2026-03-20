#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day
from execution.engine import run_backtest
from optimization.walkforward import (
    DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT,
    DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT,
    DEFAULT_WALKFORWARD_START,
    parse_friction_values,
    replay_ranked_candidates,
)
from strategies.strategy_loader import load_strategies
import optimize_superperformance as opt
from scripts.evaluate_low_turnover_alpha_frontier import _apply_realistic_costs, _limit_prepared_universe, _pack_metrics, _score

DEFAULT_SEEDS = [
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b17.json",
    "tmp/top_alpha_frontier_candidate.json",
]

MUTATION_SPACE: Dict[str, List[Any]] = {
    "rs_gate_min": [84, 85, 87, 88, 90],
    "min_entry_score": [20, 25, 30, 35],
    "vcp_lookback_bars": [25, 30, 35, 40],
    "vcp_breakout_volume_mult": [1.5, 1.75, 2.0],
    "breakout_buffer": [0.0005, 0.001, 0.002],
    "max_stop_pct": [0.05, 0.06, 0.07],
    "stop_limit_pct": [0.02, 0.03, 0.05],
    "stop_loss_atr_bull": [3.0, 3.5, 4.0],
    "stop_loss_atr_bear": [0.5, 0.75, 1.0],
    "time_stop_days": [5, 7, 10, 12],
    "time_stop_profit_pct": [0.0, 0.002, 0.005, 0.01],
    "ep_gap_pct": [6, 8, 10],
    "ep_close_near_high_min": [0.55, 0.65, 0.75],
    "ep_max_stop_pct": [0.10, 0.12, 0.15],
    "pyramid_threshold": [0.06, 0.08, 0.10],
    "pyramid_fraction": [0.25, 0.33, 0.50, 0.67],
    "pyramid_max_adds": [1, 2, 3],
    "max_positions": [2, 3, 4],
    "risk_per_trade": [0.06, 0.08, 0.10, 0.12],
    "max_pos_size_pct": [0.20, 0.25, 0.3333, 0.50],
    "max_total_exposure_pct_bull": [0.75, 0.85, 0.95, 1.0],
    "max_total_exposure_pct_bear": [0.0, 0.35, 0.7, 1.0],
    "market_exposure_mode": ["exposure", "scaled", "hybrid", "filter"],
    "use_market_regime_traffic_light": [False, True],
    "apply_traffic_light_position_caps_in_exposure": [False, True],
    "traffic_light_block_exposure_mode": [False, True],
    "bear_cash_mode": ["off", "hard"],
    "bear_max_positions": [0, 1, 2],
    "yellow_risk_scalar": [0.5, 0.6, 0.8, 1.0],
    "orange_risk_scalar": [0.15, 0.25, 0.45, 0.7],
    "yellow_max_positions": [2, 3, 4],
    "orange_max_positions": [0, 1, 2, 3],
    "vcp_gap_chase_max_pct": [0.0, 0.005, 0.01, 0.02],
    "vcp_gap_chase_rs_min": [90, 93, 95, 97],
    "vcp_gap_chase_score_min": [55, 65, 70, 75],
}


def _resolve_seed_args(values: List[str] | None) -> List[str]:
    out = [str(v) for v in (values or []) if str(v).strip()]
    return out if out else list(DEFAULT_SEEDS)


def _signature(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["type"] = "superperformance"
    out["signal_mode"] = "after_close"
    out["entry_day_stop_mode"] = "close_confirmed"
    out["enforce_next_day_exit_execution"] = True
    out["allow_margin"] = False
    out["log_regime_skips"] = False

    max_positions = max(1, min(int(out.get("max_positions", 3) or 3), 6))
    out["max_positions"] = max_positions

    max_expo_bull = max(0.5, min(float(out.get("max_total_exposure_pct_bull", 1.0) or 1.0), 1.0))
    out["max_total_exposure_pct_bull"] = max_expo_bull

    cap_by_slots = max_expo_bull / float(max_positions)
    max_pos_pct = max(0.05, min(float(out.get("max_pos_size_pct", cap_by_slots) or cap_by_slots), 0.50, cap_by_slots))
    out["max_pos_size_pct"] = max_pos_pct
    out["vcp_max_pos_size_pct"] = min(float(out.get("vcp_max_pos_size_pct", max_pos_pct) or max_pos_pct), max_pos_pct)
    out["ep_max_pos_size_pct"] = min(float(out.get("ep_max_pos_size_pct", max_pos_pct) or max_pos_pct), max_pos_pct)
    out["risk_per_trade"] = max(0.002, min(float(out.get("risk_per_trade", 0.10) or 0.10), 0.15))

    out["max_total_exposure_pct_bear"] = max(0.0, min(float(out.get("max_total_exposure_pct_bear", 1.0) or 1.0), 1.0))
    mode = str(out.get("market_exposure_mode", "exposure") or "exposure").lower()
    if mode not in {"exposure", "scaled", "hybrid", "filter"}:
        mode = "exposure"
    out["market_exposure_mode"] = mode

    tl = bool(out.get("use_market_regime_traffic_light", False))
    out["use_market_regime_traffic_light"] = tl
    out["traffic_light_block_exposure_mode"] = bool(out.get("traffic_light_block_exposure_mode", tl))
    out["apply_traffic_light_position_caps_in_exposure"] = bool(out.get("apply_traffic_light_position_caps_in_exposure", tl))

    bear_cash = str(out.get("bear_cash_mode", "off") or "off").lower()
    if bear_cash not in {"off", "hard", "cash", "all_cash"}:
        bear_cash = "off"
    out["bear_cash_mode"] = bear_cash
    if bear_cash in {"hard", "cash", "all_cash"}:
        out["max_total_exposure_pct_bear"] = 0.0
        out["bear_max_positions"] = 0
    else:
        out["bear_max_positions"] = max(0, min(int(out.get("bear_max_positions", 1) or 1), max_positions))

    out["yellow_risk_scalar"] = max(0.1, min(float(out.get("yellow_risk_scalar", 0.8) or 0.8), 1.0))
    out["orange_risk_scalar"] = max(0.05, min(float(out.get("orange_risk_scalar", 0.45) or 0.45), 1.0))
    out["yellow_max_positions"] = max(1, min(int(out.get("yellow_max_positions", max_positions) or max_positions), max_positions))
    out["orange_max_positions"] = max(0, min(int(out.get("orange_max_positions", out["yellow_max_positions"]) or out["yellow_max_positions"]), out["yellow_max_positions"]))

    out["stop_limit_pct"] = max(0.01, min(float(out.get("stop_limit_pct", 0.03) or 0.03), 0.10))
    out["max_stop_pct"] = max(0.03, min(float(out.get("max_stop_pct", 0.06) or 0.06), 0.12))
    out["ep_max_stop_pct"] = max(0.05, min(float(out.get("ep_max_stop_pct", 0.12) or 0.12), 0.20))
    out["stop_loss_atr_bull"] = max(1.5, min(float(out.get("stop_loss_atr_bull", 3.5) or 3.5), 6.0))
    out["stop_loss_atr_bear"] = max(0.3, min(float(out.get("stop_loss_atr_bear", 0.75) or 0.75), 2.0))
    out["time_stop_days"] = max(0, min(int(out.get("time_stop_days", 7) or 7), 40))
    out["time_stop_profit_pct"] = max(0.0, min(float(out.get("time_stop_profit_pct", 0.01) or 0.01), 0.05))
    out["pyramid_max_adds"] = max(0, min(int(out.get("pyramid_max_adds", 2) or 2), 4))
    out["trend_template_mode"] = str(out.get("trend_template_mode", "classic") or "classic").lower()
    if out["trend_template_mode"] not in {"classic", "strict"}:
        out["trend_template_mode"] = "classic"
    return out


def _mutate(parent: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    parent_sig = _signature(parent)
    for _ in range(24):
        child = copy.deepcopy(parent)
        for key in rng.sample(list(MUTATION_SPACE.keys()), rng.randint(3, 8)):
            options = MUTATION_SPACE[key]
            cur = child.get(key)
            choices = [v for v in options if v != cur]
            child[key] = rng.choice(choices if choices else options)
        child = _normalize(child)
        if _signature(child) != parent_sig:
            return child
    return _normalize(copy.deepcopy(parent))


def _run_period(cfg: Dict[str, Any], prepared: Any, membership: List[set], global_data: Dict[str, Any], start_date: str, end_date: str) -> Dict[str, Any]:
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
    p = argparse.ArgumentParser(description="Focused realistic-friction neighborhood search around Alpha winners.")
    p.add_argument("--seed-config", action="append", default=[])
    p.add_argument("--cache", default=str(ROOT / "data" / "cache_indicators.pkl"))
    p.add_argument("--start-5y", default="2021-03-02")
    p.add_argument("--start-10y", default="2016-03-02")
    p.add_argument("--end-date", default="2026-03-02")
    p.add_argument("--trading-days", type=int, default=3000)
    p.add_argument("--transaction-cost-bps", type=float, default=2.0)
    p.add_argument("--entry-slippage-bps", type=float, default=10.0)
    p.add_argument("--exit-slippage-bps", type=float, default=10.0)
    p.add_argument("--soft-trades-per-year", type=float, default=150.0)
    p.add_argument("--iterations", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--universe-limit", type=int, default=0)
    p.add_argument("--universe-limit-mode", choices=["sample", "first"], default="sample")
    p.add_argument("--universe-sample-seed", type=int, default=42)
    p.add_argument("--walkforward-top-n", type=int, default=0)
    p.add_argument("--walkforward-start", default=DEFAULT_WALKFORWARD_START)
    p.add_argument("--walkforward-frictions", default="10")
    p.add_argument("--walkforward-min-oos-cagr-pct", type=float, default=DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT)
    p.add_argument("--walkforward-max-full-dd-pct", type=float, default=DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT)
    p.add_argument("--walkforward-universe-limit", type=int, default=0)
    p.add_argument("--walkforward-universe-limit-mode", choices=["sample", "first"], default="sample")
    p.add_argument("--walkforward-universe-sample-seed", type=int, default=42)
    p.add_argument("--walkforward-prepared-cache", default="")
    p.add_argument("--progress", default="tmp/alpha_neighborhood_progress.jsonl")
    p.add_argument("--out", default="tmp/alpha_neighborhood_latest.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    with open((ROOT / args.cache).resolve(), "rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    prepared, universe_info = _limit_prepared_universe(
        prepared,
        int(args.universe_limit),
        str(args.universe_limit_mode),
        int(args.universe_sample_seed),
    )
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}
    print(
        f"[prep] prepared_symbols={len(getattr(prepared, 'enriched', {}) or {})} "
        f"membership={membership_source} universe_limit={universe_info['limit']} mode={universe_info['mode']}",
        flush=True,
    )

    seed_cfgs = []
    seen = set()
    for rel in _resolve_seed_args(args.seed_config):
        cfg = json.loads((ROOT / rel).resolve().read_text(encoding="utf-8"))
        cfg = _normalize(_apply_realistic_costs(cfg, args))
        sig = _signature(cfg)
        if sig not in seen:
            seen.add(sig)
            seed_cfgs.append(cfg)

    candidates = list(seed_cfgs)
    while len(candidates) < len(seed_cfgs) + args.iterations:
        parent = rng.choice(seed_cfgs if rng.random() < 0.35 else candidates)
        child = _normalize(_apply_realistic_costs(_mutate(parent, rng), args))
        sig = _signature(child)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append(child)

    rows: List[Dict[str, Any]] = []
    progress_path = (ROOT / args.progress).resolve()
    out_path = (ROOT / args.out).resolve()
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    for idx, cfg in enumerate(candidates, start=1):
        name = str(cfg.get("name", f"Candidate {idx}"))
        print(f"[{idx}/{len(candidates)}] {name}", flush=True)
        res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
        row = {
            "name": name,
            "config": cfg,
            "metrics_5y": _pack_metrics(res5, args.start_5y, args.end_date),
        }
        c5 = float(row["metrics_5y"].get("cagr_pct", 0.0) or 0.0)
        if c5 >= 28.0:
            res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)
            row["metrics_10y"] = _pack_metrics(res10, args.start_10y, args.end_date)
            row["score"] = _score(row, args.soft_trades_per_year)
        else:
            row["metrics_10y"] = {}
            row["score"] = -1000.0 + c5
        rows.append(row)
        progress_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        ranked = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)
        payload = {
            "membership_source": membership_source,
            "universe": universe_info,
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
        m5 = row["metrics_5y"]
        m10 = row["metrics_10y"]
        print(
            f"  => 5Y CAGR={float(m5.get('cagr_pct', 0.0) or 0.0):.2f}% "
            f"DD={float(m5.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m5.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"10Y CAGR={float(m10.get('cagr_pct', 0.0) or 0.0):.2f}% DD={float(m10.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m10.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"Score={row['score']:.2f}",
            flush=True,
        )

    if int(args.walkforward_top_n) > 0:
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
            report_dir=out_path.parent / f"{out_path.stem}_walkforward",
            report_prefix="alpha_neighborhood",
        )
    else:
        rows = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)

    payload = {
        "membership_source": membership_source,
        "universe": universe_info,
        "soft_trades_per_year": float(args.soft_trades_per_year),
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(out_path)


if __name__ == "__main__":
    main()
