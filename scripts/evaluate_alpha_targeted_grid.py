#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from itertools import product
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
from scripts.evaluate_low_turnover_alpha_frontier import _apply_realistic_costs, _pack_metrics, _score

DEFAULT_SEEDS = [
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b17.json",
    "tmp/top_alpha_frontier_candidate.json",
]

PARAM_CHOICES: Dict[str, List[Any]] = {
    "rs_gate_min": [85, 88, 90],
    "min_entry_score": [25, 30, 35],
    "vcp_lookback_bars": [25, 30],
    "breakout_buffer": [0.0005, 0.001],
    "max_stop_pct": [0.05, 0.06],
    "stop_limit_pct": [0.02, 0.03],
    "ep_gap_pct": [6, 8],
    "ep_close_near_high_min": [0.55, 0.65],
    "pyramid_threshold": [0.06, 0.08],
    "pyramid_fraction": [0.5, 0.67],
    "vcp_gap_chase_max_pct": [0.0, 0.01],
    "vcp_gap_chase_rs_min": [90, 95],
}


def _resolve_seed_args(values: list[str] | None) -> list[str]:
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
    out["max_positions"] = max(2, min(int(out.get("max_positions", 3) or 3), 4))
    bull = max(0.5, min(float(out.get("max_total_exposure_pct_bull", 1.0) or 1.0), 1.0))
    out["max_total_exposure_pct_bull"] = bull
    cap = bull / float(out["max_positions"])
    out["max_pos_size_pct"] = max(0.10, min(float(out.get("max_pos_size_pct", cap) or cap), 0.5, cap))
    out["vcp_max_pos_size_pct"] = min(float(out.get("vcp_max_pos_size_pct", out["max_pos_size_pct"]) or out["max_pos_size_pct"]), out["max_pos_size_pct"])
    out["ep_max_pos_size_pct"] = min(float(out.get("ep_max_pos_size_pct", out["max_pos_size_pct"]) or out["max_pos_size_pct"]), out["max_pos_size_pct"])
    out["risk_per_trade"] = max(0.02, min(float(out.get("risk_per_trade", 0.1) or 0.1), 0.12))
    out["market_exposure_mode"] = str(out.get("market_exposure_mode", "exposure") or "exposure").lower()
    if out["market_exposure_mode"] not in {"exposure", "scaled", "hybrid", "filter"}:
        out["market_exposure_mode"] = "exposure"
    out["use_market_regime_traffic_light"] = bool(out.get("use_market_regime_traffic_light", False))
    out["traffic_light_block_exposure_mode"] = bool(out.get("traffic_light_block_exposure_mode", out["use_market_regime_traffic_light"]))
    bear_cash = str(out.get("bear_cash_mode", "off") or "off").lower()
    if bear_cash not in {"off", "hard", "cash", "all_cash"}:
        bear_cash = "off"
    out["bear_cash_mode"] = bear_cash
    if bear_cash in {"hard", "cash", "all_cash"}:
        out["max_total_exposure_pct_bear"] = 0.0
        out["bear_max_positions"] = 0
    else:
        out["max_total_exposure_pct_bear"] = max(0.0, min(float(out.get("max_total_exposure_pct_bear", 1.0) or 1.0), 1.0))
        out["bear_max_positions"] = max(0, min(int(out.get("bear_max_positions", 1) or 1), out["max_positions"]))
    out["yellow_risk_scalar"] = max(0.1, min(float(out.get("yellow_risk_scalar", 0.8) or 0.8), 1.0))
    out["orange_risk_scalar"] = max(0.05, min(float(out.get("orange_risk_scalar", 0.45) or 0.45), 1.0))
    out["yellow_max_positions"] = max(1, min(int(out.get("yellow_max_positions", out["max_positions"]) or out["max_positions"]), out["max_positions"]))
    out["orange_max_positions"] = max(1, min(int(out.get("orange_max_positions", out["yellow_max_positions"]) or out["yellow_max_positions"]), out["yellow_max_positions"]))
    out["stop_limit_pct"] = max(0.01, min(float(out.get("stop_limit_pct", 0.03) or 0.03), 0.10))
    out["max_stop_pct"] = max(0.03, min(float(out.get("max_stop_pct", 0.06) or 0.06), 0.12))
    out["ep_max_stop_pct"] = max(0.05, min(float(out.get("ep_max_stop_pct", 0.12) or 0.12), 0.20))
    out["stop_loss_atr_bull"] = max(1.5, min(float(out.get("stop_loss_atr_bull", 3.5) or 3.5), 6.0))
    out["stop_loss_atr_bear"] = max(0.3, min(float(out.get("stop_loss_atr_bear", 0.75) or 0.75), 2.0))
    out["time_stop_days"] = max(1, min(int(out.get("time_stop_days", 7) or 7), 20))
    out["time_stop_profit_pct"] = max(0.0, min(float(out.get("time_stop_profit_pct", 0.01) or 0.01), 0.05))
    out["pyramid_max_adds"] = max(0, min(int(out.get("pyramid_max_adds", 2) or 2), 4))
    out["trend_template_mode"] = str(out.get("trend_template_mode", "classic") or "classic").lower()
    if out["trend_template_mode"] not in {"classic", "strict"}:
        out["trend_template_mode"] = "classic"
    return out


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


def _one_step_neighbors(seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for key, options in PARAM_CHOICES.items():
        cur = seed.get(key)
        for value in options:
            if value == cur:
                continue
            child = copy.deepcopy(seed)
            child[key] = value
            out.append(_normalize(child))
    return out


def _two_step_neighbors(seed: Dict[str, Any], params: list[str]) -> List[Dict[str, Any]]:
    out = []
    if len(params) < 2:
        return out
    for a_i in range(len(params)):
        for b_i in range(a_i + 1, len(params)):
            a = params[a_i]
            b = params[b_i]
            cur_a = seed.get(a)
            cur_b = seed.get(b)
            for va in PARAM_CHOICES[a]:
                for vb in PARAM_CHOICES[b]:
                    if va == cur_a and vb == cur_b:
                        continue
                    if va == cur_a or vb == cur_b:
                        continue
                    child = copy.deepcopy(seed)
                    child[a] = va
                    child[b] = vb
                    out.append(_normalize(child))
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted Alpha grid around surviving seeds under realistic friction.")
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
    p.add_argument("--five-year-floor", type=float, default=30.0)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--progress", default="")
    p.add_argument("--out", default="tmp/alpha_targeted_grid_latest.json")
    return p.parse_args()


def _write_checkpoint(path: Path | None, payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    if path is None:
        return
    checkpoint = dict(payload)
    checkpoint["rows"] = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    with open((ROOT / args.cache).resolve(), "rb") as handle:
        prepared = pickle.load(handle)
    prepared = opt.compress_data(prepared)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=int(args.trading_days), backtest_mode=True) or {}

    seeds: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for rel in _resolve_seed_args(args.seed_config):
        cfg = json.loads((ROOT / rel).resolve().read_text(encoding="utf-8"))
        cfg = _normalize(_apply_realistic_costs(cfg, args))
        sig = _signature(cfg)
        if sig not in seen:
            seen.add(sig)
            seeds.append(cfg)

    candidates: List[Dict[str, Any]] = []
    for seed in seeds:
        candidates.append(seed)
        for child in _one_step_neighbors(seed):
            child = _normalize(_apply_realistic_costs(child, args))
            sig = _signature(child)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append(child)

    rows: List[Dict[str, Any]] = []
    out_path = (ROOT / args.out).resolve()
    progress_path = (ROOT / args.progress).resolve() if args.progress else None
    base_payload = {
        "membership_source": membership_source,
        "soft_trades_per_year": float(args.soft_trades_per_year),
        "realistic_costs": {
            "transaction_cost_bps": args.transaction_cost_bps,
            "entry_slippage_bps": args.entry_slippage_bps,
            "exit_slippage_bps": args.exit_slippage_bps,
        },
    }
    for idx, cfg in enumerate(candidates, start=1):
        name = str(cfg.get("name", f"Candidate {idx}"))
        print(f"[{idx}/{len(candidates)}] {name}", flush=True)
        res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
        row = {
            "name": name,
            "config": cfg,
            "metrics_5y": _pack_metrics(res5, args.start_5y, args.end_date),
            "metrics_10y": {},
        }
        c5 = float(row["metrics_5y"].get("cagr_pct", 0.0) or 0.0)
        if c5 >= args.five_year_floor:
            res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)
            row["metrics_10y"] = _pack_metrics(res10, args.start_10y, args.end_date)
            row["score"] = _score(row, args.soft_trades_per_year)
        else:
            row["score"] = -1000.0 + c5
        rows.append(row)
        m5 = row["metrics_5y"]
        m10 = row["metrics_10y"]
        print(
            f"  => 5Y CAGR={float(m5.get('cagr_pct', 0.0) or 0.0):.2f}% "
            f"DD={float(m5.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m5.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"10Y CAGR={float(m10.get('cagr_pct', 0.0) or 0.0):.2f}% DD={float(m10.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m10.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"Score={row['score']:.2f}",
            flush=True,
        )

    ranked = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)
    top = ranked[: args.top_k]

    # second round: pairwise around the strongest validated candidates only
    second_candidates: List[Dict[str, Any]] = []
    for seed in top:
        cfg = seed["config"]
        # use the 4 highest-leverage knobs from the first-pass leaders
        pairwise_params = [
            "rs_gate_min",
            "min_entry_score",
            "pyramid_fraction",
            "vcp_gap_chase_max_pct",
            "breakout_buffer",
            "ep_close_near_high_min",
        ]
        for child in _two_step_neighbors(cfg, pairwise_params):
            child = _normalize(_apply_realistic_costs(child, args))
            sig = _signature(child)
            if sig in seen:
                continue
            seen.add(sig)
            second_candidates.append(child)

    for idx, cfg in enumerate(second_candidates, start=1):
        name = str(cfg.get("name", f"Round2 {idx}"))
        print(f"[round2 {idx}/{len(second_candidates)}] {name}", flush=True)
        res5 = _run_period(cfg, prepared, membership, global_data, args.start_5y, args.end_date)
        row = {
            "name": name,
            "config": cfg,
            "metrics_5y": _pack_metrics(res5, args.start_5y, args.end_date),
            "metrics_10y": {},
        }
        c5 = float(row["metrics_5y"].get("cagr_pct", 0.0) or 0.0)
        if c5 >= args.five_year_floor:
            res10 = _run_period(cfg, prepared, membership, global_data, args.start_10y, args.end_date)
            row["metrics_10y"] = _pack_metrics(res10, args.start_10y, args.end_date)
            row["score"] = _score(row, args.soft_trades_per_year)
        else:
            row["score"] = -1000.0 + c5
        rows.append(row)
        m5 = row["metrics_5y"]
        m10 = row["metrics_10y"]
        print(
            f"  => 5Y CAGR={float(m5.get('cagr_pct', 0.0) or 0.0):.2f}% "
            f"DD={float(m5.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m5.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"10Y CAGR={float(m10.get('cagr_pct', 0.0) or 0.0):.2f}% DD={float(m10.get('max_dd_pct', 0.0) or 0.0):.2f}% TPY={float(m10.get('trades_per_year', 0.0) or 0.0):.1f} | "
            f"Score={row['score']:.2f}",
            flush=True,
        )
        _write_checkpoint(progress_path, base_payload, rows)
        _write_checkpoint(out_path, base_payload, rows)

    ranked = sorted(rows, key=lambda item: item.get("score", -1_000_000.0), reverse=True)
    out = {
        **base_payload,
        "rows": ranked,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
