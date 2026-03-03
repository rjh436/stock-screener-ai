#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day, get_universe_symbols_pit_window_with_meta
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if pd.isna(out):
        return float(default)
    return out


def _profit_factor(trades_list: list[dict[str, Any]]) -> float:
    gross_wins = 0.0
    gross_losses = 0.0
    for trade in trades_list or []:
        if str(trade.get("Reason", "")).upper() == "PYRAMID_ADD":
            continue
        pnl = _safe_float(trade.get("PnL"), 0.0)
        if pnl > 0:
            gross_wins += pnl
        elif pnl < 0:
            gross_losses += abs(pnl)
    if gross_losses <= 0.0:
        return 3.0 if gross_wins > 0 else 0.0
    return gross_wins / gross_losses


def _score(metrics: dict[str, Any]) -> float:
    cagr = _safe_float(metrics.get("cagr_pct"), 0.0)
    max_dd = abs(_safe_float(metrics.get("max_drawdown_pct"), 0.0))
    pf = _safe_float(metrics.get("profit_factor"), 0.0)
    trades = int(_safe_float(metrics.get("total_trades"), 0))
    same_day_open = int(_safe_float(metrics.get("same_day_open_entries"), 0))
    max_gross = _safe_float(metrics.get("max_gross_exposure_pct"), 0.0)

    if same_day_open > 0:
        return -1_000_000.0 - same_day_open
    if max_gross > 1.001:
        return -900_000.0 - (max_gross * 1000.0)

    score = cagr
    score += min(pf, 3.0) * 2.5
    score -= max(0.0, max_dd - 22.0) * 0.8
    score -= max(0, 120 - trades) * 0.02
    return float(score)


def _candidate_tweaks() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("BASELINE", {}),
        (
            "FAST_PYR_REF",
            {
                "pyramid_threshold": 0.06,
                "pyramid_fraction": 0.75,
                "pyramid_max_adds": 2,
            },
        ),
        (
            "FAST_PYR_AGG",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
            },
        ),
        (
            "FAST_PYR_EP_ON",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "pyramid_ep_enabled": True,
            },
        ),
        (
            "FAST_PYR_EP_ON_SCORE20",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "pyramid_ep_enabled": True,
                "min_entry_score": 20,
            },
        ),
        (
            "FAST_PYR_SCORE20",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "min_entry_score": 20,
            },
        ),
        (
            "FAST_PYR_RS80_SCORE20",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "rs_gate_min": 80,
                "min_entry_score": 20,
            },
        ),
        (
            "FAST_PYR_LOOSE_EP",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "ep_gap_pct": 5,
                "ep_vol_mult": 2.5,
                "ep_close_near_high_min": 0.5,
            },
        ),
        (
            "FAST_PYR_VCP_LOOSER",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "vcp_breakout_volume_mult": 1.5,
            },
        ),
        (
            "FAST_PYR_LOWER_FRICTION",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "transaction_cost_bps": 1.0,
                "entry_slippage_bps": 3.0,
                "exit_slippage_bps": 5.0,
            },
        ),
        (
            "FAST_PYR_STRICT_FRICTION",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "transaction_cost_bps": 3.0,
                "entry_slippage_bps": 6.0,
                "exit_slippage_bps": 9.0,
            },
        ),
        (
            "FAST_PYR_MORE_CONCENTRATED",
            {
                "pyramid_threshold": 0.05,
                "pyramid_fraction": 1.0,
                "pyramid_max_adds": 2,
                "max_positions": 2,
                "max_pos_size_pct": 0.5,
                "vcp_max_pos_size_pct": 0.5,
                "ep_max_pos_size_pct": 0.5,
            },
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark focused Alpha B4 candidates on full PIT Russell 3000 window."
    )
    parser.add_argument("--start", default="2021-03-02")
    parser.add_argument("--end", default="2026-03-02")
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best candidate params back into Superperformance Alpha B4 in generated_strategies.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generated_path = ROOT / "config" / "generated_strategies.json"
    cfgs = json.loads(generated_path.read_text(encoding="utf-8"))
    if not isinstance(cfgs, list):
        raise RuntimeError("generated_strategies.json must be a list.")

    alpha_idx = None
    for i, item in enumerate(cfgs):
        if isinstance(item, dict) and str(item.get("name", "")) == "Superperformance Alpha B4":
            alpha_idx = i
            break
    if alpha_idx is None:
        raise RuntimeError("Superperformance Alpha B4 not found in generated_strategies.json")
    base_cfg = copy.deepcopy(cfgs[alpha_idx])

    symbols, universe_source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", args.start, args.end)
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if not symbols:
        raise RuntimeError("No symbols resolved for PIT Russell 3000 window.")
    print(
        f"[prep] Russell 3000 symbols={len(symbols)} source={universe_source} "
        f"window={args.start} -> {args.end}"
    )

    days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days + 420
    raw_data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
    g_raw = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days, backtest_mode=True) or {}
    global_data = {
        "SPY": g_raw.get("SPY"),
        "VIX": g_raw.get("$VIX") if g_raw.get("$VIX") is not None else g_raw.get("VIX"),
    }
    prepared = prepare_backtest_data(raw_data, symbol_universe=symbols, start_date=args.start, global_data=global_data)
    all_dates_raw = getattr(prepared, "all_dates", None)
    all_dates = list(all_dates_raw) if all_dates_raw is not None else []
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates)
    print(
        f"[prep] prepared_symbols={len(getattr(prepared, 'enriched', {}) or {})} "
        f"membership={membership_source} days={len(all_dates)}"
    )

    rows: list[dict[str, Any]] = []
    candidates = _candidate_tweaks()
    for idx, (label, tweaks) in enumerate(candidates, start=1):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(tweaks)
        cfg["name"] = f"AlphaBench::{label}"
        strat = SuperperformanceStrategy(cfg)
        result = run_backtest(
            strat,
            prepared,
            start_cash=100000.0,
            start_date=args.start,
            end_date=args.end,
            global_data=global_data,
            universe_membership_by_day=membership_by_day,
            require_pit_membership=True,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected backtest result for {label}: {type(result)}")
        audit = result.get("audit_report") if isinstance(result.get("audit_report"), dict) else {}
        metrics = {
            "label": label,
            "cagr_pct": _safe_float(result.get("cagr"), 0.0) * 100.0,
            "max_drawdown_pct": _safe_float(result.get("max_drawdown_pct"), 0.0) * 100.0,
            "final_value": _safe_float(result.get("final_value"), 0.0),
            "total_trades": int(_safe_float(result.get("total_trades"), 0.0)),
            "hit_rate_pct": _safe_float(result.get("hit_rate"), 0.0),
            "profit_factor": _profit_factor(result.get("trades_list") or []),
            "same_day_open_entries": int(_safe_float(audit.get("same_day_open_entries"), 0.0)),
            "max_gross_exposure_pct": _safe_float(audit.get("max_gross_exposure_pct"), 0.0),
            "tweaks": tweaks,
        }
        metrics["score"] = _score(metrics)
        rows.append(metrics)
        print(
            f"[{idx:02d}/{len(candidates)}] {label:<28} "
            f"CAGR={metrics['cagr_pct']:.2f}% DD={metrics['max_drawdown_pct']:.2f}% "
            f"Trades={metrics['total_trades']} PF={metrics['profit_factor']:.2f} "
            f"SameDayOpen={metrics['same_day_open_entries']} Score={metrics['score']:.2f}"
        )

    rows_sorted = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    champion = rows_sorted[0]
    print(
        "\n[champion] "
        f"{champion['label']} CAGR={champion['cagr_pct']:.2f}% "
        f"DD={champion['max_drawdown_pct']:.2f}% PF={champion['profit_factor']:.2f} "
        f"Trades={champion['total_trades']}"
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {"start": args.start, "end": args.end},
        "universe_source": universe_source,
        "membership_source": membership_source,
        "rows": rows_sorted,
        "champion": champion,
    }
    exports_dir = ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = exports_dir / f"alpha_candidate_bench_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {report_path}")

    if args.apply_best:
        cfgs[alpha_idx].update(champion["tweaks"])
        generated_path.write_text(json.dumps(cfgs, indent=2), encoding="utf-8")
        print("[apply] Updated Superperformance Alpha B4 in generated_strategies.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
