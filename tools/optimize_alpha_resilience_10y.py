#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day, get_universe_symbols_pit_window_with_meta
from execution.engine import prepare_backtest_data, run_backtest
from optimization.robustness import baseline_promotion_status, pack_result_metrics, promotion_gate_status, rejection_score
from optimization.walkforward import promotion_walkforward_status
from strategies.superperformance import SuperperformanceStrategy


@dataclass
class Context:
    start: str
    end: str
    prepared: Any
    global_data: dict[str, Any]
    membership_by_day: list[set[str] | frozenset[str] | None] | None
    universe_source: str
    membership_source: str
    symbols_requested: int
    symbols_loaded: int


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
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


def _parse_friction_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except Exception:
            continue
        if value >= 0.0:
            values.append(float(value))
    return values or [10.0]


def _prepare_context(start: str, end: str) -> Context:
    symbols, universe_source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start, end)
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if not symbols:
        raise RuntimeError("No symbols resolved for PIT Russell 3000 window.")
    print(
        f"[prep] Russell 3000 symbols={len(symbols)} source={universe_source} "
        f"window={start} -> {end}"
    )
    days = (pd.Timestamp(end) - pd.Timestamp(start)).days + 420
    raw_data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
    g_raw = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days, backtest_mode=True) or {}
    global_data = {
        "SPY": g_raw.get("SPY"),
        "VIX": g_raw.get("$VIX") if g_raw.get("$VIX") is not None else g_raw.get("VIX"),
    }
    prepared = prepare_backtest_data(raw_data, symbol_universe=symbols, start_date=start, global_data=global_data)
    symbols_loaded = len(getattr(prepared, "enriched", {}) or {})
    all_dates_raw = getattr(prepared, "all_dates", None)
    all_dates = list(all_dates_raw) if all_dates_raw is not None else []
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates)
    print(
        f"[prep] loaded_symbols={symbols_loaded}/{len(symbols)} "
        f"membership={membership_source} days={len(all_dates)}"
    )
    return Context(
        start=start,
        end=end,
        prepared=prepared,
        global_data=global_data,
        membership_by_day=membership_by_day,
        universe_source=universe_source,
        membership_source=membership_source,
        symbols_requested=len(symbols),
        symbols_loaded=symbols_loaded,
    )


def _equity_df(equity_curve: list[dict[str, Any]]) -> pd.DataFrame:
    ec = pd.DataFrame(equity_curve or [])
    if ec.empty or "Date" not in ec.columns or "Equity" not in ec.columns:
        return pd.DataFrame(columns=["Date", "Equity"])
    ec["Date"] = pd.to_datetime(ec["Date"], errors="coerce")
    ec["Equity"] = pd.to_numeric(ec["Equity"], errors="coerce")
    ec = ec.dropna(subset=["Date", "Equity"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return ec


def _window_metrics(ec: pd.DataFrame, start: str, end: str) -> dict[str, float]:
    if ec.empty:
        return {"cagr_pct": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
    st = pd.Timestamp(start)
    en = pd.Timestamp(end)
    seg = ec[(ec["Date"] >= st) & (ec["Date"] <= en)].copy()
    if len(seg) < 2:
        return {"cagr_pct": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
    start_eq = _safe_float(seg["Equity"].iloc[0], 0.0)
    end_eq = _safe_float(seg["Equity"].iloc[-1], 0.0)
    if start_eq <= 0.0 or end_eq <= 0.0:
        return {"cagr_pct": 0.0, "max_drawdown_pct": 0.0, "total_return_pct": 0.0}
    span_years = (seg["Date"].iloc[-1] - seg["Date"].iloc[0]).days / 365.25
    if span_years <= 0.0:
        cagr_pct = 0.0
    else:
        cagr_pct = ((end_eq / start_eq) ** (1.0 / span_years) - 1.0) * 100.0
    dd = (seg["Equity"] / seg["Equity"].cummax() - 1.0).min() * 100.0
    total_return = (end_eq / start_eq - 1.0) * 100.0
    return {
        "cagr_pct": float(cagr_pct),
        "max_drawdown_pct": float(abs(dd)),
        "total_return_pct": float(total_return),
    }


def _candidate_tweaks(candidate_set: str) -> list[tuple[str, dict[str, Any]]]:
    if candidate_set == "explore":
        return [
            ("BASELINE", {}),
            ("AGGR_RISK12", {"risk_per_trade": 0.12}),
            (
                "CONC_2POS",
                {
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                },
            ),
            (
                "CONC_2POS_RISK12",
                {
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                    "risk_per_trade": 0.12,
                },
            ),
            (
                "CONC_2POS_BEAR45",
                {
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                    "max_total_exposure_pct_bear": 0.45,
                },
            ),
            (
                "WIDER_ENTRY",
                {
                    "rs_gate_min": 80,
                    "min_entry_score": 20,
                    "ep_gap_pct": 5,
                    "ep_vol_mult": 2.5,
                    "vcp_breakout_volume_mult": 1.5,
                },
            ),
            (
                "WIDER_ENTRY_CONC",
                {
                    "rs_gate_min": 80,
                    "min_entry_score": 20,
                    "ep_gap_pct": 5,
                    "ep_vol_mult": 2.5,
                    "vcp_breakout_volume_mult": 1.5,
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                },
            ),
            (
                "ADAPTIVE_GREEN_GATES",
                {
                    "adaptive_breakout_gates_enabled": True,
                    "adaptive_green_rs_min": 80,
                    "adaptive_green_runup_min_pct": 20,
                    "adaptive_green_adr_min_pct": 2.5,
                    "adaptive_green_min_avg_dollar_volume_50": 5000000,
                },
            ),
            (
                "ADAPTIVE_GREEN_CONC",
                {
                    "adaptive_breakout_gates_enabled": True,
                    "adaptive_green_rs_min": 80,
                    "adaptive_green_runup_min_pct": 20,
                    "adaptive_green_adr_min_pct": 2.5,
                    "adaptive_green_min_avg_dollar_volume_50": 5000000,
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                },
            ),
            (
                "ADAPTIVE_GREEN_HOLD_LONGER_CONC",
                {
                    "adaptive_breakout_gates_enabled": True,
                    "adaptive_green_rs_min": 80,
                    "adaptive_green_runup_min_pct": 20,
                    "adaptive_green_adr_min_pct": 2.5,
                    "adaptive_green_min_avg_dollar_volume_50": 5000000,
                    "time_stop_days": 10,
                    "ep_time_stop_days": 6,
                    "vcp_time_stop_days": 15,
                    "vcp_dead_money_profit_pct": 0.01,
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                },
            ),
            (
                "HOLD_LONGER",
                {
                    "time_stop_days": 10,
                    "ep_time_stop_days": 6,
                    "vcp_time_stop_days": 15,
                    "vcp_dead_money_profit_pct": 0.01,
                },
            ),
            (
                "HOLD_LONGER_CONC",
                {
                    "time_stop_days": 10,
                    "ep_time_stop_days": 6,
                    "vcp_time_stop_days": 15,
                    "vcp_dead_money_profit_pct": 0.01,
                    "max_positions": 2,
                    "max_pos_size_pct": 0.5,
                    "vcp_max_pos_size_pct": 0.5,
                    "ep_max_pos_size_pct": 0.5,
                },
            ),
            ("NO_BREADTH_OVERLAY", {"use_market_breadth_overlay": False}),
            ("NO_TRAFFIC_LIGHT", {"use_market_regime_traffic_light": False}),
            (
                "BEAR_EXP_100",
                {
                    "max_total_exposure_pct_bear": 1.0,
                },
            ),
        ]

    return [
        ("BASELINE", {}),
        ("ADAPT_CAPS", {"apply_traffic_light_position_caps_in_exposure": True}),
        (
            "ADAPT_CAPS_BEAR45",
            {"apply_traffic_light_position_caps_in_exposure": True, "max_total_exposure_pct_bear": 0.45},
        ),
        (
            "ADAPT_CAPS_BEAR35",
            {"apply_traffic_light_position_caps_in_exposure": True, "max_total_exposure_pct_bear": 0.35},
        ),
        (
            "ADAPT_CAPS_BREADTH_BLOCK",
            {
                "apply_traffic_light_position_caps_in_exposure": True,
                "breadth_hard_block_entries": True,
                "breadth_entry_floor": 0.28,
                "breadth_risk_floor": 0.30,
            },
        ),
        (
            "ADAPT_CAPS_SOFTER_RISK",
            {"apply_traffic_light_position_caps_in_exposure": True, "risk_per_trade": 0.09},
        ),
        (
            "ADAPT_CAPS_DIV4",
            {
                "apply_traffic_light_position_caps_in_exposure": True,
                "max_positions": 4,
                "max_pos_size_pct": 0.25,
                "vcp_max_pos_size_pct": 0.25,
                "ep_max_pos_size_pct": 0.25,
            },
        ),
        (
            "ADAPT_CAPS_DIV4_BEAR45",
            {
                "apply_traffic_light_position_caps_in_exposure": True,
                "max_positions": 4,
                "max_pos_size_pct": 0.25,
                "vcp_max_pos_size_pct": 0.25,
                "ep_max_pos_size_pct": 0.25,
                "max_total_exposure_pct_bear": 0.45,
            },
        ),
        (
            "ADAPT_CAPS_PYR085",
            {"apply_traffic_light_position_caps_in_exposure": True, "pyramid_fraction": 0.85},
        ),
        (
            "ADAPT_CAPS_ONE_ADD",
            {"apply_traffic_light_position_caps_in_exposure": True, "pyramid_max_adds": 1},
        ),
        (
            "ADAPT_CAPS_PYR085_BEAR45",
            {
                "apply_traffic_light_position_caps_in_exposure": True,
                "pyramid_fraction": 0.85,
                "max_total_exposure_pct_bear": 0.45,
            },
        ),
        (
            "ADAPT_CAPS_BREADTH_SCALARS",
            {
                "apply_traffic_light_position_caps_in_exposure": True,
                "breadth_low_risk_scalar": 0.50,
                "breadth_mid_risk_scalar": 0.70,
                "breadth_high_risk_scalar": 0.90,
            },
        ),
    ]


def _base_score(row: dict[str, Any]) -> float:
    ok, code, _ = promotion_gate_status(row, metrics_5y=row, metrics_10y={}, max_drawdown_gate_pct=30.0)
    if not ok:
        return rejection_score(code or 999)

    full_cagr = _safe_float(row.get("full_cagr_pct"), 0.0)
    full_dd = _safe_float(row.get("full_max_drawdown_pct"), 0.0)
    pf = _safe_float(row.get("profit_factor"), 0.0)
    trades = int(_safe_float(row.get("total_trades"), 0.0))
    early_cagr = _safe_float(row.get("early_cagr_pct"), 0.0)
    bear_cagr = _safe_float(row.get("bear_cagr_pct"), 0.0)
    recent_cagr = _safe_float(row.get("recent_cagr_pct"), 0.0)
    bear_dd = _safe_float(row.get("bear_max_drawdown_pct"), 0.0)

    score = 0.50 * full_cagr
    score += 0.20 * early_cagr
    score += 0.20 * bear_cagr
    score += 0.10 * recent_cagr
    score += min(pf, 3.0) * 2.0
    score -= max(0.0, full_dd - 25.0) * 0.90
    score -= max(0.0, bear_dd - 22.0) * 0.80
    score -= max(0, 80 - trades) * 0.04
    if bear_cagr < 0.0:
        score -= abs(bear_cagr) * 0.6
    return float(score)


def _finalize_score(row: dict[str, Any]) -> float:
    score = _safe_float(row.get("base_score"), -1_000_000.0)
    stress_cagr = _safe_float(row.get("stress_cagr_pct"), np.nan)
    stress_dd = _safe_float(row.get("stress_max_drawdown_pct"), np.nan)
    stress_pf = _safe_float(row.get("stress_profit_factor"), np.nan)
    full_cagr = _safe_float(row.get("full_cagr_pct"), 0.0)

    if np.isfinite(stress_cagr):
        score += 0.25 * stress_cagr
        if np.isfinite(stress_pf):
            score += min(stress_pf, 3.0) * 0.8
        if np.isfinite(stress_dd):
            score -= max(0.0, stress_dd - 28.0) * 0.70
        decay = full_cagr - stress_cagr
        score -= max(0.0, decay - 4.0) * 1.6
    return float(score)


def _run_once(ctx: Context, config: dict[str, Any]) -> dict[str, Any]:
    strategy = SuperperformanceStrategy(config)
    result = run_backtest(
        strategy,
        ctx.prepared,
        start_cash=100000.0,
        start_date=ctx.start,
        end_date=ctx.end,
        global_data=ctx.global_data,
        universe_membership_by_day=ctx.membership_by_day,
        require_pit_membership=True,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected run_backtest result type: {type(result)}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Alpha B4 for 10Y resilience on PIT Russell 3000.")
    parser.add_argument("--start", default="2016-03-02")
    parser.add_argument("--end", default="2026-03-02")
    parser.add_argument("--strategy-name", default="Superperformance Alpha B4")
    parser.add_argument("--top-stress", type=int, default=4)
    parser.add_argument(
        "--candidate-set",
        choices=["defensive", "explore"],
        default="defensive",
        help="Candidate family to evaluate.",
    )
    parser.add_argument("--apply-best", action="store_true")
    parser.add_argument("--skip-walkforward-gate", action="store_true")
    parser.add_argument("--walkforward-frictions", default="10")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    generated_path = ROOT / "config" / "generated_strategies.json"
    cfgs = json.loads(generated_path.read_text(encoding="utf-8"))
    if not isinstance(cfgs, list):
        raise RuntimeError("generated_strategies.json must contain a list")

    target_idx = None
    for i, item in enumerate(cfgs):
        if isinstance(item, dict) and str(item.get("name", "")) == str(args.strategy_name):
            target_idx = i
            break
    if target_idx is None:
        raise RuntimeError(f"Strategy not found: {args.strategy_name}")
    base_cfg = copy.deepcopy(cfgs[target_idx])
    ctx = _prepare_context(args.start, args.end)

    early_start = args.start
    early_end = "2021-12-31"
    bear_start, bear_end = "2022-01-01", "2023-12-31"
    recent_start, recent_end = "2024-01-01", args.end

    rows: list[dict[str, Any]] = []
    candidates = _candidate_tweaks(args.candidate_set)
    for idx, (label, tweaks) in enumerate(candidates, start=1):
        cfg = copy.deepcopy(base_cfg)
        cfg.update(tweaks)
        cfg["name"] = f"AlphaRes10Y::{label}"
        result = _run_once(ctx, cfg)
        audit = result.get("audit_report") if isinstance(result.get("audit_report"), dict) else {}
        pf = _profit_factor(result.get("trades_list") or [])
        ec = _equity_df(result.get("equity_curve") or [])
        packed = pack_result_metrics(result, start_date=args.start, end_date=args.end)

        early = _window_metrics(ec, early_start, early_end)
        bear = _window_metrics(ec, bear_start, bear_end)
        recent = _window_metrics(ec, recent_start, recent_end)
        row = {
            "label": label,
            "tweaks": tweaks,
            **packed,
            "full_cagr_pct": packed["cagr_pct"],
            "full_max_drawdown_pct": packed["max_dd_pct"],
            "profit_factor": pf,
            "same_day_open_entries": int(_safe_float(audit.get("same_day_open_entries"), packed["same_day_open_entries"])),
            "stale_position_days": int(_safe_float(audit.get("stale_position_days"), 0.0)),
            "max_gross_exposure_pct": _safe_float(audit.get("max_gross_exposure_pct"), packed["max_gross_exposure_pct"]),
            "early_cagr_pct": early["cagr_pct"],
            "early_max_drawdown_pct": early["max_drawdown_pct"],
            "bear_cagr_pct": bear["cagr_pct"],
            "bear_max_drawdown_pct": bear["max_drawdown_pct"],
            "recent_cagr_pct": recent["cagr_pct"],
            "recent_max_drawdown_pct": recent["max_drawdown_pct"],
        }
        row["base_score"] = _base_score(row)
        rows.append(row)
        print(
            f"[{idx:02d}/{len(candidates)}] {label:<28} "
            f"10YCAGR={row['full_cagr_pct']:.2f}% 10YDD={row['full_max_drawdown_pct']:.2f}% "
            f"BearCAGR={row['bear_cagr_pct']:.2f}% Trades={row['total_trades']} "
            f"PF={row['profit_factor']:.2f} BaseScore={row['base_score']:.2f}"
        )

    ranked = sorted(rows, key=lambda r: float(r["base_score"]), reverse=True)
    stress_n = max(1, min(int(args.top_stress), len(ranked)))
    print(f"\n[stress] evaluating top {stress_n} under stricter friction...")
    for row in ranked[:stress_n]:
        label = row["label"]
        cfg = copy.deepcopy(base_cfg)
        cfg.update(row["tweaks"])
        cfg["name"] = f"AlphaRes10Y::{label}::stress"
        cfg["transaction_cost_bps"] = max(5.0, _safe_float(cfg.get("transaction_cost_bps"), 0.0))
        cfg["entry_slippage_bps"] = max(7.0, _safe_float(cfg.get("entry_slippage_bps"), 0.0))
        cfg["exit_slippage_bps"] = max(9.0, _safe_float(cfg.get("exit_slippage_bps"), 0.0))
        stress = _run_once(ctx, cfg)
        stress_pf = _profit_factor(stress.get("trades_list") or [])
        row["stress_cagr_pct"] = _safe_float(stress.get("cagr"), 0.0) * 100.0
        row["stress_max_drawdown_pct"] = _safe_float(stress.get("max_drawdown_pct"), 0.0) * 100.0
        row["stress_profit_factor"] = stress_pf
        print(
            f"[stress] {label:<28} "
            f"CAGR={row['stress_cagr_pct']:.2f}% DD={row['stress_max_drawdown_pct']:.2f}% PF={stress_pf:.2f}"
        )

    for row in ranked:
        row["final_score"] = _finalize_score(row)
    ranked = sorted(ranked, key=lambda r: float(r["final_score"]), reverse=True)
    champion = ranked[0]
    baseline = next((row for row in ranked if row["label"] == "BASELINE"), None)
    print(
        "\n[champion] "
        f"{champion['label']} | 10Y CAGR={champion['full_cagr_pct']:.2f}% "
        f"DD={champion['full_max_drawdown_pct']:.2f}% | Bear CAGR={champion['bear_cagr_pct']:.2f}% "
        f"| FinalScore={champion['final_score']:.2f}"
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exports_dir = ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    report_path = exports_dir / f"alpha_resilience_10y_{stamp}.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {"start": args.start, "end": args.end},
        "subwindows": {
            "early": {"start": early_start, "end": early_end},
            "bear": {"start": bear_start, "end": bear_end},
            "recent": {"start": recent_start, "end": recent_end},
        },
        "strategy_name": args.strategy_name,
        "universe_source": ctx.universe_source,
        "membership_source": ctx.membership_source,
        "symbols_requested": ctx.symbols_requested,
        "symbols_loaded": ctx.symbols_loaded,
        "champion": champion,
        "baseline": baseline,
        "ranked": ranked,
    }

    if args.apply_best:
        should_apply, reason = baseline_promotion_status(
            champion,
            baseline,
            score_key="final_score",
            max_drawdown_gate_pct=30.0,
        )
        promotion: dict[str, Any] = {
            "baseline_gate_passed": bool(should_apply),
            "reason": reason,
        }
        if should_apply and not args.skip_walkforward_gate:
            candidate_cfg = copy.deepcopy(base_cfg)
            candidate_cfg.update(champion["tweaks"])
            print(
                f"[walkforward] validating {champion['label']} "
                f"{args.start} -> {args.end} frictions={args.walkforward_frictions}"
            )
            wf_ok, wf_reason, wf_summary, wf_report = promotion_walkforward_status(
                candidate_cfg,
                start_date=str(args.start),
                end_date=str(args.end),
                transaction_cost_bps=_safe_float(candidate_cfg.get("transaction_cost_bps"), 2.0),
                friction_values=_parse_friction_values(args.walkforward_frictions),
                config_path=f"AlphaRes10Y::{champion['label']}",
            )
            wf_path = exports_dir / f"alpha_resilience_10y_walkforward_{stamp}.json"
            wf_path.write_text(json.dumps(wf_report, indent=2), encoding="utf-8")
            promotion["walkforward"] = {
                "ok": bool(wf_ok),
                "reason": wf_reason,
                "summary": wf_summary,
                "report_path": str(wf_path),
            }
            print(
                f"[walkforward] 36/12={wf_summary['stitched_36_12_cagr_pct']:.2f}% "
                f"60/12={wf_summary['stitched_60_12_cagr_pct']:.2f}% "
                f"DD={wf_summary['full_max_dd_pct']:.2f}%"
            )
            if not wf_ok:
                should_apply = False
                reason = wf_reason
        elif should_apply:
            promotion["walkforward"] = {"skipped": True}
        if should_apply:
            cfgs[target_idx].update(champion["tweaks"])
            generated_path.write_text(json.dumps(cfgs, indent=2), encoding="utf-8")
            print(f"[apply] Updated {args.strategy_name} in config/generated_strategies.json")
        else:
            print(f"[apply] Skipped promotion: {reason}")
        promotion["applied"] = bool(should_apply)
        promotion["reason"] = reason
        report["promotion"] = promotion

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[saved] {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
