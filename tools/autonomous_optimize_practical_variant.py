#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy


@dataclass
class WindowContext:
    label: str
    years: int
    start_date: str
    end_date: str
    symbols_requested: int
    symbols_loaded: int
    coverage: float
    universe_source: str
    membership_source: str
    prepared: Any
    global_data: Dict[str, pd.DataFrame]
    membership_by_day: Optional[List[Optional[frozenset[str]]]]


def _today_utc() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def _iso_date(ts: pd.Timestamp) -> str:
    return ts.date().isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Autonomous optimizer for Superperformance practical no-leverage variant "
            "(Russell 3000 PIT, next-day execution only)."
        )
    )
    parser.add_argument(
        "--base-config",
        default="config/superperformance_practical_no_leverage.json",
        help="Base practical config JSON.",
    )
    parser.add_argument(
        "--output-config",
        default="config/superperformance_practical_no_leverage_optimized.json",
        help="Where to write the optimized practical config.",
    )
    parser.add_argument(
        "--generated-config",
        default="config/generated_strategies.json",
        help="Generated strategy list to sync practical entry into.",
    )
    parser.add_argument(
        "--search-years",
        type=int,
        default=5,
        help="Primary optimization window in years.",
    )
    parser.add_argument(
        "--validate-years",
        type=int,
        default=10,
        help="Secondary validation window in years.",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=28,
        help="Total candidates to evaluate in search window (including seeds).",
    )
    parser.add_argument(
        "--finalists",
        type=int,
        default=6,
        help="Top search candidates to validate on long window.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Optional deterministic symbol cap for faster experiments (0 = full PIT window).",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=8,
        help="Data fetch worker count (cache mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--target-cagr",
        type=float,
        default=35.0,
        help="Early-stop target 10Y CAGR (percent) for realistic practical profile.",
    )
    return parser.parse_args()


def _resolve_window(years: int) -> Tuple[str, str]:
    end_ts = _today_utc()
    start_ts = (end_ts - pd.DateOffset(years=max(1, int(years)))).normalize()
    return _iso_date(start_ts), _iso_date(end_ts)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _compute_profit_factor(trades_list: Sequence[Dict[str, Any]]) -> float:
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
    if gross_losses <= 0:
        return 3.0 if gross_wins > 0 else 0.0
    return gross_wins / gross_losses


def _build_window_context(
    *,
    label: str,
    years: int,
    max_symbols: int,
    fetch_workers: int,
) -> WindowContext:
    start_date, end_date = _resolve_window(years)
    symbols, universe_source = get_universe_symbols_pit_window_with_meta(
        "RUSSELL3000",
        start_date,
        end_date,
    )
    if universe_source == "fallback_current":
        raise RuntimeError(
            "PIT Russell 3000 window data is required but unavailable. "
            "Set RUSSELL3000_PIT_DIR or RUSSELL3000_PIT_MEMBERSHIP_CSV."
        )
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if max_symbols > 0:
        symbols = symbols[: int(max_symbols)]
    if not symbols:
        raise RuntimeError(f"{label}: no symbols resolved for PIT window.")

    days = max(int(years * 365.25), 365) + 380
    print(
        f"\n[{label}] Loading Russell 3000 PIT window: {start_date} -> {end_date} | "
        f"symbols={len(symbols)} | source={universe_source}"
    )
    data = fetch_data_pack(
        symbols,
        days=days,
        max_workers=max(1, int(fetch_workers)),
        backtest_mode=True,
    ) or {}
    if not data:
        raise RuntimeError(f"{label}: no price data loaded.")

    g_data_raw = fetch_data_pack(
        ["SPY", "$VIX", "VIX"],
        days=days,
        max_workers=max(1, min(int(fetch_workers), 4)),
        backtest_mode=True,
    ) or {}
    vix_df = g_data_raw.get("$VIX")
    if vix_df is None or getattr(vix_df, "empty", False):
        vix_df = g_data_raw.get("VIX")
    global_data = {
        "SPY": g_data_raw.get("SPY"),
        "VIX": vix_df,
    }

    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=start_date,
        global_data=global_data,
    )
    loaded_symbols = len(getattr(prepared, "enriched", {}) or {})
    coverage = (loaded_symbols / float(len(symbols))) if symbols else 0.0

    all_dates_raw = getattr(prepared, "all_dates", [])
    all_dates = list(all_dates_raw) if all_dates_raw is not None else []
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates)
    if not membership_by_day:
        membership_by_day = None
        membership_source = "none"

    print(
        f"[{label}] Prepared symbols={loaded_symbols}/{len(symbols)} ({coverage:.1%}) | "
        f"membership={membership_source}"
    )

    return WindowContext(
        label=label,
        years=int(years),
        start_date=start_date,
        end_date=end_date,
        symbols_requested=len(symbols),
        symbols_loaded=loaded_symbols,
        coverage=float(coverage),
        universe_source=universe_source,
        membership_source=membership_source,
        prepared=prepared,
        global_data=global_data,
        membership_by_day=membership_by_day,
    )


def _enforce_practical_constraints(
    base_config: Dict[str, Any],
    tweaks: Optional[Dict[str, Any]] = None,
    *,
    candidate_name: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    for k, v in (tweaks or {}).items():
        cfg[k] = v

    cfg["name"] = candidate_name
    cfg["type"] = "superperformance"
    cfg["enabled_by_default"] = False

    # Practical EOD process: scan after close, execute next session only.
    cfg["signal_mode"] = "after_close"
    cfg["ep_entry_mode"] = "close"
    cfg["ep_force_next_day"] = True
    cfg["vcp_entry_mode"] = "next_day"
    cfg["market_exposure_mode"] = "exposure"
    cfg["use_market_regime_traffic_light"] = False
    cfg["bear_cash_mode"] = "off"
    cfg["log_regime_skips"] = False

    # No fees/commissions in this user's modeling preference.
    cfg["transaction_cost_bps"] = 0.0
    cfg["slippage_bps"] = 0.0
    cfg["entry_slippage_bps"] = 0.0
    cfg["exit_slippage_bps"] = 0.0

    # Hard no-leverage constraints.
    bull_exposure = min(1.0, max(0.20, _safe_float(cfg.get("max_total_exposure_pct_bull", 1.0), 1.0)))
    bear_exposure = min(1.0, max(0.10, _safe_float(cfg.get("max_total_exposure_pct_bear", 1.0), 1.0)))
    max_positions = max(1, int(_safe_float(cfg.get("max_positions", 5), 5)))
    max_pos_size = max(0.05, min(0.35, _safe_float(cfg.get("max_pos_size_pct", 0.2), 0.2)))
    max_pos_size = min(max_pos_size, bull_exposure / float(max_positions))

    cfg["max_total_exposure_pct_bull"] = float(bull_exposure)
    cfg["max_total_exposure_pct_bear"] = float(bear_exposure)
    cfg["max_positions"] = int(max_positions)
    cfg["max_pos_size_pct"] = float(max_pos_size)
    cfg["allow_margin"] = False

    # Keep entry type caps coherent with account-level cap.
    cfg["vcp_max_pos_size_pct"] = float(min(max_pos_size, 0.30))
    cfg["ep_max_pos_size_pct"] = float(min(max_pos_size, 0.25))

    return cfg


def _score_candidate(metrics: Dict[str, Any]) -> float:
    cagr = _safe_float(metrics.get("cagr_pct"), -100.0)
    max_dd = _safe_float(metrics.get("max_drawdown_pct"), 100.0)
    pf = _safe_float(metrics.get("profit_factor"), 0.0)
    trades = int(_safe_float(metrics.get("total_trades"), 0))
    cagr_3y = _safe_float(metrics.get("cagr_3y_pct"), float("nan"))
    cagr_5y = _safe_float(metrics.get("cagr_5y_pct"), float("nan"))
    same_day_open = int(_safe_float(metrics.get("same_day_open_entries"), 0))
    max_gross = _safe_float(metrics.get("max_gross_exposure_pct"), 0.0)

    # Hard practicality gates.
    if same_day_open > 0:
        return -1_000_000.0 - float(same_day_open)
    if max_gross > 1.001:
        return -900_000.0 - (max_gross * 1000.0)

    score = cagr
    score += min(pf, 3.0) * 2.5
    score -= max(0.0, max_dd - 25.0) * 0.7
    score -= max(0, 80 - trades) * 0.08
    if np.isfinite(cagr_5y):
        score += cagr_5y * 0.20
    if np.isfinite(cagr_3y):
        score += cagr_3y * 0.10
    return float(score)


def _recent_cagr_pct(equity_curve: Sequence[Dict[str, Any]], years: int) -> float:
    if not equity_curve:
        return float("nan")
    ec = pd.DataFrame(equity_curve)
    if ec.empty or "Date" not in ec.columns or "Equity" not in ec.columns:
        return float("nan")
    ec["Date"] = pd.to_datetime(ec["Date"], errors="coerce")
    ec["Equity"] = pd.to_numeric(ec["Equity"], errors="coerce")
    ec = ec.dropna(subset=["Date", "Equity"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if len(ec) < 2:
        return float("nan")
    end_dt = ec["Date"].iloc[-1]
    cutoff = end_dt - pd.DateOffset(years=max(1, int(years)))
    leading = ec[ec["Date"] < cutoff].tail(1)
    window = pd.concat([leading, ec[ec["Date"] >= cutoff]], ignore_index=True)
    if len(window) < 2:
        return float("nan")
    start_eq = _safe_float(window["Equity"].iloc[0], 0.0)
    end_eq = _safe_float(window["Equity"].iloc[-1], 0.0)
    if start_eq <= 0 or end_eq <= 0:
        return float("nan")
    span_years = (window["Date"].iloc[-1] - window["Date"].iloc[0]).days / 365.25
    if span_years <= 0:
        return float("nan")
    return float(((end_eq / start_eq) ** (1.0 / span_years) - 1.0) * 100.0)


def _evaluate(
    *,
    context: WindowContext,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = SuperperformanceStrategy(config)
    result = run_backtest(
        strategy,
        context.prepared,
        start_cash=100000.0,
        start_date=context.start_date,
        end_date=context.end_date,
        global_data=context.global_data,
        universe_membership_by_day=context.membership_by_day,
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"{context.label}: unexpected backtest result type {type(result)}")

    final_value = _safe_float(result.get("final_value"), 0.0)
    cagr_pct = _safe_float(result.get("cagr"), 0.0) * 100.0
    max_dd_raw = _safe_float(result.get("max_drawdown_pct"), 0.0)
    max_dd_pct = max_dd_raw * 100.0 if max_dd_raw <= 1.0 else max_dd_raw
    trades = int(_safe_float(result.get("total_trades"), 0))
    hit_rate = _safe_float(result.get("hit_rate"), 0.0)
    pf = _compute_profit_factor(result.get("trades_list") or [])
    cagr_5y = _recent_cagr_pct(result.get("equity_curve") or [], 5)
    cagr_3y = _recent_cagr_pct(result.get("equity_curve") or [], 3)
    audit = result.get("audit_report") if isinstance(result.get("audit_report"), dict) else {}
    same_day_open = int(_safe_float(audit.get("same_day_open_entries"), 0))
    stale_days = int(_safe_float(audit.get("stale_position_days"), 0))
    max_gross = _safe_float(audit.get("max_gross_exposure_pct"), 0.0)

    metrics = {
        "final_value": final_value,
        "cagr_pct": cagr_pct,
        "max_drawdown_pct": max_dd_pct,
        "total_trades": trades,
        "hit_rate_pct": hit_rate,
        "profit_factor": pf,
        "cagr_5y_pct": cagr_5y,
        "cagr_3y_pct": cagr_3y,
        "same_day_open_entries": same_day_open,
        "stale_position_days": stale_days,
        "max_gross_exposure_pct": max_gross,
    }
    metrics["score"] = _score_candidate(metrics)
    return metrics


def _candidate_space() -> Dict[str, List[Any]]:
    return {
        "rs_gate_min": [70, 75, 80, 85],
        "min_price": [4, 6, 8],
        "min_avg_volume_30": [25000, 50000, 100000],
        "fundamental_growth_min_pct": [0, 5, 10],
        "high_tight_flag_override_pct": [85, 90, 95],
        "vcp_lookback_bars": [30, 40, 60],
        "vcp_extrema_order": [2, 3],
        "vcp_breakout_volume_mult": [1.25, 1.5, 2.0],
        "breakout_buffer": [0.0, 0.0005, 0.001],
        "ep_gap_pct": [4, 6, 8],
        "ep_vol_mult": [1.5, 2.0, 3.0],
        "ep_close_near_high_min": [0.55, 0.65, 0.75],
        "ep_max_stop_pct": [0.10, 0.12, 0.15],
        "technical_weight": [0.50, 0.625, 0.75, 0.85],
        "fundamental_weight": [0.15, 0.25, 0.375, 0.50],
        "min_entry_score": [15, 20, 25, 30, 35],
        "max_stop_pct": [0.05, 0.06, 0.08],
        "stop_limit_pct": [0.02, 0.03],
        "stop_loss_atr_bull": [3, 4, 5, 6],
        "stop_loss_atr_bear": [0.5, 0.75, 1.0],
        "time_stop_days": [3, 4, 5, 7],
        "pyramid_threshold": [0.06, 0.08, 0.10, 0.12],
        "pyramid_fraction": [0.33, 0.50],
        "pyramid_max_adds": [1, 2],
        "max_positions": [4, 5, 6, 7, 8],
        "risk_per_trade": [0.05, 0.07, 0.10, 0.12],
        "max_pos_size_pct": [0.15, 0.20, 0.25, 0.30],
        "max_total_exposure_pct_bear": [0.6, 0.8, 1.0],
    }


def _random_candidate(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {k: rng.choice(v) for k, v in space.items()}


def _config_signature(cfg: Dict[str, Any]) -> str:
    # Ignore cosmetic display keys so search dedupes by effective behavior.
    normalized = {k: v for k, v in cfg.items() if k not in {"name", "enabled_by_default"}}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _seed_candidates() -> List[Dict[str, Any]]:
    return [
        {},
        {"max_positions": 5, "max_pos_size_pct": 0.20, "max_total_exposure_pct_bear": 1.0},
        {"max_positions": 5, "risk_per_trade": 0.07, "max_pos_size_pct": 0.20, "max_total_exposure_pct_bear": 0.8},
        {"max_positions": 4, "risk_per_trade": 0.10, "max_pos_size_pct": 0.25, "max_total_exposure_pct_bear": 1.0},
        {"max_positions": 6, "risk_per_trade": 0.05, "max_pos_size_pct": 0.15, "max_total_exposure_pct_bear": 0.5},
    ]


def _sync_generated_practical(
    generated_path: Path,
    champion_cfg: Dict[str, Any],
) -> None:
    payload = json.loads(generated_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"{generated_path} must contain a JSON list.")
    target_idx = None
    for i, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "")
        typ = str(item.get("type", "") or "").lower()
        if typ == "superperformance" and "practical eod no leverage" in name.lower():
            target_idx = i
            break
    if target_idx is None:
        payload.append(copy.deepcopy(champion_cfg))
    else:
        payload[target_idx] = copy.deepcopy(champion_cfg)
    generated_path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    rng = random.Random(int(args.seed))

    base_path = Path(args.base_config)
    output_path = Path(args.output_config)
    generated_path = Path(args.generated_config)

    base_cfg = json.loads(base_path.read_text(encoding="utf-8"))
    if not isinstance(base_cfg, dict):
        raise RuntimeError(f"{base_path} must be a JSON object.")

    t0 = time.time()
    search_ctx = _build_window_context(
        label="SEARCH",
        years=int(args.search_years),
        max_symbols=int(args.max_symbols),
        fetch_workers=int(args.fetch_workers),
    )

    candidate_space = _candidate_space()
    seeds = _seed_candidates()
    target_evals = max(int(args.max_evals), len(seeds))
    eval_rows: List[Dict[str, Any]] = []

    print(
        f"\n[SEARCH] Evaluating {target_evals} candidates "
        f"(seed={args.seed}, years={args.search_years}, no-leverage hard constraints enabled)"
    )
    best_row: Optional[Dict[str, Any]] = None
    seen_signatures: set[str] = set()
    i = 0
    attempts = 0
    seed_idx = 0
    max_attempts = max(target_evals * 30, 200)
    while i < target_evals and attempts < max_attempts:
        attempts += 1
        if seed_idx < len(seeds):
            tweaks = seeds[seed_idx]
            seed_idx += 1
        else:
            tweaks = _random_candidate(candidate_space, rng)
        cfg = _enforce_practical_constraints(
            base_cfg,
            tweaks,
            candidate_name=f"PracticalCandidate_{i+1:03d}",
        )
        sig = _config_signature(cfg)
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        metrics = _evaluate(context=search_ctx, config=cfg)
        row = {
            "candidate_index": i + 1,
            "window": search_ctx.label,
            "tweaks": tweaks,
            "config": cfg,
            **metrics,
        }
        eval_rows.append(row)
        if best_row is None or float(row["score"]) > float(best_row["score"]):
            best_row = row

        print(
            f"[SEARCH {i+1:03d}/{target_evals}] "
            f"CAGR={row['cagr_pct']:.2f}% | DD={row['max_drawdown_pct']:.2f}% | "
            f"Trades={row['total_trades']} | PF={row['profit_factor']:.2f} | "
            f"SameDayOpen={row['same_day_open_entries']} | MaxGross={row['max_gross_exposure_pct']:.2%} | "
            f"Score={row['score']:.2f}"
        )
        gc.collect()
        i += 1

    if i < target_evals:
        print(
            f"⚠️ Search dedupe exhausted candidate space before target evals: "
            f"evaluated={i}/{target_evals}, attempts={attempts}/{max_attempts}"
        )

    eval_rows_sorted = sorted(eval_rows, key=lambda r: float(r["score"]), reverse=True)
    finalists_n = max(1, min(int(args.finalists), len(eval_rows_sorted)))
    finalists = eval_rows_sorted[:finalists_n]

    print(
        f"\n[VALIDATE] Running {finalists_n} finalists on {args.validate_years}Y Russell 3000 PIT window..."
    )
    validate_ctx = _build_window_context(
        label="VALIDATE",
        years=int(args.validate_years),
        max_symbols=int(args.max_symbols),
        fetch_workers=max(1, int(args.fetch_workers) - 2),
    )

    validated_rows: List[Dict[str, Any]] = []
    champion: Optional[Dict[str, Any]] = None
    for rank, row in enumerate(finalists, start=1):
        cfg = copy.deepcopy(row["config"])
        cfg["name"] = f"PracticalFinalist_{rank:02d}"
        m10 = _evaluate(context=validate_ctx, config=cfg)

        combined_score = (
            (0.55 * _safe_float(m10.get("cagr_pct"), 0.0))
            + (0.45 * _safe_float(row.get("cagr_pct"), 0.0))
            + (0.10 * _safe_float(m10.get("profit_factor"), 0.0))
            - (0.15 * max(0.0, _safe_float(m10.get("max_drawdown_pct"), 0.0) - 30.0))
        )
        if int(m10.get("same_day_open_entries", 0)) > 0 or _safe_float(m10.get("max_gross_exposure_pct"), 0.0) > 1.001:
            combined_score = -1_000_000.0

        out = {
            "rank_from_search": rank,
            "search_metrics": {
                k: row[k]
                for k in [
                    "cagr_pct",
                    "max_drawdown_pct",
                    "total_trades",
                    "profit_factor",
                    "same_day_open_entries",
                    "max_gross_exposure_pct",
                    "score",
                ]
            },
            "validate_metrics": m10,
            "combined_score": float(combined_score),
            "config": cfg,
        }
        validated_rows.append(out)

        print(
            f"[VALIDATE {rank:02d}/{finalists_n}] "
            f"10Y CAGR={m10['cagr_pct']:.2f}% | DD={m10['max_drawdown_pct']:.2f}% | "
            f"Trades={m10['total_trades']} | PF={m10['profit_factor']:.2f} | "
            f"SameDayOpen={m10['same_day_open_entries']} | MaxGross={m10['max_gross_exposure_pct']:.2%} | "
            f"Combined={combined_score:.2f}"
        )

        if champion is None or float(out["combined_score"]) > float(champion["combined_score"]):
            champion = out
        gc.collect()

    if champion is None:
        raise RuntimeError("No champion candidate produced.")

    champion_cfg = copy.deepcopy(champion["config"])
    champion_cfg["name"] = "Superperformance (Practical EOD No Leverage)"
    champion_cfg["type"] = "superperformance"
    champion_cfg["enabled_by_default"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(champion_cfg, indent=4), encoding="utf-8")
    _sync_generated_practical(generated_path, champion_cfg)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = ROOT / "exports" / f"practical_no_leverage_optimization_{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "search_window": {
            "years": search_ctx.years,
            "start_date": search_ctx.start_date,
            "end_date": search_ctx.end_date,
            "symbols_requested": search_ctx.symbols_requested,
            "symbols_loaded": search_ctx.symbols_loaded,
            "coverage": search_ctx.coverage,
            "universe_source": search_ctx.universe_source,
            "membership_source": search_ctx.membership_source,
        },
        "validate_window": {
            "years": validate_ctx.years,
            "start_date": validate_ctx.start_date,
            "end_date": validate_ctx.end_date,
            "symbols_requested": validate_ctx.symbols_requested,
            "symbols_loaded": validate_ctx.symbols_loaded,
            "coverage": validate_ctx.coverage,
            "universe_source": validate_ctx.universe_source,
            "membership_source": validate_ctx.membership_source,
        },
        "args": {
            "max_evals": int(args.max_evals),
            "finalists": int(args.finalists),
            "max_symbols": int(args.max_symbols),
            "fetch_workers": int(args.fetch_workers),
            "seed": int(args.seed),
            "target_cagr": float(args.target_cagr),
        },
        "champion": champion,
        "top_search": eval_rows_sorted[: min(10, len(eval_rows_sorted))],
        "top_validated": sorted(validated_rows, key=lambda r: float(r["combined_score"]), reverse=True),
        "output_config": str(output_path),
        "synced_generated_config": str(generated_path),
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    c10 = champion["validate_metrics"]
    c5 = champion["search_metrics"]
    print("\n✅ Practical no-leverage optimization complete")
    print(
        f"Champion 5Y: CAGR={_safe_float(c5.get('cagr_pct'), 0.0):.2f}% | "
        f"DD={_safe_float(c5.get('max_drawdown_pct'), 0.0):.2f}% | Trades={int(_safe_float(c5.get('total_trades'), 0))}"
    )
    print(
        f"Champion 10Y: CAGR={_safe_float(c10.get('cagr_pct'), 0.0):.2f}% | "
        f"DD={_safe_float(c10.get('max_drawdown_pct'), 0.0):.2f}% | Trades={int(_safe_float(c10.get('total_trades'), 0))}"
    )
    print(
        f"Audit: same_day_open={int(_safe_float(c10.get('same_day_open_entries'), 0))} | "
        f"max_gross={_safe_float(c10.get('max_gross_exposure_pct'), 0.0):.2%}"
    )
    print(f"Saved optimized config: {output_path}")
    print(f"Saved report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
