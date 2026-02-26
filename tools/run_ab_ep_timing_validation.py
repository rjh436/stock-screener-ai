#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy


def _today_utc_date() -> pd.Timestamp:
    return pd.Timestamp.utcnow().tz_localize(None).normalize()


def _iso_date(value: pd.Timestamp) -> str:
    return value.date().isoformat()


def _make_strategy_payload(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    payload = copy.deepcopy(config)
    payload["name"] = name
    return payload


def _result_summary(res: Dict[str, Any]) -> Dict[str, Any]:
    audit = res.get("audit_report") if isinstance(res.get("audit_report"), dict) else {}
    return {
        "final_value": float(res.get("final_value", 0.0) or 0.0),
        "cagr": float(res.get("cagr", 0.0) or 0.0),
        "max_drawdown_pct": float(res.get("max_drawdown_pct", 0.0) or 0.0),
        "total_trades": int(res.get("total_trades", 0) or 0),
        "hit_rate": float(res.get("hit_rate", 0.0) or 0.0),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "stale_position_days": int(audit.get("stale_position_days", 0) or 0),
        "max_gross_exposure_pct": float(audit.get("max_gross_exposure_pct", 0.0) or 0.0),
    }


def _load_winner_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked A/B backtest: current EP timing vs EP next-day-safe timing",
    )
    parser.add_argument(
        "--config",
        default="config/superperformance_winner.json",
        help="Path to winner config JSON",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Backtest window in years",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=True,
        help="Use cached price history only (default: true).",
    )
    parser.add_argument(
        "--allow-refresh",
        action="store_true",
        help="Allow network refresh instead of cache-only mode.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path for JSON report.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=0,
        help="Optional deterministic cap on PIT symbols (0 = no cap).",
    )
    return parser.parse_args()


def _resolve_window(years: int) -> Tuple[str, str]:
    today = _today_utc_date()
    start = (today - pd.DateOffset(years=max(1, years))).normalize()
    return _iso_date(start), _iso_date(today)


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    base_cfg = _load_winner_config(config_path)
    start_date, end_date = _resolve_window(args.years)

    require_pit = str(os.getenv("APEX_REQUIRE_PIT_UNIVERSE", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    symbols, universe_source = get_universe_symbols_pit_window_with_meta(
        "RUSSELL3000",
        start_date,
        end_date,
    )
    if require_pit and universe_source == "fallback_current":
        raise RuntimeError(
            "PIT universe required, but Russell 3000 PIT data was not found. "
            "Set RUSSELL3000_PIT_DIR or RUSSELL3000_PIT_MEMBERSHIP_CSV."
        )
    if not symbols:
        raise RuntimeError("No symbols loaded for Russell 3000 PIT universe.")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if int(args.max_symbols or 0) > 0:
        symbols = symbols[: int(args.max_symbols)]
        if not symbols:
            raise RuntimeError("No symbols remain after applying --max-symbols cap.")

    use_cache_only = bool(args.cache_only and not args.allow_refresh)
    years = max(1, int(args.years))
    fetch_days = max(int(years * 365.25), 365) + 320

    data = fetch_data_pack(
        symbols,
        days=fetch_days,
        backtest_mode=use_cache_only,
    ) or {}
    if not data:
        raise RuntimeError("No price data loaded for requested symbols.")

    g_data = fetch_data_pack(
        ["SPY", "$VIX", "VIX"],
        days=fetch_days,
        backtest_mode=use_cache_only,
    ) or {}
    vix_df = g_data.get("$VIX")
    if vix_df is None or getattr(vix_df, "empty", False):
        vix_df = g_data.get("VIX")
    global_data = {"SPY": g_data.get("SPY"), "VIX": vix_df}

    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=start_date,
        global_data=global_data,
    )
    if prepared is None or not getattr(prepared, "enriched", None):
        raise RuntimeError("Prepared dataset is empty.")

    prepared_dates = getattr(prepared, "all_dates", None)
    prepared_dates_seq = list(prepared_dates) if prepared_dates is not None else []
    membership_by_day, membership_source = build_russell3000_membership_by_day(prepared_dates_seq)
    if not membership_by_day:
        membership_by_day = None
        membership_source = "none"

    baseline_cfg = _make_strategy_payload(base_cfg, "Superperformance_AB_Baseline")
    safe_cfg = _make_strategy_payload(base_cfg, "Superperformance_AB_EP_NextDay")
    safe_cfg["ep_force_next_day"] = True

    baseline_strategy = SuperperformanceStrategy(baseline_cfg)
    safe_strategy = SuperperformanceStrategy(safe_cfg)

    baseline_res = run_backtest(
        baseline_strategy,
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership_by_day,
    )
    safe_res = run_backtest(
        safe_strategy,
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership_by_day,
    )

    baseline_summary = _result_summary(baseline_res)
    safe_summary = _result_summary(safe_res)
    diff = {
        "final_value_delta": safe_summary["final_value"] - baseline_summary["final_value"],
        "cagr_delta": safe_summary["cagr"] - baseline_summary["cagr"],
        "max_drawdown_delta": safe_summary["max_drawdown_pct"] - baseline_summary["max_drawdown_pct"],
        "total_trades_delta": safe_summary["total_trades"] - baseline_summary["total_trades"],
        "hit_rate_delta": safe_summary["hit_rate"] - baseline_summary["hit_rate"],
        "same_day_open_entries_delta": (
            safe_summary["same_day_open_entries"] - baseline_summary["same_day_open_entries"]
        ),
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": {
            "start_date": start_date,
            "end_date": end_date,
            "years": years,
        },
        "dataset": {
            "requested_symbols": int(len(symbols)),
            "loaded_symbols": int(len(getattr(prepared, "enriched", {}) or {})),
            "coverage": float(
                (len(getattr(prepared, "enriched", {}) or {}) / float(len(symbols))) if symbols else 0.0
            ),
            "universe_source": universe_source,
            "membership_timeline_source": membership_source,
            "cache_only": bool(use_cache_only),
        },
        "config": {
            "path": str(config_path),
            "sha1": hashlib.sha1(
                json.dumps(base_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "fees_or_slippage_changed": False,
        },
        "baseline": baseline_summary,
        "ep_next_day_safe": safe_summary,
        "diff": diff,
    }

    output_path = Path(args.output) if args.output else Path(
        "exports"
    ) / f"ab_ep_timing_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Window: {start_date} -> {end_date} ({years}Y)")
    print(
        "Dataset: "
        f"{payload['dataset']['loaded_symbols']}/{payload['dataset']['requested_symbols']} "
        f"({payload['dataset']['coverage']:.1%}) | "
        f"universe={universe_source} | membership={membership_source} | "
        f"cache_only={use_cache_only}"
    )
    print(
        "Baseline: "
        f"final=${baseline_summary['final_value']:,.2f} "
        f"CAGR={baseline_summary['cagr']:.2%} "
        f"trades={baseline_summary['total_trades']} "
        f"same_day_open={baseline_summary['same_day_open_entries']}"
    )
    print(
        "EP Next-Day Safe: "
        f"final=${safe_summary['final_value']:,.2f} "
        f"CAGR={safe_summary['cagr']:.2%} "
        f"trades={safe_summary['total_trades']} "
        f"same_day_open={safe_summary['same_day_open_entries']}"
    )
    print(
        "Delta (safe - baseline): "
        f"final=${diff['final_value_delta']:,.2f} "
        f"CAGR={diff['cagr_delta']:+.2%} "
        f"trades={diff['total_trades_delta']:+d} "
        f"same_day_open={diff['same_day_open_entries_delta']:+d}"
    )
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
