#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day, get_universe_symbols_pit_window_with_meta
from execution.engine import prepare_backtest_data
from scripts.run_factor_walkforward import _daily_membership_price_coverage, _pct_dd, _safe_float
from scripts.run_smid_pullback_walkforward import (
    _build_smid_pullback_scores,
    _extract_feature_arrays,
    _load_config,
    _run_window,
    _slice_prices,
)


DEFAULT_CONFIGS = [
    ROOT / "config" / "smid_pullback_broad_v1.json",
    ROOT / "config" / "smid_pullback_r3000_tb006_v1.json",
]
DEFAULT_MIN_UNIVERSE_COVERAGE = 0.60
DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE = 0.90
DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10 = 0.85


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMID pullback configs on a train/holdout split.")
    parser.add_argument("--configs", nargs="*", default=[str(p) for p in DEFAULT_CONFIGS])
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--train-start-date", default="2016-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3200)
    parser.add_argument("--min-universe-coverage", type=float, default=DEFAULT_MIN_UNIVERSE_COVERAGE)
    parser.add_argument("--min-daily-membership-coverage", type=float, default=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE)
    parser.add_argument("--min-daily-membership-coverage-p10", type=float, default=DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10)
    parser.add_argument("--friction-multiplier", type=float, default=1.0)
    parser.add_argument("--extra-transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--extra-entry-slippage-bps", type=float, default=0.0)
    parser.add_argument("--extra-exit-slippage-bps", type=float, default=0.0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _feature_config_for_paths(config_paths: Sequence[str | Path] | None) -> Dict[str, Any]:
    modes: set[str] = set()
    for cfg_path in config_paths or []:
        try:
            cfg = _load_config(cfg_path)
        except Exception:
            continue
        mode = str(cfg.get("price_filter_mode", "adjusted") or "adjusted").strip().lower()
        if mode:
            modes.add(mode)
    if any(mode in {"raw", "nominal", "unadjusted"} for mode in modes):
        return {"price_filter_mode": "nominal"}
    return {"price_filter_mode": "adjusted"}


def _window_metrics(run: Dict[str, Any]) -> Dict[str, float | int]:
    audit = dict(run.get("audit_report") or {})
    cagr_pct = float(_safe_float(run.get("cagr"), 0.0) * 100.0)
    max_dd_pct = float(_pct_dd(run.get("max_drawdown_pct", 0.0)))
    return {
        "cagr_pct": cagr_pct,
        "max_dd_pct": max_dd_pct,
        "calmar": float(cagr_pct / max_dd_pct) if max_dd_pct > 0 else float("nan"),
        "avg_annual_turnover_pct": float(_safe_float(run.get("avg_annual_turnover_pct"), float("nan"))),
        "total_trades": int(run.get("total_trades", 0) or 0),
        "final_value": float(_safe_float(run.get("final_value"), 0.0)),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
    }


def _resolve_symbols(universe: str, start_date: str, end_date: str) -> Tuple[List[str], str]:
    name = str(universe).strip().upper()
    if name != "RUSSELL3000":
        raise ValueError("Holdout evaluator currently supports only RUSSELL3000.")
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    return list(symbols), str(source)


def _coverage_ratio(requested_symbols: Sequence[str], loaded_symbols: Sequence[str]) -> float:
    requested = int(len(requested_symbols))
    if requested <= 0:
        return 0.0
    return float(len(loaded_symbols)) / float(requested)


def _coverage_gate_failures(
    *,
    coverage_ratio: float,
    coverage_stats: Mapping[str, Any],
    min_universe_coverage: float,
    min_daily_membership_coverage: float,
    min_daily_membership_coverage_p10: float,
) -> List[str]:
    failures: List[str] = []
    if float(coverage_ratio) < float(min_universe_coverage):
        failures.append(
            f"universe_coverage {float(coverage_ratio):.1%} < {float(min_universe_coverage):.1%}"
        )

    mean_daily_coverage = _safe_float(coverage_stats.get("mean"), float("nan"))
    if not np.isfinite(mean_daily_coverage) or float(mean_daily_coverage) < float(min_daily_membership_coverage):
        failures.append(
            f"daily_membership_coverage_mean {float(mean_daily_coverage):.1%} < {float(min_daily_membership_coverage):.1%}"
        )

    p10_daily_coverage = _safe_float(coverage_stats.get("p10"), float("nan"))
    if not np.isfinite(p10_daily_coverage) or float(p10_daily_coverage) < float(min_daily_membership_coverage_p10):
        failures.append(
            f"daily_membership_coverage_p10 {float(p10_daily_coverage):.1%} < {float(min_daily_membership_coverage_p10):.1%}"
        )
    return failures


def _apply_friction_stress(
    cfg: Mapping[str, Any],
    *,
    friction_multiplier: float = 1.0,
    extra_transaction_cost_bps: float = 0.0,
    extra_entry_slippage_bps: float = 0.0,
    extra_exit_slippage_bps: float = 0.0,
) -> Dict[str, Any]:
    out = copy.deepcopy(dict(cfg))
    friction = dict(out.get("friction") or {})
    mult = max(0.0, float(friction_multiplier or 0.0))
    friction["transaction_cost_bps"] = float(friction.get("transaction_cost_bps", 15.0) or 15.0) * mult + float(
        extra_transaction_cost_bps or 0.0
    )
    friction["entry_slippage_bps"] = float(friction.get("entry_slippage_bps", 10.0) or 10.0) * mult + float(
        extra_entry_slippage_bps or 0.0
    )
    friction["exit_slippage_bps"] = float(friction.get("exit_slippage_bps", 10.0) or 10.0) * mult + float(
        extra_exit_slippage_bps or 0.0
    )
    out["friction"] = friction
    return out


def build_smid_pullback_context(
    *,
    universe: str,
    start_date: str,
    end_date: str,
    days: int,
    config_paths: Sequence[str | Path] | None = None,
) -> Dict[str, Any]:
    requested_symbols, symbol_source = _resolve_symbols(universe, start_date, end_date)
    quality_report: Dict[str, Any] = {}
    data = fetch_data_pack(
        requested_symbols,
        days=int(days),
        backtest_mode=True,
        quality_report=quality_report,
    ) or {}
    loaded_symbols = sorted(data.keys())
    global_data = fetch_data_pack(["SPY", "VIX", "HYG", "LQD"], days=int(days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, loaded_symbols, start_date=start_date, global_data=global_data)
    membership_by_day, membership_source = build_russell3000_membership_by_day(
        list(prepared.all_dates),
        allow_missing_days=False,
    )
    if not membership_by_day or len(membership_by_day) != len(prepared.all_dates):
        raise RuntimeError(f"Failed to build Russell 3000 PIT membership timeline (source={membership_source}).")
    feature_cfg = _feature_config_for_paths(config_paths)
    features = _extract_feature_arrays(prepared, membership_by_day=membership_by_day, cfg=feature_cfg)
    coverage_stats = _daily_membership_price_coverage(features)
    return {
        "requested_symbols": list(requested_symbols),
        "symbol_source": str(symbol_source),
        "data": data,
        "loaded_symbols": list(loaded_symbols),
        "global_data": global_data,
        "prepared": prepared,
        "membership_source": str(membership_source),
        "feature_cfg": dict(feature_cfg),
        "features": features,
        "coverage_stats": coverage_stats,
        "quality_report": quality_report,
    }


def evaluate_smid_pullback_configs_on_context(
    context: Mapping[str, Any],
    *,
    config_paths: Sequence[str | Path],
    universe: str,
    train_start_date: str,
    train_end_date: str,
    holdout_start_date: str,
    holdout_end_date: str,
    min_universe_coverage: float = DEFAULT_MIN_UNIVERSE_COVERAGE,
    min_daily_membership_coverage: float = DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
    min_daily_membership_coverage_p10: float = DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
    friction_multiplier: float = 1.0,
    extra_transaction_cost_bps: float = 0.0,
    extra_entry_slippage_bps: float = 0.0,
    extra_exit_slippage_bps: float = 0.0,
    raise_on_coverage_fail: bool = True,
    include_runs: bool = False,
) -> Dict[str, Any]:
    requested_symbols = list(context.get("requested_symbols") or [])
    loaded_symbols = list(context.get("loaded_symbols") or [])
    prepared = context.get("prepared")
    global_data = dict(context.get("global_data") or {})
    features = context.get("features")
    coverage_stats = dict(context.get("coverage_stats") or {})
    nominal_price_proxy_symbols = int(context.get("features", {}).get("nominal_price_proxy_symbols", 0) or 0)
    nominal_price_proxy_missing_symbols = int(
        context.get("features", {}).get("nominal_price_proxy_missing_symbols", 0) or 0
    )
    quality_report = dict(context.get("quality_report") or {})
    symbol_source = str(context.get("symbol_source", ""))
    membership_source = str(context.get("membership_source", ""))

    coverage_ratio = _coverage_ratio(requested_symbols, loaded_symbols)
    coverage_gate_failures = _coverage_gate_failures(
        coverage_ratio=coverage_ratio,
        coverage_stats=coverage_stats,
        min_universe_coverage=float(min_universe_coverage),
        min_daily_membership_coverage=float(min_daily_membership_coverage),
        min_daily_membership_coverage_p10=float(min_daily_membership_coverage_p10),
    )
    if coverage_gate_failures and raise_on_coverage_fail:
        raise RuntimeError(
            "SMID pullback coverage gate failed: " + "; ".join(str(item) for item in coverage_gate_failures)
        )

    reports: List[Dict[str, Any]] = []
    if not coverage_gate_failures:
        for cfg_path in config_paths:
            base_cfg = _load_config(cfg_path)
            cfg = _apply_friction_stress(
                base_cfg,
                friction_multiplier=float(friction_multiplier),
                extra_transaction_cost_bps=float(extra_transaction_cost_bps),
                extra_entry_slippage_bps=float(extra_entry_slippage_bps),
                extra_exit_slippage_bps=float(extra_exit_slippage_bps),
            )
            scores = _build_smid_pullback_scores(features, cfg)
            if scores.empty:
                reports.append(
                    {
                        "config": str(cfg_path),
                        "strategy_name": str(cfg.get("name", "") or Path(cfg_path).stem),
                        "error": "no_scores",
                    }
                )
                continue

            price_frames = _slice_prices(features, ["open", "close"], list(scores.columns))
            train_run = _run_window(
                cfg=cfg,
                prices_close=price_frames["close"],
                prices_open=price_frames["open"],
                scores=scores,
                global_data=global_data,
                start_date=train_start_date,
                end_date=train_end_date,
            )
            holdout_run = _run_window(
                cfg=cfg,
                prices_close=price_frames["close"],
                prices_open=price_frames["open"],
                scores=scores,
                global_data=global_data,
                start_date=holdout_start_date,
                end_date=holdout_end_date,
            )

            report: Dict[str, Any] = {
                "config": str(cfg_path),
                "strategy_name": str(cfg.get("name", "") or Path(cfg_path).stem),
                "requested_symbols": int(len(requested_symbols)),
                "loaded_symbols": int(len(loaded_symbols)),
                "prepared_symbols": int(len(getattr(prepared, "enriched", {}) or {})),
                "active_scored_symbols": int(len(scores.columns)),
                "coverage_ratio": float(coverage_ratio),
                "nominal_price_proxy_symbols": int(nominal_price_proxy_symbols),
                "nominal_price_proxy_missing_symbols": int(nominal_price_proxy_missing_symbols),
                "universe": str(universe),
                "universe_source": symbol_source,
                "membership_source": membership_source,
                "data_quality": {
                    "missing_symbols": int(quality_report.get("missing", 0) or 0),
                    "incomplete_history": int(quality_report.get("incomplete_history", 0) or 0),
                    "stale": int(quality_report.get("stale", 0) or 0),
                },
                "daily_membership_price_coverage": coverage_stats,
                "train": _window_metrics(train_run),
                "holdout": _window_metrics(holdout_run),
            }
            if include_runs:
                report["train_run"] = train_run
                report["holdout_run"] = holdout_run
            reports.append(report)

    return {
        "train_start_date": str(train_start_date),
        "train_end_date": str(train_end_date),
        "holdout_start_date": str(holdout_start_date),
        "holdout_end_date": str(holdout_end_date),
        "coverage_gate_passed": not bool(coverage_gate_failures),
        "coverage_gate_failures": list(coverage_gate_failures),
        "min_universe_coverage": float(min_universe_coverage),
        "min_daily_membership_coverage": float(min_daily_membership_coverage),
        "min_daily_membership_coverage_p10": float(min_daily_membership_coverage_p10),
        "friction_stress": {
            "friction_multiplier": float(friction_multiplier),
            "extra_transaction_cost_bps": float(extra_transaction_cost_bps),
            "extra_entry_slippage_bps": float(extra_entry_slippage_bps),
            "extra_exit_slippage_bps": float(extra_exit_slippage_bps),
        },
        "reports": reports,
    }


def main() -> None:
    args = _parse_args()
    context = build_smid_pullback_context(
        universe=str(args.universe),
        start_date=str(args.train_start_date),
        end_date=str(args.holdout_end_date),
        days=int(args.days),
        config_paths=args.configs,
    )
    requested_symbols = list(context.get("requested_symbols") or [])
    loaded_symbols = list(context.get("loaded_symbols") or [])
    coverage_stats = dict(context.get("coverage_stats") or {})
    print(
        f"smid holdout eval: requested_symbols={len(requested_symbols)} "
        f"universe={args.universe} source={context.get('symbol_source', '')}"
    )
    print(f"loaded_symbols={len(loaded_symbols)}")
    if coverage_stats.get("mean") == coverage_stats.get("mean"):
        print(f"daily_pit_price_coverage_mean={float(coverage_stats['mean']):.1%}")

    payload = evaluate_smid_pullback_configs_on_context(
        context,
        config_paths=args.configs,
        universe=str(args.universe),
        train_start_date=str(args.train_start_date),
        train_end_date=str(args.train_end_date),
        holdout_start_date=str(args.holdout_start_date),
        holdout_end_date=str(args.holdout_end_date),
        min_universe_coverage=float(args.min_universe_coverage),
        min_daily_membership_coverage=float(args.min_daily_membership_coverage),
        min_daily_membership_coverage_p10=float(args.min_daily_membership_coverage_p10),
        friction_multiplier=float(args.friction_multiplier),
        extra_transaction_cost_bps=float(args.extra_transaction_cost_bps),
        extra_entry_slippage_bps=float(args.extra_entry_slippage_bps),
        extra_exit_slippage_bps=float(args.extra_exit_slippage_bps),
        raise_on_coverage_fail=True,
        include_runs=False,
    )
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
