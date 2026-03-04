#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy


DEFAULT_CONFIG = ROOT / "config" / "superperformance_multi_sleeve_allocator_v1.json"


@dataclass(frozen=True)
class FrictionScenario:
    name: str
    transaction_cost_bps: float
    entry_slippage_bps: float
    exit_slippage_bps: float


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _pct_dd(value: Any) -> float:
    dd = _safe_float(value, 0.0)
    if dd <= 1.0:
        dd *= 100.0
    return dd


def _days_for_range(start_date: str, end_date: str, warmup_days: int = 420) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span_days = max(1, int((end_ts - start_ts).days))
    return span_days + int(max(0, warmup_days))


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return cfg


def _enforce_cash_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["allow_margin"] = False
    out["max_total_exposure_pct_bull"] = min(1.0, max(0.0, _safe_float(out.get("max_total_exposure_pct_bull"), 1.0)))
    out["max_total_exposure_pct_bear"] = min(1.0, max(0.0, _safe_float(out.get("max_total_exposure_pct_bear"), 0.0)))
    out["max_pos_size_pct"] = min(1.0, max(0.0, _safe_float(out.get("max_pos_size_pct"), 0.2)))
    out["vcp_max_pos_size_pct"] = min(out["max_pos_size_pct"], max(0.0, _safe_float(out.get("vcp_max_pos_size_pct"), out["max_pos_size_pct"])))
    out["ep_max_pos_size_pct"] = min(out["max_pos_size_pct"], max(0.0, _safe_float(out.get("ep_max_pos_size_pct"), out["max_pos_size_pct"])))
    return out


def _apply_friction(cfg: Dict[str, Any], scenario: FrictionScenario) -> Dict[str, Any]:
    out = dict(cfg)
    out["transaction_cost_bps"] = float(max(0.0, scenario.transaction_cost_bps))
    out["entry_slippage_bps"] = float(max(0.0, scenario.entry_slippage_bps))
    out["exit_slippage_bps"] = float(max(0.0, scenario.exit_slippage_bps))
    out["slippage_bps"] = float((out["entry_slippage_bps"] + out["exit_slippage_bps"]) / 2.0)
    return out


def _run_window(
    prepared: Any,
    global_data: Dict[str, pd.DataFrame],
    membership_by_day: List[set[str]],
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    strat = SuperperformanceStrategy(config)
    result = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership_by_day,
        require_pit_membership=True,
    )
    out = result[0] if isinstance(result, list) else result
    return out if isinstance(out, dict) else {}


def _annualized_cagr_from_values(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return float("nan")
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


def _build_test_windows(start_date: str, end_date: str, train_months: int, test_months: int, step_months: int = 12) -> List[Tuple[str, str]]:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    windows: List[Tuple[str, str]] = []

    test_start = start_ts + pd.DateOffset(months=int(train_months))
    while True:
        test_end = test_start + pd.DateOffset(months=int(test_months))
        if test_end > end_ts:
            break
        windows.append((test_start.date().isoformat(), test_end.date().isoformat()))
        test_start = test_start + pd.DateOffset(months=int(step_months))
    return windows


def _stitch_test_windows(
    prepared: Any,
    global_data: Dict[str, pd.DataFrame],
    membership_by_day: List[set[str]],
    cfg: Dict[str, Any],
    windows: List[Tuple[str, str]],
    test_months: int,
) -> Dict[str, Any]:
    fold_returns = []
    fold_rows: List[Dict[str, Any]] = []

    for i, (w_start, w_end) in enumerate(windows, start=1):
        out = _run_window(prepared, global_data, membership_by_day, cfg, w_start, w_end)
        final_val = _safe_float(out.get("final_value"), 100000.0)
        ret = (final_val / 100000.0) - 1.0 if final_val > 0 else float("nan")
        dd = _pct_dd(out.get("max_drawdown_pct", 0.0))
        cagr = _safe_float(out.get("cagr"), float("nan"))
        if np.isfinite(cagr):
            cagr *= 100.0
        trades = int(out.get("total_trades", 0) or 0)
        fold_rows.append(
            {
                "fold": i,
                "start": w_start,
                "end": w_end,
                "return_pct": float(ret * 100.0) if np.isfinite(ret) else float("nan"),
                "cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
                "max_dd_pct": float(dd),
                "trades": trades,
            }
        )
        if np.isfinite(ret):
            fold_returns.append(float(ret))

    if not fold_returns:
        return {
            "folds": fold_rows,
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
        }

    compounded = 1.0
    for r in fold_returns:
        compounded *= (1.0 + r)
    total_test_years = float(len(fold_returns) * (float(test_months) / 12.0))
    stitched_cagr = _annualized_cagr_from_values(1.0, compounded, total_test_years)
    worst_fold = float(min(fold_returns) * 100.0)
    return {
        "folds": fold_rows,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "worst_fold_return_pct": worst_fold,
    }


def _acceptance_snapshot(full_res: Dict[str, Any], stitched_36_12: Dict[str, Any], stitched_60_12: Dict[str, Any]) -> Dict[str, Any]:
    max_gross = _safe_float(((full_res.get("audit_report") or {}).get("max_gross_exposure_pct")), 0.0)
    cagr = _safe_float(full_res.get("cagr"), float("nan"))
    if np.isfinite(cagr):
        cagr *= 100.0
    mdd = _pct_dd(full_res.get("max_drawdown_pct", 0.0))
    oos_36 = _safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan"))
    oos_60 = _safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan"))

    return {
        "cash_only_ok": bool(np.isfinite(max_gross) and max_gross <= 1.0001),
        "full_cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
        "full_max_dd_pct": float(mdd),
        "oos_36_12_cagr_pct": float(oos_36) if np.isfinite(oos_36) else float("nan"),
        "oos_60_12_cagr_pct": float(oos_60) if np.isfinite(oos_60) else float("nan"),
        "meets_min_robust_target": bool(np.isfinite(oos_36) and np.isfinite(oos_60) and min(oos_36, oos_60) >= 16.0),
        "meets_stretch_target": bool(np.isfinite(oos_36) and np.isfinite(oos_60) and min(oos_36, oos_60) >= 20.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cash-only Russell 3000 walk-forward robustness for Superperformance config")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to strategy JSON config")
    parser.add_argument("--start", default="2016-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.now().date().isoformat(), help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="", help="Output JSON report path")
    parser.add_argument("--universe-limit", type=int, default=0, help="Optional limit for quick local smoke tests")
    parser.add_argument(
        "--universe-limit-mode",
        choices=["sample", "first"],
        default="sample",
        help="When universe-limit is used: deterministic random sample (default) or first-N symbols.",
    )
    parser.add_argument(
        "--universe-sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic universe sampling when universe-limit-mode=sample.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Default transaction cost bps")
    parser.add_argument(
        "--frictions",
        default="5,10,20,35",
        help="Comma-separated slippage bps pairs (entry=exit), e.g. '5,10,20'",
    )
    args = parser.parse_args()

    start_date = str(args.start)
    end_date = str(args.end)
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = _enforce_cash_only(_load_config(cfg_path))

    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError("PIT Russell 3000 universe is required. Source was non-PIT.")
    if not symbols:
        raise RuntimeError("PIT Russell 3000 universe returned zero symbols.")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if args.universe_limit and args.universe_limit > 0:
        limit = int(args.universe_limit)
        if limit < len(symbols):
            if str(args.universe_limit_mode).lower() == "sample":
                rng = np.random.default_rng(int(args.universe_sample_seed))
                selected_idx = np.sort(rng.choice(len(symbols), size=limit, replace=False))
                symbols = [symbols[int(i)] for i in selected_idx]
                symbols = sorted(symbols)
            else:
                symbols = symbols[:limit]

    days = _days_for_range(start_date, end_date, warmup_days=420)
    print(f"Loading price pack for {len(symbols)} Russell 3000 symbols ({start_date} -> {end_date}, days={days})")
    data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership or len(membership) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    friction_vals: List[float] = []
    for part in str(args.frictions or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        val = _safe_float(txt, float("nan"))
        if np.isfinite(val) and val >= 0:
            friction_vals.append(float(val))
    if not friction_vals:
        friction_vals = [10.0]
    friction_grid = [
        FrictionScenario(f"{int(v)}x{int(v)}", args.transaction_cost_bps, float(v), float(v))
        for v in friction_vals
    ]

    windows_36_12 = _build_test_windows(start_date, end_date, train_months=36, test_months=12)
    windows_60_12 = _build_test_windows(start_date, end_date, train_months=60, test_months=12)

    scenarios = []
    for fr in friction_grid:
        cfg_fr = _apply_friction(cfg, fr)
        print(f"Running scenario {fr.name}: tc={fr.transaction_cost_bps}bps, entry={fr.entry_slippage_bps}bps, exit={fr.exit_slippage_bps}bps")
        full = _run_window(prepared, global_data, membership, cfg_fr, start_date, end_date)
        stitched_36_12 = _stitch_test_windows(prepared, global_data, membership, cfg_fr, windows_36_12, test_months=12)
        stitched_60_12 = _stitch_test_windows(prepared, global_data, membership, cfg_fr, windows_60_12, test_months=12)
        snapshot = _acceptance_snapshot(full, stitched_36_12, stitched_60_12)
        scenarios.append(
            {
                "scenario": fr.name,
                "friction": {
                    "transaction_cost_bps": fr.transaction_cost_bps,
                    "entry_slippage_bps": fr.entry_slippage_bps,
                    "exit_slippage_bps": fr.exit_slippage_bps,
                },
                "full": {
                    "cagr_pct": _safe_float(full.get("cagr"), float("nan")) * 100.0 if np.isfinite(_safe_float(full.get("cagr"), float("nan"))) else float("nan"),
                    "max_dd_pct": _pct_dd(full.get("max_drawdown_pct", 0.0)),
                    "trades": int(full.get("total_trades", 0) or 0),
                    "final_value": _safe_float(full.get("final_value"), float("nan")),
                    "audit": full.get("audit_report", {}),
                },
                "stitched_36_12": stitched_36_12,
                "stitched_60_12": stitched_60_12,
                "acceptance": snapshot,
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path),
        "start_date": start_date,
        "end_date": end_date,
        "universe": {
            "name": "RUSSELL3000",
            "count": len(symbols),
            "source": source,
            "membership_source": membership_source,
            "limit_mode": str(args.universe_limit_mode),
            "sample_seed": int(args.universe_sample_seed),
        },
        "constraints": {
            "allow_margin": False,
            "max_total_exposure_pct_bull": min(1.0, float(cfg.get("max_total_exposure_pct_bull", 1.0) or 1.0)),
            "max_total_exposure_pct_bear": min(1.0, float(cfg.get("max_total_exposure_pct_bear", 0.0) or 0.0)),
        },
        "walkforward_windows": {
            "36_12": windows_36_12,
            "60_12": windows_60_12,
        },
        "scenarios": scenarios,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (ROOT / "logs" / f"walkforward_superperformance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    main()
