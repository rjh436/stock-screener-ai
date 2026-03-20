from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import PreparedBacktestData, _normalize_prepared_calendar, prepare_backtest_data, run_backtest
from optimization.robustness import rejection_score
from strategies.superperformance import SuperperformanceStrategy


DEFAULT_WALKFORWARD_START = "2016-01-01"
DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT = 16.0
DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT = 30.0


@dataclass(frozen=True)
class FrictionScenario:
    name: str
    transaction_cost_bps: float
    entry_slippage_bps: float
    exit_slippage_bps: float


@dataclass(frozen=True)
class WalkforwardContext:
    start_date: str
    end_date: str
    symbols: tuple[str, ...]
    source: str
    membership_source: str
    universe_limit_mode: str
    universe_sample_seed: int
    prepared_data_source: str
    prepared: Any
    global_data: Dict[str, pd.DataFrame]
    membership_by_day: List[set[str]]
    windows_36_12: tuple[tuple[str, str], ...]
    windows_60_12: tuple[tuple[str, str], ...]


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def parse_friction_values(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, (list, tuple)):
        out = []
        for item in raw:
            value = safe_float(item, float("nan"))
            if np.isfinite(value) and value >= 0.0:
                out.append(float(value))
        return tuple(out) or (10.0,)
    out = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        value = safe_float(text, float("nan"))
        if np.isfinite(value) and value >= 0.0:
            out.append(float(value))
    return tuple(out) or (10.0,)


def pct_dd(value: Any) -> float:
    dd = safe_float(value, 0.0)
    if dd <= 1.0:
        dd *= 100.0
    return float(abs(dd))


def days_for_range(start_date: str, end_date: str, warmup_days: int = 420) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    span_days = max(1, int((end_ts - start_ts).days))
    return span_days + int(max(0, warmup_days))


def enforce_cash_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["allow_margin"] = False
    out["max_total_exposure_pct_bull"] = min(1.0, max(0.0, safe_float(out.get("max_total_exposure_pct_bull"), 1.0)))
    out["max_total_exposure_pct_bear"] = min(1.0, max(0.0, safe_float(out.get("max_total_exposure_pct_bear"), 0.0)))
    out["max_pos_size_pct"] = min(1.0, max(0.0, safe_float(out.get("max_pos_size_pct"), 0.2)))
    out["vcp_max_pos_size_pct"] = min(
        out["max_pos_size_pct"],
        max(0.0, safe_float(out.get("vcp_max_pos_size_pct"), out["max_pos_size_pct"])),
    )
    out["ep_max_pos_size_pct"] = min(
        out["max_pos_size_pct"],
        max(0.0, safe_float(out.get("ep_max_pos_size_pct"), out["max_pos_size_pct"])),
    )
    return out


def apply_friction(cfg: Dict[str, Any], scenario: FrictionScenario) -> Dict[str, Any]:
    out = dict(cfg)
    out["transaction_cost_bps"] = float(max(0.0, scenario.transaction_cost_bps))
    out["entry_slippage_bps"] = float(max(0.0, scenario.entry_slippage_bps))
    out["exit_slippage_bps"] = float(max(0.0, scenario.exit_slippage_bps))
    out["slippage_bps"] = float((out["entry_slippage_bps"] + out["exit_slippage_bps"]) / 2.0)
    return out


def annualized_cagr_from_values(start_val: float, end_val: float, years: float) -> float:
    if start_val <= 0 or end_val <= 0 or years <= 0:
        return float("nan")
    return ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0


def build_test_windows(start_date: str, end_date: str, train_months: int, test_months: int, step_months: int = 12) -> List[Tuple[str, str]]:
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


def stitch_test_windows(
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
        final_val = safe_float(out.get("final_value"), 100000.0)
        ret = (final_val / 100000.0) - 1.0 if final_val > 0 else float("nan")
        dd = pct_dd(out.get("max_drawdown_pct", 0.0))
        cagr = safe_float(out.get("cagr"), float("nan"))
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
    for ret in fold_returns:
        compounded *= (1.0 + ret)
    total_test_years = float(len(fold_returns) * (float(test_months) / 12.0))
    stitched_cagr = annualized_cagr_from_values(1.0, compounded, total_test_years)
    worst_fold = float(min(fold_returns) * 100.0)
    return {
        "folds": fold_rows,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "worst_fold_return_pct": worst_fold,
    }


def acceptance_snapshot(full_res: Dict[str, Any], stitched_36_12: Dict[str, Any], stitched_60_12: Dict[str, Any]) -> Dict[str, Any]:
    max_gross = safe_float(((full_res.get("audit_report") or {}).get("max_gross_exposure_pct")), 0.0)
    cagr = safe_float(full_res.get("cagr"), float("nan"))
    if np.isfinite(cagr):
        cagr *= 100.0
    mdd = pct_dd(full_res.get("max_drawdown_pct", 0.0))
    oos_36 = safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan"))
    oos_60 = safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan"))

    return {
        "cash_only_ok": bool(np.isfinite(max_gross) and max_gross <= 1.0001),
        "full_cagr_pct": float(cagr) if np.isfinite(cagr) else float("nan"),
        "full_max_dd_pct": float(mdd),
        "oos_36_12_cagr_pct": float(oos_36) if np.isfinite(oos_36) else float("nan"),
        "oos_60_12_cagr_pct": float(oos_60) if np.isfinite(oos_60) else float("nan"),
        "meets_min_robust_target": bool(np.isfinite(oos_36) and np.isfinite(oos_60) and min(oos_36, oos_60) >= DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT),
        "meets_stretch_target": bool(np.isfinite(oos_36) and np.isfinite(oos_60) and min(oos_36, oos_60) >= 20.0),
    }


def _limit_symbols(symbols: List[str], *, universe_limit: int, universe_limit_mode: str, universe_sample_seed: int) -> List[str]:
    if universe_limit <= 0 or universe_limit >= len(symbols):
        return symbols
    if str(universe_limit_mode).lower() == "sample":
        rng = np.random.default_rng(int(universe_sample_seed))
        selected_idx = np.sort(rng.choice(len(symbols), size=int(universe_limit), replace=False))
        chosen = [symbols[int(i)] for i in selected_idx]
        return sorted(chosen)
    return symbols[: int(universe_limit)]


def load_prepared_cache(cache_path: str | Path, symbols: Sequence[str]) -> PreparedBacktestData:
    path = Path(cache_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Prepared cache not found: {path}")
    with path.open("rb") as handle:
        prepared = pickle.load(handle)
    if not isinstance(prepared, PreparedBacktestData):
        raise RuntimeError(f"Prepared cache did not contain PreparedBacktestData: {path}")
    prepared = _normalize_prepared_calendar(prepared)
    selected = tuple(sorted({str(sym).upper() for sym in symbols if str(sym).upper() in prepared.enriched}))
    if not selected:
        raise RuntimeError(f"Prepared cache had no overlap with requested PIT universe: {path}")
    return PreparedBacktestData(
        enriched={sym: prepared.enriched[sym] for sym in selected},
        all_dates=np.array(prepared.all_dates, copy=False),
    )


def prepare_walkforward_context(
    *,
    start_date: str = DEFAULT_WALKFORWARD_START,
    end_date: str | None = None,
    universe_limit: int = 0,
    universe_limit_mode: str = "sample",
    universe_sample_seed: int = 42,
    prepared_cache_path: str = "",
) -> WalkforwardContext:
    end_value = end_date or pd.Timestamp.now().date().isoformat()
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_value)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError("PIT Russell 3000 universe is required. Source was non-PIT.")
    if not symbols:
        raise RuntimeError("PIT Russell 3000 universe returned zero symbols.")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    symbols = _limit_symbols(
        symbols,
        universe_limit=int(universe_limit),
        universe_limit_mode=str(universe_limit_mode),
        universe_sample_seed=int(universe_sample_seed),
    )

    days = days_for_range(start_date, end_value, warmup_days=420)
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}
    prepared_data_source = "live_fetch"
    if prepared_cache_path:
        prepared = load_prepared_cache(prepared_cache_path, symbols)
        symbols = sorted(prepared.enriched)
        prepared_data_source = str(Path(prepared_cache_path).expanduser().resolve())
    else:
        data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
        prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership or len(membership) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    windows_36_12 = build_test_windows(start_date, end_value, train_months=36, test_months=12)
    windows_60_12 = build_test_windows(start_date, end_value, train_months=60, test_months=12)

    return WalkforwardContext(
        start_date=str(start_date),
        end_date=str(end_value),
        symbols=tuple(symbols),
        source=str(source),
        membership_source=str(membership_source),
        universe_limit_mode=str(universe_limit_mode),
        universe_sample_seed=int(universe_sample_seed),
        prepared_data_source=prepared_data_source,
        prepared=prepared,
        global_data=global_data,
        membership_by_day=membership,
        windows_36_12=tuple(windows_36_12),
        windows_60_12=tuple(windows_60_12),
    )


def run_walkforward_report_with_context(
    config: Dict[str, Any],
    context: WalkforwardContext,
    *,
    transaction_cost_bps: float = 2.0,
    friction_values: Sequence[float] = (10.0,),
    config_path: str = "",
) -> Dict[str, Any]:
    cfg = enforce_cash_only(copy.deepcopy(config))

    friction_grid = [
        FrictionScenario(f"{int(v)}x{int(v)}", float(transaction_cost_bps), float(v), float(v))
        for v in (float(max(0.0, safe_float(v, 0.0))) for v in friction_values)
        if np.isfinite(float(v))
    ]
    if not friction_grid:
        friction_grid = [FrictionScenario("10x10", float(transaction_cost_bps), 10.0, 10.0)]

    scenarios = []
    for fr in friction_grid:
        cfg_fr = apply_friction(cfg, fr)
        full = _run_window(context.prepared, context.global_data, context.membership_by_day, cfg_fr, context.start_date, context.end_date)
        stitched_36_12 = stitch_test_windows(
            context.prepared,
            context.global_data,
            context.membership_by_day,
            cfg_fr,
            list(context.windows_36_12),
            test_months=12,
        )
        stitched_60_12 = stitch_test_windows(
            context.prepared,
            context.global_data,
            context.membership_by_day,
            cfg_fr,
            list(context.windows_60_12),
            test_months=12,
        )
        snapshot = acceptance_snapshot(full, stitched_36_12, stitched_60_12)
        scenarios.append(
            {
                "scenario": fr.name,
                "friction": {
                    "transaction_cost_bps": fr.transaction_cost_bps,
                    "entry_slippage_bps": fr.entry_slippage_bps,
                    "exit_slippage_bps": fr.exit_slippage_bps,
                },
                "full": {
                    "cagr_pct": safe_float(full.get("cagr"), float("nan")) * 100.0 if np.isfinite(safe_float(full.get("cagr"), float("nan"))) else float("nan"),
                    "max_dd_pct": pct_dd(full.get("max_drawdown_pct", 0.0)),
                    "trades": int(full.get("total_trades", 0) or 0),
                    "final_value": safe_float(full.get("final_value"), float("nan")),
                    "audit": full.get("audit_report", {}),
                },
                "stitched_36_12": stitched_36_12,
                "stitched_60_12": stitched_60_12,
                "acceptance": snapshot,
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path or ""),
        "start_date": context.start_date,
        "end_date": context.end_date,
        "universe": {
            "name": "RUSSELL3000",
            "count": len(context.symbols),
            "source": context.source,
            "membership_source": context.membership_source,
            "limit_mode": context.universe_limit_mode,
            "sample_seed": context.universe_sample_seed,
            "prepared_data_source": context.prepared_data_source,
        },
        "constraints": {
            "allow_margin": False,
            "max_total_exposure_pct_bull": min(1.0, float(cfg.get("max_total_exposure_pct_bull", 1.0) or 1.0)),
            "max_total_exposure_pct_bear": min(1.0, float(cfg.get("max_total_exposure_pct_bear", 0.0) or 0.0)),
        },
        "walkforward_windows": {
            "36_12": list(context.windows_36_12),
            "60_12": list(context.windows_60_12),
        },
        "scenarios": scenarios,
    }


def run_walkforward_report(
    config: Dict[str, Any],
    *,
    start_date: str = DEFAULT_WALKFORWARD_START,
    end_date: str | None = None,
    transaction_cost_bps: float = 2.0,
    friction_values: Sequence[float] = (10.0,),
    universe_limit: int = 0,
    universe_limit_mode: str = "sample",
    universe_sample_seed: int = 42,
    prepared_cache_path: str = "",
    config_path: str = "",
) -> Dict[str, Any]:
    context = prepare_walkforward_context(
        start_date=start_date,
        end_date=end_date,
        universe_limit=universe_limit,
        universe_limit_mode=universe_limit_mode,
        universe_sample_seed=universe_sample_seed,
        prepared_cache_path=prepared_cache_path,
    )
    return run_walkforward_report_with_context(
        config,
        context,
        transaction_cost_bps=transaction_cost_bps,
        friction_values=friction_values,
        config_path=config_path,
    )


def run_walkforward_batch_reports(
    configs: Sequence[Dict[str, Any]],
    *,
    start_date: str = DEFAULT_WALKFORWARD_START,
    end_date: str | None = None,
    transaction_cost_bps: float = 2.0,
    friction_values: Sequence[float] = (10.0,),
    universe_limit: int = 0,
    universe_limit_mode: str = "sample",
    universe_sample_seed: int = 42,
    prepared_cache_path: str = "",
    config_paths: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    context = prepare_walkforward_context(
        start_date=start_date,
        end_date=end_date,
        universe_limit=universe_limit,
        universe_limit_mode=universe_limit_mode,
        universe_sample_seed=universe_sample_seed,
        prepared_cache_path=prepared_cache_path,
    )
    path_list = list(config_paths or [])
    reports: List[Dict[str, Any]] = []
    for idx, config in enumerate(configs):
        cfg_path = path_list[idx] if idx < len(path_list) else ""
        reports.append(
            run_walkforward_report_with_context(
                config,
                context,
                transaction_cost_bps=transaction_cost_bps,
                friction_values=friction_values,
                config_path=cfg_path,
            )
        )
    return reports


def select_walkforward_scenario(report: Dict[str, Any], scenario_name: str = "") -> Dict[str, Any]:
    scenarios = report.get("scenarios") or []
    if not isinstance(scenarios, list) or not scenarios:
        raise RuntimeError("Walk-forward report is missing scenarios.")
    if scenario_name:
        target = str(scenario_name).strip()
        for scenario in scenarios:
            if str(scenario.get("scenario", "")) == target:
                return scenario
        raise RuntimeError(f"Walk-forward scenario not found: {target}")
    return scenarios[0]


def walkforward_summary(report: Dict[str, Any], *, scenario_name: str = "") -> Dict[str, Any]:
    scenario = select_walkforward_scenario(report, scenario_name=scenario_name)
    acceptance = scenario.get("acceptance") or {}
    full = scenario.get("full") or {}
    stitched_36_12 = scenario.get("stitched_36_12") or {}
    stitched_60_12 = scenario.get("stitched_60_12") or {}
    return {
        "scenario": str(scenario.get("scenario", "")),
        "full_cagr_pct": safe_float(full.get("cagr_pct"), float("nan")),
        "full_max_dd_pct": safe_float(full.get("max_dd_pct"), float("nan")),
        "full_trades": int(full.get("trades", 0) or 0),
        "same_day_open_entries": int(((full.get("audit") or {}).get("same_day_open_entries", 0)) or 0),
        "max_gross_exposure_pct": safe_float(((full.get("audit") or {}).get("max_gross_exposure_pct")), float("nan")),
        "stitched_36_12_cagr_pct": safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan")),
        "stitched_36_12_worst_fold_return_pct": safe_float(stitched_36_12.get("worst_fold_return_pct"), float("nan")),
        "stitched_36_12_fold_count": int(stitched_36_12.get("fold_count", 0) or 0),
        "stitched_60_12_cagr_pct": safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan")),
        "stitched_60_12_worst_fold_return_pct": safe_float(stitched_60_12.get("worst_fold_return_pct"), float("nan")),
        "stitched_60_12_fold_count": int(stitched_60_12.get("fold_count", 0) or 0),
        "cash_only_ok": bool(acceptance.get("cash_only_ok", False)),
        "meets_min_robust_target": bool(acceptance.get("meets_min_robust_target", False)),
        "meets_stretch_target": bool(acceptance.get("meets_stretch_target", False)),
    }


def walkforward_gate_status(
    report: Dict[str, Any],
    *,
    scenario_name: str = "",
    min_oos_cagr_pct: float = DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT,
    max_full_drawdown_pct: float = DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT,
) -> tuple[bool, str, Dict[str, Any]]:
    summary = walkforward_summary(report, scenario_name=scenario_name)
    if not summary["cash_only_ok"]:
        return False, "walk-forward cash-only gate failed", summary
    full_dd_pct = safe_float(summary.get("full_max_dd_pct"), float("nan"))
    if np.isfinite(full_dd_pct) and full_dd_pct > float(max_full_drawdown_pct):
        return False, f"walk-forward full max drawdown {full_dd_pct:.2f}% exceeded {float(max_full_drawdown_pct):.2f}%", summary
    oos_36 = safe_float(summary.get("stitched_36_12_cagr_pct"), float("nan"))
    if not np.isfinite(oos_36) or oos_36 < float(min_oos_cagr_pct):
        return False, f"walk-forward 36/12 stitched CAGR {oos_36:.2f}% below {float(min_oos_cagr_pct):.2f}%", summary
    oos_60 = safe_float(summary.get("stitched_60_12_cagr_pct"), float("nan"))
    if not np.isfinite(oos_60) or oos_60 < float(min_oos_cagr_pct):
        return False, f"walk-forward 60/12 stitched CAGR {oos_60:.2f}% below {float(min_oos_cagr_pct):.2f}%", summary
    return True, "", summary


def promotion_walkforward_status(
    config: Dict[str, Any],
    *,
    start_date: str = DEFAULT_WALKFORWARD_START,
    end_date: str | None = None,
    transaction_cost_bps: float = 2.0,
    friction_values: Sequence[float] = (10.0,),
    scenario_name: str = "",
    min_oos_cagr_pct: float = DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT,
    max_full_drawdown_pct: float = DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT,
    universe_limit: int = 0,
    universe_limit_mode: str = "sample",
    universe_sample_seed: int = 42,
    prepared_cache_path: str = "",
    config_path: str = "",
) -> tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
    report = run_walkforward_report(
        config,
        start_date=start_date,
        end_date=end_date,
        transaction_cost_bps=transaction_cost_bps,
        friction_values=friction_values,
        universe_limit=universe_limit,
        universe_limit_mode=universe_limit_mode,
        universe_sample_seed=universe_sample_seed,
        prepared_cache_path=prepared_cache_path,
        config_path=config_path,
    )
    ok, reason, summary = walkforward_gate_status(
        report,
        scenario_name=scenario_name,
        min_oos_cagr_pct=min_oos_cagr_pct,
        max_full_drawdown_pct=max_full_drawdown_pct,
    )
    return ok, reason, summary, report


def replay_ranked_candidates(
    rows: Sequence[Dict[str, Any]],
    *,
    top_n: int,
    start_date: str = DEFAULT_WALKFORWARD_START,
    end_date: str | None = None,
    transaction_cost_bps: float = 2.0,
    friction_values: Sequence[float] = (10.0,),
    scenario_name: str = "",
    min_oos_cagr_pct: float = DEFAULT_WALKFORWARD_MIN_OOS_CAGR_PCT,
    max_full_drawdown_pct: float = DEFAULT_WALKFORWARD_MAX_FULL_DD_PCT,
    universe_limit: int = 0,
    universe_limit_mode: str = "sample",
    universe_sample_seed: int = 42,
    prepared_cache_path: str = "",
    report_dir: str | Path | None = None,
    report_prefix: str = "walkforward_candidate",
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    annotated = [copy.deepcopy(row) for row in rows]
    if top_n <= 0 or not annotated:
        return sorted(annotated, key=lambda row: float(row.get(score_key, -1_000_000.0) or -1_000_000.0), reverse=True)

    ranked = sorted(annotated, key=lambda row: float(row.get(score_key, -1_000_000.0) or -1_000_000.0), reverse=True)
    base_dir = Path(report_dir).expanduser().resolve() if report_dir else None
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)

    replay_rows = [row for row in ranked[: int(top_n)] if isinstance(row.get("config"), dict)]
    context = None
    if replay_rows:
        context = prepare_walkforward_context(
            start_date=start_date,
            end_date=end_date,
            universe_limit=universe_limit,
            universe_limit_mode=universe_limit_mode,
            universe_sample_seed=universe_sample_seed,
            prepared_cache_path=prepared_cache_path,
        )

    for idx, row in enumerate(ranked):
        row["walkforward_replayed"] = False
        row["walkforward_ok"] = None
        if idx >= int(top_n):
            continue
        row["walkforward_replayed"] = True
        cfg = row.get("config")
        if not isinstance(cfg, dict):
            row["walkforward_ok"] = False
            row["walkforward_reason"] = "candidate config missing"
            row["score_pre_walkforward"] = float(row.get(score_key, rejection_score(999)) or rejection_score(999))
            row[score_key] = rejection_score(400)
            continue

        name = str(row.get("name", f"candidate_{idx+1}") or f"candidate_{idx+1}")
        if context is None:
            row["walkforward_ok"] = False
            row["walkforward_reason"] = "walk-forward context unavailable"
            row["score_pre_walkforward"] = float(row.get(score_key, rejection_score(999)) or rejection_score(999))
            row[score_key] = rejection_score(400 + idx)
            continue

        report = run_walkforward_report_with_context(
            cfg,
            context,
            transaction_cost_bps=transaction_cost_bps,
            friction_values=friction_values,
            config_path=name,
        )
        ok, reason, summary = walkforward_gate_status(
            report,
            scenario_name=scenario_name,
            min_oos_cagr_pct=min_oos_cagr_pct,
            max_full_drawdown_pct=max_full_drawdown_pct,
        )
        row["walkforward_ok"] = bool(ok)
        row["walkforward_reason"] = str(reason)
        row["walkforward_summary"] = summary
        if base_dir is not None:
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or f"candidate_{idx+1}"
            report_path = base_dir / f"{report_prefix}_{idx+1:02d}_{safe_name}.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            row["walkforward_report_path"] = str(report_path)
        if not ok:
            row["score_pre_walkforward"] = float(row.get(score_key, rejection_score(999)) or rejection_score(999))
            row[score_key] = rejection_score(400 + idx)

    return sorted(ranked, key=lambda row: float(row.get(score_key, -1_000_000.0) or -1_000_000.0), reverse=True)
