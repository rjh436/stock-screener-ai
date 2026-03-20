#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.engine import _normalize_prepared_calendar
from optimization.walkforward import (
    DEFAULT_WALKFORWARD_START,
    FrictionScenario,
    annualized_cagr_from_values,
    apply_friction,
    build_test_windows,
    days_for_range,
    enforce_cash_only,
    pct_dd,
    prepare_walkforward_context,
    safe_float,
)
from optimization.walkforward import _run_window as _run_superperformance_window
from scripts.evaluate_etf_rotation_holdout import _config_symbols as _etf_config_symbols
from scripts.evaluate_etf_rotation_holdout import _requested_symbols as _etf_requested_symbols
from scripts.evaluate_smid_pullback_holdout import build_smid_pullback_context
from scripts.run_etf_rotation_walkforward import (
    _build_allocator_target_schedule,
    _build_residual_defensive_target_schedule,
    _build_rotation_scores,
    _download_yfinance_pack,
    _load_config as _load_etf_config,
    _normalize_symbol_list,
    _price_frame_from_data,
    _run_rotation_window,
    _run_rotation_window_with_targets,
)
from scripts.run_smid_pullback_walkforward import (
    _build_smid_pullback_scores,
    _load_config as _load_smid_config,
    _run_window as _run_smid_window,
    _slice_prices,
)


DEFAULT_ETF_CONFIG = ROOT / "config" / "etf_rotation_growth_core5_residual_defensive_calmar_v1.json"
DEFAULT_SMID_CONFIG = ROOT / "config" / "smid_pullback_r3000_tc4_tb003_v1.json"
DEFAULT_B17_CONFIG = ROOT / "config" / "superperformance_alpha_b17_promoted_v1.json"
DEFAULT_PREPARED_CACHE = ROOT / "data" / "cache_indicators.pkl"

DEFAULT_ALLOCATIONS = {
    "H2": {"etf": 0.00, "smid": 0.70, "b17": 0.30},
    "H3": {"etf": 0.10, "smid": 0.70, "b17": 0.20},
    "H5": {"etf": 0.20, "smid": 0.60, "b17": 0.20},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-allocation ETF/SMID/B17 walk-forward validation.")
    parser.add_argument("--start-date", default=DEFAULT_WALKFORWARD_START)
    parser.add_argument("--end-date", default="", help="Defaults to the last date available in the prepared Superperformance cache.")
    parser.add_argument("--etf-config", default=str(DEFAULT_ETF_CONFIG))
    parser.add_argument("--smid-config", default=str(DEFAULT_SMID_CONFIG))
    parser.add_argument("--b17-config", default=str(DEFAULT_B17_CONFIG))
    parser.add_argument("--prepared-cache", default=str(DEFAULT_PREPARED_CACHE))
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="Entry and exit slippage bps for the B17 sleeve.")
    parser.add_argument("--min-oos-cagr-pct", type=float, default=20.0)
    parser.add_argument("--max-full-dd-pct", type=float, default=20.0)
    parser.add_argument("--labels", default="H2,H3,H5", help="Comma-separated allocation labels to evaluate.")
    parser.add_argument("--out", default="")
    parser.add_argument("--out-csv", default="")
    return parser.parse_args()


def _resolve_end_date(requested: str, prepared_cache_path: str | Path) -> str:
    raw = str(requested or "").strip()
    if raw:
        return pd.Timestamp(raw).date().isoformat()
    cache_path = Path(prepared_cache_path).expanduser().resolve()
    with cache_path.open("rb") as handle:
        prepared = pickle.load(handle)
    prepared = _normalize_prepared_calendar(prepared)
    all_dates = list(getattr(prepared, "all_dates", []))
    if not all_dates:
        raise RuntimeError(f"Prepared cache had no dates: {cache_path}")
    return pd.Timestamp(all_dates[-1]).date().isoformat()


def _load_json_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config payload in {cfg_path}")
    return dict(payload)


def _parse_labels(raw: str) -> List[str]:
    labels = [part.strip().upper() for part in str(raw or "").split(",") if part.strip()]
    if not labels:
        labels = ["H2", "H3", "H5"]
    unknown = [label for label in labels if label not in DEFAULT_ALLOCATIONS]
    if unknown:
        raise ValueError(f"Unknown allocation labels: {unknown}")
    return labels


def _ordered_unique_windows(*window_lists: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    seen: set[Tuple[str, str]] = set()
    out: List[Tuple[str, str]] = []
    for windows in window_lists:
        for window in windows:
            item = (str(window[0]), str(window[1]))
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def _equity_curve_to_series(run: Mapping[str, Any]) -> pd.Series:
    curve = list((run or {}).get("equity_curve") or [])
    if not curve:
        raise RuntimeError("Run result did not include an equity curve.")
    idx = pd.to_datetime([row.get("Date") for row in curve], errors="coerce")
    vals = pd.to_numeric([row.get("Equity") for row in curve], errors="coerce")
    ser = pd.Series(vals, index=idx, dtype="float64").dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    if ser.empty:
        raise RuntimeError("Run result produced an empty equity curve.")
    return ser


def _series_returns(equity: pd.Series) -> pd.Series:
    ser = pd.to_numeric(equity, errors="coerce").dropna().astype("float64")
    if ser.empty:
        return pd.Series(dtype="float64")
    return ser.pct_change().fillna(0.0).astype("float64")


def _blend_equity_curves(curves: Mapping[str, pd.Series], weights: Mapping[str, float], *, start_cash: float = 100000.0) -> pd.Series:
    idx = pd.Index([])
    for ser in curves.values():
        idx = idx.union(ser.index)
    idx = pd.DatetimeIndex(idx).sort_values()
    blended_ret = pd.Series(0.0, index=idx, dtype="float64")
    for sleeve_name, weight in weights.items():
        equity = curves.get(sleeve_name)
        if equity is None:
            raise KeyError(f"Missing equity curve for sleeve '{sleeve_name}'")
        blended_ret = blended_ret.add(_series_returns(equity).reindex(idx).fillna(0.0) * float(weight), fill_value=0.0)
    return (1.0 + blended_ret).cumprod() * float(start_cash)


def _metrics_from_equity(equity: pd.Series, *, start_cash: float = 100000.0) -> Dict[str, float]:
    ser = pd.to_numeric(equity, errors="coerce").dropna().astype("float64")
    if ser.empty or len(ser) < 2:
        return {
            "cagr_pct": float("nan"),
            "max_dd_pct": float("nan"),
            "final_value": float("nan"),
            "days": int(len(ser)),
            "return_pct": float("nan"),
        }
    start_val = float(ser.iloc[0])
    end_val = float(ser.iloc[-1])
    total_days = int((ser.index[-1] - ser.index[0]).days)
    years = float(total_days) / 365.25 if total_days > 0 else float("nan")
    cagr_pct = annualized_cagr_from_values(start_val, end_val, years) if years > 0 else float("nan")
    running_max = ser.cummax()
    drawdown = (ser / running_max) - 1.0
    return {
        "cagr_pct": float(cagr_pct) if np.isfinite(cagr_pct) else float("nan"),
        "max_dd_pct": abs(float(drawdown.min()) * 100.0),
        "final_value": end_val,
        "days": int(len(ser)),
        "return_pct": ((end_val / float(start_cash)) - 1.0) * 100.0,
    }


def _cash_only_from_run(run: Mapping[str, Any]) -> bool:
    max_gross = safe_float(((run or {}).get("audit_report") or {}).get("max_gross_exposure_pct"), float("nan"))
    return bool(np.isfinite(max_gross) and max_gross <= 1.0001)


def _build_etf_runner(cfg_path: Path, *, start_date: str, end_date: str) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    cfg = _load_etf_config(cfg_path)
    cfg["_config_path"] = str(cfg_path)
    requested_symbols = _etf_requested_symbols([cfg])
    data = _download_yfinance_pack(requested_symbols, start_date=start_date, end_date=end_date)
    if not data:
        raise RuntimeError("ETF data pack download returned no data.")

    global_symbols = _normalize_symbol_list(cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))
    global_data = {sym: data[sym] for sym in global_symbols if sym in data}
    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})

    if isinstance(sleeves, Mapping) and sleeves:
        close_px, open_px, target_schedule, _ = _build_allocator_target_schedule(data, cfg)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError(f"Failed to build ETF allocator matrices for {cfg_path}")

        def _runner(window_start: str, window_end: str) -> Dict[str, Any]:
            return _run_rotation_window_with_targets(
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=window_start,
                end_date=window_end,
            )

    elif defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        close_px, open_px, target_schedule = _build_residual_defensive_target_schedule(data, cfg, global_data)
        if close_px.empty or open_px.empty or target_schedule.empty:
            raise RuntimeError(f"Failed to build ETF residual defensive matrices for {cfg_path}")
        run_cfg = copy.deepcopy(cfg)
        run_cfg["market_regime"] = {"enabled": False}
        run_cfg["exposure_control"] = {}

        def _runner(window_start: str, window_end: str) -> Dict[str, Any]:
            return _run_rotation_window_with_targets(
                cfg=run_cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=window_start,
                end_date=window_end,
            )

    else:
        symbols = [sym for sym in _etf_config_symbols(cfg) if sym in data]
        close_px = _price_frame_from_data(data, symbols, "close")
        open_px = _price_frame_from_data(data, symbols, "open").reindex(close_px.index)
        scores = _build_rotation_scores(close_px, cfg)
        if close_px.empty or open_px.empty or scores.empty:
            raise RuntimeError(f"Failed to build ETF ranked matrices for {cfg_path}")

        def _runner(window_start: str, window_end: str) -> Dict[str, Any]:
            return _run_rotation_window(
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                scores=scores,
                global_data=global_data,
                start_date=window_start,
                end_date=window_end,
            )

    meta = {
        "config_path": str(cfg_path),
        "config_name": str(cfg.get("name", "") or cfg_path.stem),
    }
    return _runner, meta


def _build_smid_runner(cfg_path: Path, *, start_date: str, end_date: str) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any]]:
    cfg = _load_smid_config(cfg_path)
    days = days_for_range(start_date, end_date, warmup_days=420)
    context = build_smid_pullback_context(
        universe="RUSSELL3000",
        start_date=start_date,
        end_date=end_date,
        days=int(days),
        config_paths=[str(cfg_path)],
    )
    scores = _build_smid_pullback_scores(context["features"], cfg)
    if scores.empty:
        raise RuntimeError(f"SMID score matrix was empty for {cfg_path}")
    price_frames = _slice_prices(context["features"], ["open", "close"], list(scores.columns))

    def _runner(window_start: str, window_end: str) -> Dict[str, Any]:
        return _run_smid_window(
            cfg=cfg,
            prices_close=price_frames["close"],
            prices_open=price_frames["open"],
            scores=scores,
            global_data=context["global_data"],
            start_date=window_start,
            end_date=window_end,
        )

    meta = {
        "config_path": str(cfg_path),
        "config_name": str(cfg.get("name", "") or cfg_path.stem),
        "coverage_stats": context.get("coverage_stats", {}),
        "requested_symbols": int(len(context.get("requested_symbols") or [])),
        "loaded_symbols": int(len(context.get("loaded_symbols") or [])),
    }
    return _runner, meta


def _build_b17_runner(
    cfg_path: Path,
    *,
    start_date: str,
    end_date: str,
    prepared_cache_path: Path,
    transaction_cost_bps: float,
    slippage_bps: float,
) -> Tuple[Callable[[str, str], Dict[str, Any]], Dict[str, Any], Any]:
    cfg = enforce_cash_only(_load_json_config(cfg_path))
    friction = FrictionScenario("10x10", float(transaction_cost_bps), float(slippage_bps), float(slippage_bps))
    cfg = apply_friction(cfg, friction)
    context = prepare_walkforward_context(
        start_date=start_date,
        end_date=end_date,
        prepared_cache_path=str(prepared_cache_path),
    )

    def _runner(window_start: str, window_end: str) -> Dict[str, Any]:
        return _run_superperformance_window(
            context.prepared,
            context.global_data,
            context.membership_by_day,
            cfg,
            window_start,
            window_end,
        )

    meta = {
        "config_path": str(cfg_path),
        "config_name": str(cfg.get("name", "") or cfg_path.stem),
        "universe": {
            "count": len(context.symbols),
            "source": context.source,
            "membership_source": context.membership_source,
            "prepared_data_source": context.prepared_data_source,
        },
        "friction": {
            "transaction_cost_bps": friction.transaction_cost_bps,
            "entry_slippage_bps": friction.entry_slippage_bps,
            "exit_slippage_bps": friction.exit_slippage_bps,
        },
    }
    return _runner, meta, context


def _run_component_set(
    runner: Callable[[str, str], Dict[str, Any]],
    *,
    full_window: Tuple[str, str],
    windows: Sequence[Tuple[str, str]],
    label: str,
) -> Dict[str, Any]:
    full_run = runner(full_window[0], full_window[1])
    runs_by_window: Dict[Tuple[str, str], Dict[str, Any]] = {}
    cash_only_ok = _cash_only_from_run(full_run)
    for window_start, window_end in windows:
        print(f"[{label}] {window_start} -> {window_end}", flush=True)
        run = runner(window_start, window_end)
        runs_by_window[(window_start, window_end)] = run
        cash_only_ok = cash_only_ok and _cash_only_from_run(run)
    return {
        "full_run": full_run,
        "runs_by_window": runs_by_window,
        "cash_only_ok": bool(cash_only_ok),
    }


def _blend_window_metrics(
    components: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    window: Tuple[str, str],
) -> Dict[str, Any]:
    curves = {
        sleeve: _equity_curve_to_series(payload["runs_by_window"][window])
        for sleeve, payload in components.items()
    }
    equity = _blend_equity_curves(curves, weights)
    metrics = _metrics_from_equity(equity)
    return {
        "start": str(window[0]),
        "end": str(window[1]),
        "return_pct": float(metrics["return_pct"]),
        "cagr_pct": float(metrics["cagr_pct"]),
        "max_dd_pct": float(metrics["max_dd_pct"]),
    }


def _stitched_metrics(
    components: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    windows: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    folds = [_blend_window_metrics(components, weights, window) for window in windows]
    fold_returns = [float(row["return_pct"]) / 100.0 for row in folds if np.isfinite(float(row["return_pct"]))]
    if not fold_returns:
        return {
            "folds": folds,
            "fold_count": 0,
            "stitched_cagr_pct": float("nan"),
            "worst_fold_return_pct": float("nan"),
        }
    compounded = 1.0
    for ret in fold_returns:
        compounded *= (1.0 + ret)
    total_years = float(len(fold_returns))
    stitched_cagr = annualized_cagr_from_values(1.0, compounded, total_years)
    worst_fold = min(float(row["return_pct"]) for row in folds if np.isfinite(float(row["return_pct"])))
    return {
        "folds": folds,
        "fold_count": len(fold_returns),
        "stitched_cagr_pct": float(stitched_cagr) if np.isfinite(stitched_cagr) else float("nan"),
        "worst_fold_return_pct": float(worst_fold),
    }


def _build_allocation_report(
    label: str,
    weights: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
    *,
    windows_36_12: Sequence[Tuple[str, str]],
    windows_60_12: Sequence[Tuple[str, str]],
    min_oos_cagr_pct: float,
    max_full_dd_pct: float,
) -> Dict[str, Any]:
    full_curves = {
        sleeve: _equity_curve_to_series(payload["full_run"])
        for sleeve, payload in components.items()
    }
    full_equity = _blend_equity_curves(full_curves, weights)
    full_metrics = _metrics_from_equity(full_equity)
    stitched_36_12 = _stitched_metrics(components, weights, windows_36_12)
    stitched_60_12 = _stitched_metrics(components, weights, windows_60_12)
    component_cash_only = {sleeve: bool(payload["cash_only_ok"]) for sleeve, payload in components.items()}
    cash_only_ok = all(component_cash_only.values())
    worst_fold = min(
        value
        for value in [
            safe_float(stitched_36_12.get("worst_fold_return_pct"), float("nan")),
            safe_float(stitched_60_12.get("worst_fold_return_pct"), float("nan")),
        ]
        if np.isfinite(value)
    ) if any(np.isfinite(safe_float(item.get("worst_fold_return_pct"), float("nan"))) for item in [stitched_36_12, stitched_60_12]) else float("nan")
    accept = bool(
        cash_only_ok
        and np.isfinite(float(full_metrics["max_dd_pct"]))
        and float(full_metrics["max_dd_pct"]) <= float(max_full_dd_pct)
        and np.isfinite(safe_float(stitched_36_12.get("stitched_cagr_pct"), float("nan")))
        and np.isfinite(safe_float(stitched_60_12.get("stitched_cagr_pct"), float("nan")))
        and float(stitched_36_12["stitched_cagr_pct"]) >= float(min_oos_cagr_pct)
        and float(stitched_60_12["stitched_cagr_pct"]) >= float(min_oos_cagr_pct)
    )
    return {
        "label": str(label),
        "allocation": {name: int(round(float(weight) * 100.0)) for name, weight in weights.items()},
        "full_cagr_pct": float(full_metrics["cagr_pct"]),
        "full_max_dd_pct": float(full_metrics["max_dd_pct"]),
        "oos_36_12_cagr_pct": float(stitched_36_12["stitched_cagr_pct"]),
        "oos_60_12_cagr_pct": float(stitched_60_12["stitched_cagr_pct"]),
        "worst_12mo_fold_return_pct": float(worst_fold),
        "accept": bool(accept),
        "cash_only_ok": bool(cash_only_ok),
        "component_cash_only": component_cash_only,
        "stitched_36_12": stitched_36_12,
        "stitched_60_12": stitched_60_12,
    }


def _rows_to_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(
        [
            {
                "label": row.get("label"),
                "etf_pct": (row.get("allocation") or {}).get("etf"),
                "smid_pct": (row.get("allocation") or {}).get("smid"),
                "b17_pct": (row.get("allocation") or {}).get("b17"),
                "full_cagr_pct": row.get("full_cagr_pct"),
                "full_max_dd_pct": row.get("full_max_dd_pct"),
                "oos_36_12_cagr_pct": row.get("oos_36_12_cagr_pct"),
                "oos_60_12_cagr_pct": row.get("oos_60_12_cagr_pct"),
                "worst_12mo_fold_return_pct": row.get("worst_12mo_fold_return_pct"),
                "cash_only_ok": row.get("cash_only_ok"),
                "accept": row.get("accept"),
            }
            for row in rows
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    start_date = pd.Timestamp(args.start_date).date().isoformat()
    end_date = _resolve_end_date(args.end_date, args.prepared_cache)
    labels = _parse_labels(args.labels)

    etf_config_path = Path(args.etf_config).expanduser().resolve()
    smid_config_path = Path(args.smid_config).expanduser().resolve()
    b17_config_path = Path(args.b17_config).expanduser().resolve()
    prepared_cache_path = Path(args.prepared_cache).expanduser().resolve()

    b17_runner, b17_meta, b17_context = _build_b17_runner(
        b17_config_path,
        start_date=start_date,
        end_date=end_date,
        prepared_cache_path=prepared_cache_path,
        transaction_cost_bps=float(args.transaction_cost_bps),
        slippage_bps=float(args.slippage_bps),
    )
    windows_36_12 = list(b17_context.windows_36_12)
    windows_60_12 = list(b17_context.windows_60_12)
    unique_windows = _ordered_unique_windows(windows_36_12, windows_60_12)
    full_window = (start_date, end_date)

    etf_runner, etf_meta = _build_etf_runner(etf_config_path, start_date=start_date, end_date=end_date)
    smid_runner, smid_meta = _build_smid_runner(smid_config_path, start_date=start_date, end_date=end_date)

    components = {
        "etf": _run_component_set(etf_runner, full_window=full_window, windows=unique_windows, label="ETF"),
        "smid": _run_component_set(smid_runner, full_window=full_window, windows=unique_windows, label="SMID"),
        "b17": _run_component_set(b17_runner, full_window=full_window, windows=unique_windows, label="B17"),
    }

    rows = [
        _build_allocation_report(
            label,
            DEFAULT_ALLOCATIONS[label],
            components,
            windows_36_12=windows_36_12,
            windows_60_12=windows_60_12,
            min_oos_cagr_pct=float(args.min_oos_cagr_pct),
            max_full_dd_pct=float(args.max_full_dd_pct),
        )
        for label in labels
    ]

    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "walkforward_config": {
            "start_date": start_date,
            "end_date": end_date,
            "train_windows": [36, 60],
            "test_window_months": 12,
            "step_months": 12,
            "min_oos_cagr_pct": float(args.min_oos_cagr_pct),
            "max_full_dd_pct": float(args.max_full_dd_pct),
        },
        "component_sources": {
            "etf": etf_meta,
            "smid": smid_meta,
            "b17": b17_meta,
        },
        "component_cash_only": {name: bool(payload["cash_only_ok"]) for name, payload in components.items()},
        "results": rows,
    }

    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.out_csv:
        _rows_to_csv(Path(args.out_csv).expanduser().resolve(), rows)


if __name__ == "__main__":
    main()
