#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
from scripts.run_factor_walkforward import (
    FrictionScenario,
    _build_market_risk_scalar,
    _build_momentum_quality_scores,
    _load_symbol_map,
    _build_value_proxy_scores,
    _extract_feature_arrays,
    _pct_dd,
    _run_factor_window,
    _safe_float,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return cfg


def _days_for_range(start_date: str, end_date: str, warmup_days: int = 420) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return max(1, int((end_ts - start_ts).days)) + int(max(0, warmup_days))


def _equity_series(result: Dict[str, Any]) -> pd.Series:
    rows = result.get("equity_curve", []) or []
    if not rows:
        return pd.Series(dtype=float)
    dates = []
    vals = []
    for row in rows:
        d = row.get("Date")
        e = _safe_float(row.get("Equity"), float("nan"))
        if d is None or not np.isfinite(e):
            continue
        dates.append(pd.Timestamp(d))
        vals.append(float(e))
    if not dates:
        return pd.Series(dtype=float)
    s = pd.Series(vals, index=pd.DatetimeIndex(dates)).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _blend_curves(curves: Mapping[str, pd.Series], weights: Mapping[str, float]) -> pd.Series:
    valid = {k: v for k, v in curves.items() if isinstance(v, pd.Series) and not v.empty and k in weights}
    if not valid:
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in valid.values()])))
    if len(idx) == 0:
        return pd.Series(dtype=float)

    wsum = sum(float(weights[k]) for k in valid)
    if wsum <= 0:
        return pd.Series(dtype=float)

    norm_weights = {k: float(weights[k]) / wsum for k in valid}
    blended = pd.Series(0.0, index=idx, dtype=float)

    for name, series in valid.items():
        aligned = series.reindex(idx).ffill().bfill()
        if aligned.empty:
            continue
        base = float(aligned.iloc[0])
        if not np.isfinite(base) or base <= 0:
            continue
        norm = aligned / base
        blended += norm * norm_weights[name]

    if blended.empty:
        return blended
    return blended * 100000.0


def _curve_metrics(curve: pd.Series, start_cash: float = 100000.0) -> Dict[str, float]:
    if curve is None or curve.empty:
        return {"final_value": start_cash, "cagr_pct": 0.0, "max_dd_pct": 0.0}

    eq = curve.astype(float)
    final_value = float(eq.iloc[-1])
    total_days = max(1, int((eq.index[-1] - eq.index[0]).days))
    years = max(0.1, total_days / 365.25)
    cagr = ((final_value / float(start_cash)) ** (1.0 / years) - 1.0) * 100.0 if start_cash > 0 else 0.0

    peaks = eq.cummax()
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = (eq - peaks) / peaks
    max_dd = abs(float(dd.min())) * 100.0 if len(dd) else 0.0

    return {"final_value": final_value, "cagr_pct": float(cagr), "max_dd_pct": float(max_dd)}


def _run_event_window(
    *,
    cfg: Dict[str, Any],
    prepared,
    global_data: Dict[str, pd.DataFrame],
    membership_by_day: Sequence[Optional[Iterable[str]]],
    start_date: str,
    end_date: str,
    friction: FrictionScenario,
) -> Dict[str, Any]:
    local = dict(cfg)
    local["allow_margin"] = False
    local["transaction_cost_bps"] = float(friction.transaction_cost_bps)
    local["entry_slippage_bps"] = float(friction.entry_slippage_bps)
    local["exit_slippage_bps"] = float(friction.exit_slippage_bps)

    strat = SuperperformanceStrategy(local)
    out = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership_by_day,
        require_pit_membership=True,
        backtest_mode="event",
    )
    return out[0] if isinstance(out, list) else out


def _is_factor_config(cfg: Dict[str, Any]) -> bool:
    return str(cfg.get("strategy_type", "") or "").lower() in {"cross_sectional_momentum", "separate_value_momentum"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alpha sleeve ablation and blended portfolio diagnostics")
    parser.add_argument("--configs", nargs="+", required=True, help="Config paths to include in ablation")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--frictions", default="10,20,35,50")
    parser.add_argument("--cache-only", action="store_true", help="Use local cache only")
    parser.add_argument("--min-universe-coverage", type=float, default=0.85)
    args = parser.parse_args()

    start_date = str(args.start)
    end_date = str(args.end)
    config_paths = [Path(p).expanduser().resolve() for p in args.configs]
    configs: List[Dict[str, Any]] = []
    for p in config_paths:
        cfg = _load_config(p)
        cfg["_path"] = str(p)
        cfg["_label"] = str(cfg.get("name") or p.stem)
        cfg["_blend_weight"] = float(cfg.get("blend_weight", 1.0) or 1.0)
        configs.append(cfg)

    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError(f"PIT Russell 3000 universe required, got source={source}")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})

    days = _days_for_range(start_date, end_date)
    print(f"Loading Russell 3000 PIT union: {len(symbols)} symbols ({start_date} -> {end_date}, days={days})")
    data = fetch_data_pack(symbols, days=days, backtest_mode=bool(args.cache_only)) or {}
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    loaded_symbols = sorted(list(data.keys()))
    coverage = (len(loaded_symbols) / float(len(symbols))) if symbols else 0.0
    print(f"Loaded symbols with price history: {len(loaded_symbols)} / {len(symbols)} ({coverage:.1%})")
    if coverage < float(args.min_universe_coverage):
        raise RuntimeError(
            f"Universe coverage too low ({coverage:.1%} < {float(args.min_universe_coverage):.1%}). "
            "Refusing to run reduced-universe ablation."
        )

    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership_by_day or len(membership_by_day) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    # Precompute factor matrices once.
    features = _extract_feature_arrays(prepared, membership_by_day)
    prices = pd.DataFrame(features["close"], index=features["dates"], columns=features["symbols"])
    eps_yoy_df = pd.DataFrame(features["eps_yoy"], index=features["dates"], columns=features["symbols"])
    sales_yoy_df = pd.DataFrame(features["sales_yoy"], index=features["dates"], columns=features["symbols"])
    sector_map = _load_symbol_map(ROOT / "config" / "sectors.json")
    industry_map = _load_symbol_map(ROOT / "config" / "industries.json")
    if not industry_map:
        industry_map = _load_symbol_map(ROOT / "config" / "industry.json")

    factor_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
    for cfg in configs:
        if not _is_factor_config(cfg):
            continue
        mom = _build_momentum_quality_scores(features, cfg)
        val = None
        if str(cfg.get("strategy_type", "")).lower() == "separate_value_momentum":
            val = _build_value_proxy_scores(features, cfg)
        risk_scalar = _build_market_risk_scalar(global_data, prices.index, cfg)
        factor_cache[cfg["_path"]] = {"momentum": mom, "value": val, "risk_scalar": risk_scalar}

    friction_vals: List[float] = []
    for part in str(args.frictions or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        val = _safe_float(txt, float("nan"))
        if np.isfinite(val) and val >= 0:
            friction_vals.append(float(val))
    if not friction_vals:
        friction_vals = [20.0]

    friction_grid = [
        FrictionScenario(
            name=f"{int(v)}x{int(v)}",
            transaction_cost_bps=float(args.transaction_cost_bps),
            entry_slippage_bps=float(v),
            exit_slippage_bps=float(v),
        )
        for v in friction_vals
    ]

    scenarios = []
    for fr in friction_grid:
        print(
            f"Ablation scenario {fr.name}: tc={fr.transaction_cost_bps}bps, "
            f"entry={fr.entry_slippage_bps}bps, exit={fr.exit_slippage_bps}bps"
        )

        sleeve_runs: Dict[str, Dict[str, Any]] = {}
        sleeve_curves: Dict[str, pd.Series] = {}
        sleeve_weights: Dict[str, float] = {}

        for cfg in configs:
            label = cfg["_label"]
            sleeve_weights[label] = float(cfg.get("_blend_weight", 1.0))

            if _is_factor_config(cfg):
                mats = factor_cache[cfg["_path"]]
                run = _run_factor_window(
                    cfg=cfg,
                    prices=prices,
                    momentum_scores=mats["momentum"],
                    value_scores=mats["value"],
                    eps_yoy_df=eps_yoy_df,
                    sales_yoy_df=sales_yoy_df,
                    risk_scalar_by_date=mats.get("risk_scalar"),
                    sector_map=sector_map,
                    industry_map=industry_map,
                    start_date=start_date,
                    end_date=end_date,
                    friction=fr,
                )
            else:
                run = _run_event_window(
                    cfg=cfg,
                    prepared=prepared,
                    global_data=global_data,
                    membership_by_day=membership_by_day,
                    start_date=start_date,
                    end_date=end_date,
                    friction=fr,
                )

            sleeve_runs[label] = run
            sleeve_curves[label] = _equity_series(run)

        blended_all_curve = _blend_curves(sleeve_curves, sleeve_weights)
        blended_all_metrics = _curve_metrics(blended_all_curve)

        leave_one_out: Dict[str, Dict[str, float]] = {}
        for drop_label in sleeve_curves.keys():
            keep_weights = {k: v for k, v in sleeve_weights.items() if k != drop_label}
            loo_curve = _blend_curves(sleeve_curves, keep_weights)
            leave_one_out[drop_label] = _curve_metrics(loo_curve)

        # Contribution proxy by sleeve ending value relative to equal-start normalized curves.
        normalized_end: Dict[str, float] = {}
        for label, curve in sleeve_curves.items():
            if curve.empty:
                normalized_end[label] = 0.0
                continue
            base = float(curve.iloc[0])
            endv = float(curve.iloc[-1])
            normalized_end[label] = (endv / base) if base > 0 else 0.0
        total_norm_end = sum(normalized_end.values())
        pnl_share = {
            k: (float(v) / float(total_norm_end)) if total_norm_end > 0 else 0.0
            for k, v in normalized_end.items()
        }

        sleeves_payload = {}
        for label, run in sleeve_runs.items():
            sleeves_payload[label] = {
                "cagr_pct": _safe_float(run.get("cagr"), 0.0) * 100.0,
                "max_dd_pct": _pct_dd(run.get("max_drawdown_pct", 0.0)),
                "trades": int(run.get("total_trades", 0) or 0),
                "final_value": _safe_float(run.get("final_value"), 0.0),
                "avg_annual_turnover_pct": _safe_float(run.get("avg_annual_turnover_pct"), float("nan")),
                "max_gross_exposure_pct": _safe_float((run.get("audit_report") or {}).get("max_gross_exposure_pct"), 0.0),
                "pnl_share_proxy": pnl_share.get(label, 0.0),
            }

        scenarios.append(
            {
                "scenario": fr.name,
                "friction": {
                    "transaction_cost_bps": fr.transaction_cost_bps,
                    "entry_slippage_bps": fr.entry_slippage_bps,
                    "exit_slippage_bps": fr.exit_slippage_bps,
                },
                "sleeves": sleeves_payload,
                "blend_all": blended_all_metrics,
                "leave_one_out": leave_one_out,
                "go_no_go": {
                    "blend_20x20_gate": bool(fr.name == "20x20" and blended_all_metrics.get("cagr_pct", 0.0) >= 10.0),
                    "blend_35x35_gate": bool(fr.name == "35x35" and blended_all_metrics.get("cagr_pct", 0.0) >= 8.0),
                    "blend_mdd_gate": bool(blended_all_metrics.get("max_dd_pct", 999.0) <= 28.0),
                    "single_sleeve_dominance_over_60pct": bool(any(v > 0.60 for v in pnl_share.values())),
                },
            }
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start_date": start_date,
        "end_date": end_date,
        "configs": [str(p) for p in config_paths],
        "universe": {
            "name": "RUSSELL3000",
            "source": source,
            "membership_source": membership_source,
            "requested_symbol_count": len(symbols),
            "loaded_symbol_count": len(loaded_symbols),
            "coverage_ratio": coverage,
            "cache_only": bool(args.cache_only),
        },
        "scenarios": scenarios,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (ROOT / "logs" / f"ablation_multi_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {output_path}")


if __name__ == "__main__":
    main()
