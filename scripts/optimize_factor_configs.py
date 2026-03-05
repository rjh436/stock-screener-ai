#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
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
from execution.engine import prepare_backtest_data
from scripts.run_factor_walkforward import (
    FrictionScenario,
    _daily_membership_price_coverage,
    _build_low_vol_scores,
    _build_market_risk_scalar,
    _build_momentum_quality_scores,
    _build_test_windows,
    _extract_feature_arrays,
    _load_symbol_map,
    _run_factor_window,
    _safe_float,
    _stitch_test_windows,
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


def _candidate_grid(base_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    strategy_type = str(base_cfg.get("strategy_type", "cross_sectional_momentum") or "cross_sectional_momentum").lower()
    if strategy_type == "momentum_low_vol_blend":
        for rebalance_freq, mom_count, lv_count, lv_weight, hold_buffer_mult, turnover_budget, max_pos, risk_off in itertools.product(
            ["M", "Q"],
            [14, 16, 18],
            [6, 8, 10],
            [0.20, 0.30, 0.40],
            [1.50, 1.75],
            [0.12, 0.18, 0.24],
            [0.05, 0.06, 0.075],
            [None, 0.35, 0.50],
        ):
            cfg = dict(base_cfg)
            cfg["rebalance_freq"] = rebalance_freq
            cfg["momentum_target_count"] = int(mom_count)
            cfg["low_vol_target_count"] = int(lv_count)
            cfg["momentum_weight"] = float(max(0.0, 1.0 - float(lv_weight)))
            cfg["low_vol_weight"] = float(lv_weight)
            cfg["hold_buffer_mult"] = float(hold_buffer_mult)
            cfg["turnover_budget"] = float(turnover_budget)
            cfg["max_position_weight"] = float(max_pos)
            cfg["name"] = (
                f"mlv_opt_rf{rebalance_freq}_m{mom_count}_lv{lv_count}_lvw{int(lv_weight*100):02d}"
                f"_hb{hold_buffer_mult:.2f}_tb{turnover_budget:.2f}_cap{max_pos:.3f}_ro"
                f"{'none' if risk_off is None else str(risk_off).replace('.', '')}"
            )
            if risk_off is None:
                cfg["market_regime"] = {"enabled": False}
            else:
                cfg["market_regime"] = {
                    "enabled": True,
                    "symbol": "SPY",
                    "ma_days": 200,
                    "risk_off_scalar": float(risk_off),
                }
            candidates.append(cfg)
        return candidates

    for rebalance_freq, target_count, hold_buffer_mult, turnover_budget, max_pos, risk_off in itertools.product(
        ["M", "Q"],
        [20, 24, 28],
        [1.50, 1.75],
        [0.12, 0.18, 0.24],
        [0.05, 0.06, 0.075],
        [None, 0.35, 0.50],
    ):
        cfg = dict(base_cfg)
        cfg["rebalance_freq"] = rebalance_freq
        cfg["target_count"] = int(target_count)
        cfg["hold_buffer_mult"] = float(hold_buffer_mult)
        cfg["turnover_budget"] = float(turnover_budget)
        cfg["max_position_weight"] = float(max_pos)
        cfg["name"] = (
            f"momoq_opt_rf{rebalance_freq}_n{target_count}_hb{hold_buffer_mult:.2f}"
            f"_tb{turnover_budget:.2f}_cap{max_pos:.3f}_ro"
            f"{'none' if risk_off is None else str(risk_off).replace('.', '')}"
        )

        if risk_off is None:
            cfg["market_regime"] = {"enabled": False}
        else:
            cfg["market_regime"] = {
                "enabled": True,
                "symbol": "SPY",
                "ma_days": 200,
                "risk_off_scalar": float(risk_off),
            }
        candidates.append(cfg)
    return candidates


def _pct(value: Any) -> float:
    v = _safe_float(value, float("nan"))
    return v * 100.0 if np.isfinite(v) else float("nan")


def _score_stage1(m20: Dict[str, Any], m35: Dict[str, Any]) -> float:
    c20 = _pct(m20.get("cagr"))
    c35 = _pct(m35.get("cagr"))
    dd20 = _safe_float(m20.get("max_drawdown_pct"), 0.0) * 100.0
    t20 = _safe_float(m20.get("avg_annual_turnover_pct"), 999.0)
    if not np.isfinite(c20):
        c20 = -999.0
    if not np.isfinite(c35):
        c35 = -999.0
    penalty = max(0.0, dd20 - 28.0) * 1.75 + max(0.0, t20 - 200.0) * 0.10
    return (0.60 * c20) + (0.40 * c35) - penalty


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize factor config variants against CAGR/DD/turnover gates")
    parser.add_argument("--base-config", default="config/factor_momo_quality_v1.json")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--output", default="")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0)
    parser.add_argument("--frictions", default="20,35")
    parser.add_argument("--top-stage2", type=int, default=20)
    parser.add_argument("--min-universe-coverage", type=float, default=0.60)
    parser.add_argument("--min-daily-membership-coverage", type=float, default=0.75)
    args = parser.parse_args()

    base_path = Path(args.base_config).expanduser().resolve()
    base_cfg = _load_config(base_path)

    start_date = str(args.start)
    end_date = str(args.end)
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError(f"PIT Russell 3000 universe is required, got source={source}")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})

    days = _days_for_range(start_date, end_date, warmup_days=420)
    print(f"Loading Russell 3000 PIT union: {len(symbols)} symbols ({start_date} -> {end_date}, days={days})")
    data = fetch_data_pack(symbols, days=days, backtest_mode=bool(args.cache_only)) or {}
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    loaded_symbols = sorted(list(data.keys()))
    coverage = (len(loaded_symbols) / float(len(symbols))) if symbols else 0.0
    print(f"Loaded symbols with price history: {len(loaded_symbols)} / {len(symbols)} ({coverage:.1%})")
    if coverage < float(args.min_universe_coverage):
        raise RuntimeError(
            f"Universe coverage too low ({coverage:.1%} < {float(args.min_universe_coverage):.1%}). "
            "Refusing to run reduced-universe optimization."
        )

    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=global_data)
    all_dates = list(getattr(prepared, "all_dates", []))
    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates, allow_missing_days=False)
    if not membership_by_day or len(membership_by_day) != len(all_dates):
        raise RuntimeError("PIT day membership unavailable/misaligned. Failing closed.")

    features = _extract_feature_arrays(prepared, membership_by_day)
    coverage_stats = _daily_membership_price_coverage(features)
    mean_daily_coverage = _safe_float(coverage_stats.get("mean"), 0.0)
    print(f"Daily PIT membership price coverage (mean): {mean_daily_coverage:.1%}")
    if mean_daily_coverage < float(args.min_daily_membership_coverage):
        raise RuntimeError(
            f"Daily PIT coverage too low ({mean_daily_coverage:.1%} < {float(args.min_daily_membership_coverage):.1%}). "
            "Refusing to run biased optimization."
        )

    prices = pd.DataFrame(features["close"], index=features["dates"], columns=features["symbols"])
    eps_yoy_df = pd.DataFrame(features["eps_yoy"], index=features["dates"], columns=features["symbols"])
    sales_yoy_df = pd.DataFrame(features["sales_yoy"], index=features["dates"], columns=features["symbols"])
    sector_map = _load_symbol_map(ROOT / "config" / "sectors.json")
    industry_map = _load_symbol_map(ROOT / "config" / "industries.json")
    if not industry_map:
        industry_map = _load_symbol_map(ROOT / "config" / "industry.json")

    # We intentionally keep signal weights fixed in this optimizer and sweep construction/risk controls.
    momentum_scores = _build_momentum_quality_scores(features, base_cfg)
    base_strategy_type = str(base_cfg.get("strategy_type", "cross_sectional_momentum") or "cross_sectional_momentum").lower()
    low_vol_scores = _build_low_vol_scores(features, base_cfg) if base_strategy_type == "momentum_low_vol_blend" else None

    friction_vals: List[float] = []
    for part in str(args.frictions or "").split(","):
        txt = part.strip()
        if not txt:
            continue
        val = _safe_float(txt, float("nan"))
        if np.isfinite(val) and val >= 0:
            friction_vals.append(float(val))
    if not friction_vals:
        friction_vals = [20.0, 35.0]
    scenarios = {
        int(v): FrictionScenario(
            name=f"{int(v)}x{int(v)}",
            transaction_cost_bps=float(args.transaction_cost_bps),
            entry_slippage_bps=float(v),
            exit_slippage_bps=float(v),
        )
        for v in friction_vals
    }
    if 20 not in scenarios:
        scenarios[20] = FrictionScenario("20x20", float(args.transaction_cost_bps), 20.0, 20.0)
    if 35 not in scenarios:
        scenarios[35] = FrictionScenario("35x35", float(args.transaction_cost_bps), 35.0, 35.0)

    windows_36_12 = _build_test_windows(start_date, end_date, train_months=36, test_months=12)
    windows_60_12 = _build_test_windows(start_date, end_date, train_months=60, test_months=12)

    candidates = _candidate_grid(base_cfg)
    stage1_rows: List[Dict[str, Any]] = []
    print(f"Stage 1: evaluating {len(candidates)} candidates")

    for idx, cfg in enumerate(candidates, start=1):
        risk_scalar = _build_market_risk_scalar(global_data, prices.index, cfg)
        full20 = _run_factor_window(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=None,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar,
            sector_map=sector_map,
            industry_map=industry_map,
            start_date=start_date,
            end_date=end_date,
            friction=scenarios[20],
        )
        full35 = _run_factor_window(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=None,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar,
            sector_map=sector_map,
            industry_map=industry_map,
            start_date=start_date,
            end_date=end_date,
            friction=scenarios[35],
        )
        score = _score_stage1(full20, full35)
        stage1_rows.append(
            {
                "cfg": cfg,
                "stage1_score": score,
                "full20": full20,
                "full35": full35,
            }
        )
        if idx % 20 == 0 or idx == len(candidates):
            print(f"  evaluated {idx}/{len(candidates)}")

    stage1_rows.sort(key=lambda x: float(x.get("stage1_score", -9999.0)), reverse=True)
    top_n = max(1, int(args.top_stage2))
    stage2_pool = stage1_rows[:top_n]
    print(f"Stage 2: stitched OOS on top {len(stage2_pool)} candidates")

    final_rows: List[Dict[str, Any]] = []
    for row in stage2_pool:
        cfg = row["cfg"]
        risk_scalar = _build_market_risk_scalar(global_data, prices.index, cfg)
        stitched20 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=None,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_60_12,
            test_months=12,
            friction=scenarios[20],
        )
        stitched35 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=None,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_60_12,
            test_months=12,
            friction=scenarios[35],
        )
        stitched36_20 = _stitch_test_windows(
            cfg=cfg,
            prices=prices,
            momentum_scores=momentum_scores,
            value_scores=None,
            low_vol_scores=low_vol_scores,
            eps_yoy_df=eps_yoy_df,
            sales_yoy_df=sales_yoy_df,
            risk_scalar_by_date=risk_scalar,
            sector_map=sector_map,
            industry_map=industry_map,
            windows=windows_36_12,
            test_months=12,
            friction=scenarios[20],
        )

        full20 = row["full20"]
        full35 = row["full35"]
        c20 = _pct(full20.get("cagr"))
        c35 = _pct(full35.get("cagr"))
        dd20 = _safe_float(full20.get("max_drawdown_pct"), 0.0) * 100.0
        turnover20 = _safe_float(full20.get("avg_annual_turnover_pct"), float("nan"))
        oos60_20 = _safe_float(stitched20.get("stitched_cagr_pct"), float("nan"))
        oos60_35 = _safe_float(stitched35.get("stitched_cagr_pct"), float("nan"))
        oos36_20 = _safe_float(stitched36_20.get("stitched_cagr_pct"), float("nan"))

        gates = {
            "oos60_20_ge_10": bool(np.isfinite(oos60_20) and oos60_20 >= 10.0),
            "oos60_35_ge_8": bool(np.isfinite(oos60_35) and oos60_35 >= 8.0),
            "dd20_le_28": bool(np.isfinite(dd20) and dd20 <= 28.0),
            "turnover20_le_200": bool(np.isfinite(turnover20) and turnover20 <= 200.0),
        }
        gate_count = sum(1 for v in gates.values() if v)

        final_score = (
            (gate_count * 1000.0)
            + (oos60_20 if np.isfinite(oos60_20) else -999.0) * 10.0
            + (oos60_35 if np.isfinite(oos60_35) else -999.0) * 8.0
            + (c20 if np.isfinite(c20) else -999.0) * 2.0
            - max(0.0, dd20 - 28.0) * 3.0
            - max(0.0, (turnover20 if np.isfinite(turnover20) else 300.0) - 200.0) * 0.5
        )

        final_rows.append(
            {
                "final_score": final_score,
                "gate_count": gate_count,
                "gates": gates,
                "params": {
                    "rebalance_freq": cfg.get("rebalance_freq"),
                    "target_count": cfg.get("target_count"),
                    "momentum_target_count": cfg.get("momentum_target_count"),
                    "low_vol_target_count": cfg.get("low_vol_target_count"),
                    "momentum_weight": cfg.get("momentum_weight"),
                    "low_vol_weight": cfg.get("low_vol_weight"),
                    "hold_buffer_mult": cfg.get("hold_buffer_mult"),
                    "turnover_budget": cfg.get("turnover_budget"),
                    "max_position_weight": cfg.get("max_position_weight"),
                    "market_regime": cfg.get("market_regime"),
                },
                "metrics": {
                    "full20_cagr_pct": c20,
                    "full35_cagr_pct": c35,
                    "full20_max_dd_pct": dd20,
                    "full20_avg_annual_turnover_pct": turnover20,
                    "oos36_20_cagr_pct": oos36_20,
                    "oos60_20_cagr_pct": oos60_20,
                    "oos60_35_cagr_pct": oos60_35,
                },
                "config": cfg,
            }
        )

    final_rows.sort(key=lambda x: float(x.get("final_score", -999999.0)), reverse=True)
    best = final_rows[0] if final_rows else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_config_path": str(base_path),
        "start_date": start_date,
        "end_date": end_date,
        "universe": {
            "name": "RUSSELL3000",
            "source": source,
            "membership_source": membership_source,
            "requested_symbol_count": len(symbols),
            "loaded_symbol_count": len(loaded_symbols),
            "coverage_ratio": coverage,
            "daily_membership_price_coverage": coverage_stats,
            "min_daily_membership_coverage": float(args.min_daily_membership_coverage),
            "cache_only": bool(args.cache_only),
        },
        "stage1_candidate_count": len(candidates),
        "stage2_candidate_count": len(stage2_pool),
        "best": best,
        "top10": final_rows[:10],
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else (
        ROOT / "tmp" / f"opt_factor_configs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written: {output_path}")
    if best:
        print(json.dumps({"best_params": best.get("params"), "best_metrics": best.get("metrics"), "gates": best.get("gates")}, indent=2))


if __name__ == "__main__":
    main()
