#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

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


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _to_pct_dd(raw_dd: Any) -> float:
    dd = _safe_float(raw_dd, float("nan"))
    if np.isfinite(dd) and dd <= 1.0:
        dd *= 100.0
    return dd


def _enforce_cash_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["allow_margin"] = False
    out["max_total_exposure_pct_bull"] = min(1.0, max(0.0, _safe_float(out.get("max_total_exposure_pct_bull"), 1.0)))
    out["max_total_exposure_pct_bear"] = min(1.0, max(0.0, _safe_float(out.get("max_total_exposure_pct_bear"), 1.0)))
    out["max_pos_size_pct"] = min(1.0, max(0.0, _safe_float(out.get("max_pos_size_pct"), 0.25)))
    out["vcp_max_pos_size_pct"] = min(out["max_pos_size_pct"], max(0.0, _safe_float(out.get("vcp_max_pos_size_pct"), out["max_pos_size_pct"])))
    out["ep_max_pos_size_pct"] = min(out["max_pos_size_pct"], max(0.0, _safe_float(out.get("ep_max_pos_size_pct"), out["max_pos_size_pct"])))
    return out


def _combo_space() -> Dict[str, List[Any]]:
    return {
        "trend_template_mode": ["strict", "classic"],
        "prior_runup_min_pct": [20, 25, 30],
        "adr_min_pct": [2.5, 3.0, 3.5],
        "vcp_lookback_bars": [30, 40, 60],
        "vcp_breakout_volume_mult": [1.75, 2.0],
        "min_entry_score": [20, 25, 30],
        "rs_percentile_min": [70, 75, 80],
        "max_stop_pct": [0.06, 0.08],
        "time_stop_days": [40, 60, 80],
    }


def _iter_combos(space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(space.keys())
    for vals in itertools.product(*(space[k] for k in keys)):
        yield {k: v for k, v in zip(keys, vals)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast cached-data sweep for hold60 breakout parameter variants")
    parser.add_argument("--base-config", default=str(ROOT / "config" / "superperformance_practical_no_leverage_optimized_hold60.json"))
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--universe-limit", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-combos", type=int, default=64)
    parser.add_argument("--output", default=str(ROOT / "tmp" / "sweep_hold60_params_results.json"))
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base_cfg = json.load(f)
    base_cfg = _enforce_cash_only(base_cfg)
    base_cfg["transaction_cost_bps"] = 2.0
    base_cfg["entry_slippage_bps"] = 10.0
    base_cfg["exit_slippage_bps"] = 10.0
    base_cfg["slippage_bps"] = 10.0
    base_cfg["multi_sleeve_enabled"] = False
    base_cfg["entry_mode"] = "both"
    base_cfg["use_market_regime_traffic_light"] = False
    base_cfg["use_market_breadth_overlay"] = False
    base_cfg["market_exposure_mode"] = "exposure"
    base_cfg["bear_cash_mode"] = "off"

    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", args.start, args.end)
    if source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
        raise RuntimeError(f"PIT universe required; got source={source}")
    symbols = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if args.universe_limit > 0 and args.universe_limit < len(symbols):
        rng = np.random.default_rng(int(args.sample_seed))
        selected_idx = np.sort(rng.choice(len(symbols), size=int(args.universe_limit), replace=False))
        symbols = [symbols[int(i)] for i in selected_idx]

    days = max(1, (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days) + 420
    print(f"Loading data for {len(symbols)} symbols (days={days})...")
    data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}
    prepared = prepare_backtest_data(data, symbols, start_date=args.start, global_data=g_data)
    membership_by_day, membership_source = build_russell3000_membership_by_day(
        list(getattr(prepared, "all_dates", [])),
        allow_missing_days=False,
    )
    print(f"PIT membership source: {membership_source}")

    combos = list(_iter_combos(_combo_space()))
    rng = random.Random(int(args.sample_seed))
    rng.shuffle(combos)
    if args.max_combos > 0:
        combos = combos[: int(args.max_combos)]

    rows: List[Dict[str, Any]] = []
    for idx, tweak in enumerate(combos, start=1):
        cfg = dict(base_cfg)
        cfg.update(tweak)
        strat = SuperperformanceStrategy(cfg)
        result = run_backtest(
            [strat],
            prepared,
            start_cash=100000.0,
            start_date=args.start,
            end_date=args.end,
            global_data=g_data,
            universe_membership_by_day=membership_by_day,
            require_pit_membership=True,
        )
        out = result[0] if isinstance(result, list) else result
        if not isinstance(out, dict):
            continue
        cagr_pct = _safe_float(out.get("cagr"), float("nan"))
        if np.isfinite(cagr_pct):
            cagr_pct *= 100.0
        dd_pct = _to_pct_dd(out.get("max_drawdown_pct"))
        trades = int(out.get("total_trades", 0) or 0)
        final_value = _safe_float(out.get("final_value"), float("nan"))
        row = {
            "rank_hint": idx,
            "cagr_pct": float(cagr_pct) if np.isfinite(cagr_pct) else float("nan"),
            "max_dd_pct": float(dd_pct) if np.isfinite(dd_pct) else float("nan"),
            "trades": trades,
            "final_value": float(final_value) if np.isfinite(final_value) else float("nan"),
            "params": tweak,
        }
        rows.append(row)
        print(
            f"[{idx:03d}/{len(combos):03d}] CAGR={row['cagr_pct']:.2f}% DD={row['max_dd_pct']:.2f}% "
            f"Trades={trades} params={tweak}"
        )

    rows.sort(key=lambda r: (_safe_float(r.get("cagr_pct"), -1e9), -_safe_float(r.get("max_dd_pct"), 1e9)), reverse=True)
    out_obj = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "start_date": args.start,
        "end_date": args.end,
        "universe_count": len(symbols),
        "universe_source": source,
        "membership_source": membership_source,
        "max_combos": len(combos),
        "results": rows,
    }
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"Wrote sweep results to: {out_path}")


if __name__ == "__main__":
    main()
