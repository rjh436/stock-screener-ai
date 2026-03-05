#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_etf_rotation_walkforward import (
    _build_rotation_scores,
    _download_yfinance_pack,
    _load_config,
    _price_frame_from_data,
    _run_rotation_window,
)
from scripts.run_factor_walkforward import _build_test_windows, _pct_dd, _safe_float, _stitch_test_windows_warm


UNIVERSE_PRESETS: Dict[str, List[str]] = {
    "growth6": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD"],
    "growth5": ["TQQQ", "SOXL", "TECL", "UPRO", "USD"],
    "mixed4": ["QQQ", "SMH", "SPY", "USD"],
    "broad8": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD", "FAS", "CURE"],
    "liquid6": ["TQQQ", "SOXL", "USD", "QQQ", "SMH", "SPY"],
}

REGIME_PRESETS: Dict[str, Dict[str, Any]] = {
    "base": {
        "enabled": True,
        "symbol": "SPY",
        "ma_days": 200,
        "risk_off_scalar": 0.0,
        "use_vix_overlay": True,
        "use_credit_overlay": True,
        "overlay_weight": 0.35,
    },
    "ma_only": {
        "enabled": True,
        "symbol": "SPY",
        "ma_days": 200,
        "risk_off_scalar": 0.0,
        "use_vix_overlay": False,
        "use_credit_overlay": False,
        "overlay_weight": 0.0,
    },
    "off": {"enabled": False},
}


def _parse_csv_list(raw: str, cast=int) -> List[Any]:
    out: List[Any] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        out.append(cast(token))
    return out


def _parse_bool_list(raw: str) -> List[bool]:
    mapping = {"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False}
    out: List[bool] = []
    for part in str(raw or "").split(","):
        token = part.strip().lower()
        if token in mapping:
            out.append(mapping[token])
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local ETF rotation grid search.")
    parser.add_argument("--base-config", default=str(ROOT / "config" / "etf_rotation_3x_growth_v1.json"))
    parser.add_argument("--universes", default="growth6,mixed4,broad8")
    parser.add_argument("--regimes", default="ma_only,base")
    parser.add_argument("--freqs", default="W,M")
    parser.add_argument("--targets", default="1,2")
    parser.add_argument("--turnover-budgets", default="1.0,0.5")
    parser.add_argument("--require-fast", default="true,false")
    parser.add_argument("--abs-lookbacks", default="63,126")
    parser.add_argument("--start-date", default="2011-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--out", default="tmp/etf_rotation_grid.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = _load_config(Path(args.base_config).resolve())
    base["prefer_yfinance"] = True

    universe_names = [u.strip() for u in str(args.universes).split(",") if u.strip()]
    regime_names = [r.strip() for r in str(args.regimes).split(",") if r.strip()]
    freqs = [f.strip().upper() for f in str(args.freqs).split(",") if f.strip()]
    targets = _parse_csv_list(args.targets, int)
    turnover_budgets = _parse_csv_list(args.turnover_budgets, float)
    require_fast = _parse_bool_list(args.require_fast)
    abs_lookbacks = _parse_csv_list(args.abs_lookbacks, int)

    requested_symbols = sorted({sym for name in universe_names for sym in UNIVERSE_PRESETS.get(name, [])} | {"SPY", "VIX", "HYG", "LQD"})
    data = _download_yfinance_pack(requested_symbols, start_date=str(args.start_date), end_date=str(args.end_date))
    windows24 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=24, test_months=12)
    windows60 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=60, test_months=12)

    rows: List[Dict[str, Any]] = []
    for uname in universe_names:
        symbols = UNIVERSE_PRESETS.get(uname)
        if not symbols:
            continue
        close = _price_frame_from_data(data, symbols, "close")
        open_px = _price_frame_from_data(data, symbols, "open").reindex(close.index)
        global_data = {k: data[k] for k in ("SPY", "VIX", "HYG", "LQD") if k in data}
        if close.empty or open_px.empty:
            continue

        for regime_name in regime_names:
            regime = REGIME_PRESETS.get(regime_name)
            if regime is None:
                continue
            for freq in freqs:
                for target in targets:
                    for tb in turnover_budgets:
                        for fast in require_fast:
                            for abs_lb in abs_lookbacks:
                                cfg = copy.deepcopy(base)
                                cfg["symbols"] = list(symbols)
                                cfg["rebalance_freq"] = freq
                                cfg["target_count"] = int(target)
                                cfg["turnover_budget"] = float(tb)
                                cfg["require_fast_above_trend"] = bool(fast)
                                cfg["absolute_momentum_lookback"] = int(abs_lb)
                                cfg["market_regime"] = copy.deepcopy(regime)
                                scores = _build_rotation_scores(close, cfg)
                                full = _run_rotation_window(
                                    cfg=cfg,
                                    close_px=close,
                                    open_px=open_px,
                                    scores=scores,
                                    global_data=global_data,
                                    start_date=str(args.start_date),
                                    end_date=str(args.end_date),
                                )
                                o24 = _stitch_test_windows_warm(full_run=full, windows=windows24, test_months=12)
                                o60 = _stitch_test_windows_warm(full_run=full, windows=windows60, test_months=12)
                                full_cagr = _safe_float(full.get("cagr"), 0.0) * 100.0
                                full_dd = _pct_dd(full.get("max_drawdown_pct", 0.0))
                                row = {
                                    "universe": uname,
                                    "regime": regime_name,
                                    "freq": freq,
                                    "target": int(target),
                                    "turnover_budget": float(tb),
                                    "require_fast": bool(fast),
                                    "abs_lookback": int(abs_lb),
                                    "full_cagr_pct": full_cagr,
                                    "full_max_dd_pct": full_dd,
                                    "avg_annual_turnover_pct": _safe_float(full.get("avg_annual_turnover_pct"), float("nan")),
                                    "oos_24_12_cagr_pct_warm": _safe_float(o24.get("stitched_cagr_pct"), float("nan")),
                                    "oos_60_12_cagr_pct_warm": _safe_float(o60.get("stitched_cagr_pct"), float("nan")),
                                    "calmar": (full_cagr / full_dd) if full_dd > 0 else float("nan"),
                                }
                                rows.append(row)
                                print(json.dumps(row))

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_path.write_text("", encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
