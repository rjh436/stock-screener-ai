#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    "growth_core5": ["CURE", "SOXL", "TNA", "TQQQ", "UPRO"],
    "mixed4": ["QQQ", "SMH", "SPY", "USD"],
    "broad8": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD", "FAS", "CURE"],
    "liquid6": ["TQQQ", "SOXL", "USD", "QQQ", "SMH", "SPY"],
}

POOL_PRESETS: Dict[str, List[str]] = {
    "leveraged11": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD", "FAS", "CURE", "DRN", "ERX", "YINN"],
    "leveraged13": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD", "FAS", "CURE", "DRN", "ERX", "YINN", "NAIL", "LABU"],
    "growth8": ["TQQQ", "SOXL", "TECL", "UPRO", "TNA", "USD", "FAS", "SMH"],
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

LOOKBACK_PRESETS: Dict[str, Dict[str, float]] = {
    "base_growth": {"21": 0.10, "63": 0.45, "126": 0.35, "252": 0.10},
    "growth_heavy63": {"21": 0.05, "63": 0.50, "126": 0.35, "252": 0.10},
    "growth_heavy21": {"21": 0.20, "63": 0.40, "126": 0.30, "252": 0.10},
    "growth_balanced": {"21": 0.15, "63": 0.35, "126": 0.35, "252": 0.15},
    "broad_v2": {"21": 0.50, "63": 0.30, "126": 0.15, "252": 0.05},
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


def _first_valid_date(frame: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if frame is None or frame.empty:
        return None
    idx = pd.to_datetime(frame.index, errors="coerce")
    if len(idx) == 0:
        return None
    idx = pd.DatetimeIndex(idx).tz_localize(None)
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return None
    return pd.Timestamp(idx.min()).normalize()


def _filter_symbols_by_history(
    data: Mapping[str, pd.DataFrame],
    symbols: Sequence[str],
    *,
    start_date: str,
    max_inception_lag_days: int,
) -> List[str]:
    start_ts = pd.Timestamp(start_date).normalize()
    max_lag = max(0, int(max_inception_lag_days or 0))
    cutoff = start_ts + pd.Timedelta(days=max_lag)
    out: List[str] = []
    for sym in symbols:
        first_dt = _first_valid_date(data.get(str(sym)))
        if first_dt is None:
            continue
        if first_dt <= cutoff:
            out.append(str(sym))
    return out


def _sample_combinations(
    symbols: Sequence[str],
    subset_size: int,
    *,
    max_subsets: int,
    seed: int,
) -> List[Tuple[str, ...]]:
    ordered = tuple(sorted({str(sym) for sym in symbols if str(sym).strip()}))
    if subset_size <= 0 or subset_size > len(ordered):
        return []
    total = math.comb(len(ordered), subset_size)
    if max_subsets <= 0 or total <= max_subsets:
        return [tuple(combo) for combo in itertools.combinations(ordered, subset_size)]

    rng = random.Random(seed + subset_size * 1009 + len(ordered) * 17)
    seen = set()
    out: List[Tuple[str, ...]] = []
    target = min(total, max_subsets)
    max_attempts = max(target * 20, 1000)
    attempts = 0
    while len(out) < target and attempts < max_attempts:
        attempts += 1
        combo = tuple(sorted(rng.sample(list(ordered), subset_size)))
        if combo in seen:
            continue
        seen.add(combo)
        out.append(combo)
    return sorted(out)


def _expand_universe_specs(
    *,
    universe_names: Sequence[str],
    candidate_pool_names: Sequence[str],
    subset_sizes: Sequence[int],
    max_subsets: int,
    seed: int,
    data: Mapping[str, pd.DataFrame],
    start_date: str,
    max_inception_lag_days: int,
) -> List[Tuple[str, List[str], Optional[str]]]:
    specs: List[Tuple[str, List[str], Optional[str]]] = []
    seen = set()

    for uname in universe_names:
        symbols = list(UNIVERSE_PRESETS.get(uname) or [])
        if not symbols:
            continue
        key = tuple(symbols)
        if key in seen:
            continue
        seen.add(key)
        specs.append((uname, symbols, None))

    clean_sizes = sorted({int(s) for s in subset_sizes if int(s) > 0})
    for pname in candidate_pool_names:
        base_symbols = list(POOL_PRESETS.get(pname) or [])
        if not base_symbols:
            continue
        eligible = _filter_symbols_by_history(
            data,
            base_symbols,
            start_date=start_date,
            max_inception_lag_days=max_inception_lag_days,
        )
        if not eligible:
            continue
        if not clean_sizes:
            key = tuple(eligible)
            if key in seen:
                continue
            seen.add(key)
            specs.append((pname, eligible, pname))
            continue
        for size in clean_sizes:
            for idx, combo in enumerate(
                _sample_combinations(eligible, size, max_subsets=max_subsets, seed=seed),
                start=1,
            ):
                combo_list = list(combo)
                key = tuple(combo_list)
                if key in seen:
                    continue
                seen.add(key)
                name = f"{pname}_s{size}_{idx:03d}"
                specs.append((name, combo_list, pname))
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local ETF rotation grid search.")
    parser.add_argument("--base-config", default=str(ROOT / "config" / "etf_rotation_3x_growth_v1.json"))
    parser.add_argument("--universes", default="growth6,mixed4,broad8")
    parser.add_argument("--candidate-pools", default="")
    parser.add_argument("--subset-sizes", default="")
    parser.add_argument("--max-subsets", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-inception-lag-days", type=int, default=365)
    parser.add_argument("--regimes", default="ma_only,base")
    parser.add_argument("--freqs", default="W,M")
    parser.add_argument("--targets", default="1,2")
    parser.add_argument("--turnover-budgets", default="1.0,0.5")
    parser.add_argument("--require-fast", default="true,false")
    parser.add_argument("--abs-lookbacks", default="63,126")
    parser.add_argument("--base-gross-values", default="1.0")
    parser.add_argument("--vol-target-values", default="0.0")
    parser.add_argument("--lookback-presets", default="")
    parser.add_argument("--start-date", default="2011-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--out", default="tmp/etf_rotation_grid.csv")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = _load_config(Path(args.base_config).resolve())
    base["prefer_yfinance"] = True

    universe_names = [u.strip() for u in str(args.universes).split(",") if u.strip()]
    candidate_pool_names = [u.strip() for u in str(args.candidate_pools).split(",") if u.strip()]
    subset_sizes = _parse_csv_list(args.subset_sizes, int)
    regime_names = [r.strip() for r in str(args.regimes).split(",") if r.strip()]
    freqs = [f.strip().upper() for f in str(args.freqs).split(",") if f.strip()]
    targets = _parse_csv_list(args.targets, int)
    turnover_budgets = _parse_csv_list(args.turnover_budgets, float)
    require_fast = _parse_bool_list(args.require_fast)
    abs_lookbacks = _parse_csv_list(args.abs_lookbacks, int)
    base_gross_values = _parse_csv_list(args.base_gross_values, float)
    vol_target_values = _parse_csv_list(args.vol_target_values, float)
    lookback_preset_names = [s.strip() for s in str(args.lookback_presets).split(",") if s.strip()]
    if not lookback_preset_names:
        lookback_preset_names = ["__base__"]
    if not base_gross_values:
        base_gross_values = [1.0]
    if not vol_target_values:
        vol_target_values = [0.0]

    requested_symbols = sorted(
        {sym for name in universe_names for sym in UNIVERSE_PRESETS.get(name, [])}
        | {sym for name in candidate_pool_names for sym in POOL_PRESETS.get(name, [])}
        | {"SPY", "VIX", "HYG", "LQD"}
    )
    data = _download_yfinance_pack(requested_symbols, start_date=str(args.start_date), end_date=str(args.end_date))
    windows24 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=24, test_months=12)
    windows60 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=60, test_months=12)
    universe_specs = _expand_universe_specs(
        universe_names=universe_names,
        candidate_pool_names=candidate_pool_names,
        subset_sizes=subset_sizes,
        max_subsets=int(args.max_subsets),
        seed=int(args.seed),
        data=data,
        start_date=str(args.start_date),
        max_inception_lag_days=int(args.max_inception_lag_days),
    )

    rows: List[Dict[str, Any]] = []
    for uname, symbols, candidate_pool in universe_specs:
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
                                for base_gross in base_gross_values:
                                    for vol_target in vol_target_values:
                                        for lookback_preset in lookback_preset_names:
                                            cfg = copy.deepcopy(base)
                                            cfg["symbols"] = list(symbols)
                                            cfg["rebalance_freq"] = freq
                                            cfg["target_count"] = int(target)
                                            cfg["turnover_budget"] = float(tb)
                                            cfg["require_fast_above_trend"] = bool(fast)
                                            cfg["absolute_momentum_lookback"] = int(abs_lb)
                                            cfg["market_regime"] = copy.deepcopy(regime)
                                            if lookback_preset != "__base__":
                                                lb_map = LOOKBACK_PRESETS.get(lookback_preset)
                                                if not lb_map:
                                                    continue
                                                cfg["lookbacks"] = dict(lb_map)
                                            exposure_cfg: Dict[str, Any] = {
                                                "base_gross_exposure": float(base_gross),
                                                "proxy_symbols": list(symbols),
                                                "max_gross_exposure": 1.0,
                                            }
                                            if float(vol_target) > 0.0:
                                                exposure_cfg["vol_target_annual"] = float(vol_target)
                                                exposure_cfg["vol_lookback_days"] = 21
                                                exposure_cfg["min_gross_exposure"] = 0.20
                                            cfg["exposure_control"] = exposure_cfg
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
                                                "candidate_pool": candidate_pool or "",
                                                "symbols": ",".join(symbols),
                                                "symbol_count": int(len(symbols)),
                                                "regime": regime_name,
                                                "freq": freq,
                                                "target": int(target),
                                                "turnover_budget": float(tb),
                                                "require_fast": bool(fast),
                                                "abs_lookback": int(abs_lb),
                                                "base_gross_exposure": float(base_gross),
                                                "vol_target_annual": float(vol_target),
                                                "lookback_preset": "" if lookback_preset == "__base__" else lookback_preset,
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
