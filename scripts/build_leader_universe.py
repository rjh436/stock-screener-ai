#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.custom_universe import load_symbol_file
from data.universe import build_russell3000_membership_by_day


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a leader-preserving triage universe from PIT Russell 3000 membership and cached RS data."
    )
    parser.add_argument("--cache", default="data/cache_indicators.pkl")
    parser.add_argument("--start-date", default="2016-03-02")
    parser.add_argument("--end-date", default="2026-03-02")
    parser.add_argument(
        "--leader-pct",
        type=float,
        default=20.0,
        help="At each month-end, keep the top X percent of PIT members by rs_rating.",
    )
    parser.add_argument(
        "--monthly-top-n",
        type=int,
        default=0,
        help="Optional fixed top-N override per month-end. When > 0 it takes precedence over --leader-pct.",
    )
    parser.add_argument("--min-rs-rating", type=float, default=80.0)
    parser.add_argument("--min-volume", type=float, default=0.0)
    parser.add_argument(
        "--liquidity-anchor-count",
        type=int,
        default=200,
        help="Add the top N symbols by trailing average volume at the end date.",
    )
    parser.add_argument("--liquidity-lookback-days", type=int, default=63)
    parser.add_argument(
        "--seed-symbol-file",
        action="append",
        default=[],
        help="Optional symbol file to union into the result. Repeatable.",
    )
    parser.add_argument(
        "--seed-symbol",
        action="append",
        default=[],
        help="Optional symbol to union into the result. Repeatable.",
    )
    parser.add_argument("--out-symbols", default="tmp/leader_triage_universe_20260319.txt")
    parser.add_argument("--out-summary", default="tmp/leader_triage_universe_20260319.json")
    return parser.parse_args()


def _load_prepared(cache_path: Path) -> Any:
    with cache_path.open("rb") as handle:
        return pickle.load(handle)


def _month_end_indices(all_dates: Sequence[Any], start_date: str, end_date: str) -> List[int]:
    idx = pd.DatetimeIndex(pd.to_datetime(list(all_dates)))
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    in_window = (idx >= start_ts) & (idx <= end_ts)
    if not in_window.any():
        return []

    positions = np.flatnonzero(in_window)
    by_period: Dict[pd.Period, int] = {}
    for pos in positions:
        by_period[idx[int(pos)].to_period("M")] = int(pos)
    return [by_period[key] for key in sorted(by_period)]


def _symbol_snapshot(
    sym_data: Any,
    global_idx: int,
) -> tuple[float, float]:
    gidx = getattr(sym_data, "gidx", None)
    rsrating = getattr(sym_data, "rsrating", None)
    if gidx is None or rsrating is None:
        return float("nan"), float("nan")

    pos = int(np.searchsorted(gidx, int(global_idx)))
    if pos >= len(gidx) or int(gidx[pos]) != int(global_idx):
        return float("nan"), float("nan")

    rs_val = float(rsrating[pos]) if pos < len(rsrating) else float("nan")
    volume = float("nan")
    df = getattr(sym_data, "df", None)
    if df is not None and pos < len(df.index):
        raw_volume = df.iloc[pos].get("volume", np.nan)
        volume = float(raw_volume) if pd.notna(raw_volume) else float("nan")
    return rs_val, volume


def _avg_volume_as_of(sym_data: Any, global_idx: int, lookback_days: int) -> float:
    gidx = getattr(sym_data, "gidx", None)
    df = getattr(sym_data, "df", None)
    if gidx is None or df is None or lookback_days <= 0:
        return float("nan")

    pos = int(np.searchsorted(gidx, int(global_idx)))
    if pos >= len(gidx):
        pos = len(gidx) - 1
    elif int(gidx[pos]) != int(global_idx):
        pos -= 1
    if pos < 0:
        return float("nan")

    start = max(0, pos - int(lookback_days) + 1)
    window = pd.to_numeric(df.iloc[start : pos + 1]["volume"], errors="coerce")
    if window.empty:
        return float("nan")
    value = float(window.mean())
    return value if math.isfinite(value) else float("nan")


def _load_seed_symbols(files: Iterable[str], raw_symbols: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []

    def _append(value: object) -> None:
        sym = str(value or "").strip().upper()
        if not sym or sym in seen:
            return
        seen.add(sym)
        out.append(sym)

    for file_path in files:
        for sym in load_symbol_file(file_path):
            _append(sym)
    for sym in raw_symbols:
        _append(sym)
    return out


def main() -> None:
    args = _parse_args()
    cache_path = (ROOT / args.cache).resolve()
    prepared = _load_prepared(cache_path)
    raw_all_dates = getattr(prepared, "all_dates", None)
    all_dates = list(raw_all_dates) if raw_all_dates is not None else []
    if not all_dates:
        raise RuntimeError("Prepared cache did not contain any all_dates values.")

    monthly_indices = _month_end_indices(all_dates, str(args.start_date), str(args.end_date))
    if not monthly_indices:
        raise RuntimeError("No month-end dates were available for the requested window.")

    membership_by_day, membership_source = build_russell3000_membership_by_day(all_dates)
    if not membership_by_day:
        raise RuntimeError("PIT Russell 3000 membership was unavailable for the prepared cache calendar.")

    seed_symbols = _load_seed_symbols(args.seed_symbol_file, args.seed_symbol)
    selected_symbols = set(seed_symbols)
    month_stats: List[Dict[str, Any]] = []
    seen_counts: Dict[str, int] = defaultdict(int)

    prepared_symbols = getattr(prepared, "enriched", {}) or {}
    for global_idx in monthly_indices:
        membership = membership_by_day[int(global_idx)]
        if not membership:
            continue

        ranked: List[tuple[str, float, float]] = []
        for symbol in sorted(str(sym).upper() for sym in membership):
            sym_data = prepared_symbols.get(symbol)
            if sym_data is None:
                continue
            rs_val, volume = _symbol_snapshot(sym_data, int(global_idx))
            if not math.isfinite(rs_val) or rs_val < float(args.min_rs_rating):
                continue
            if math.isfinite(volume) and volume < float(args.min_volume):
                continue
            ranked.append((symbol, rs_val, volume))

        ranked.sort(key=lambda item: (item[1], item[2] if math.isfinite(item[2]) else -1.0), reverse=True)
        if not ranked:
            continue

        if int(args.monthly_top_n) > 0:
            keep_count = min(len(ranked), int(args.monthly_top_n))
        else:
            keep_count = max(1, int(math.ceil(len(ranked) * float(args.leader_pct) / 100.0)))

        chosen = ranked[:keep_count]
        for symbol, _rs, _vol in chosen:
            selected_symbols.add(symbol)
            seen_counts[symbol] += 1

        top_rs = float(chosen[0][1]) if chosen else float("nan")
        cutoff_rs = float(chosen[-1][1]) if chosen else float("nan")
        month_stats.append(
            {
                "date": pd.Timestamp(all_dates[int(global_idx)]).date().isoformat(),
                "membership_count": int(len(membership)),
                "eligible_count": int(len(ranked)),
                "selected_count": int(len(chosen)),
                "top_rs_rating": top_rs,
                "cutoff_rs_rating": cutoff_rs,
            }
        )

    liquidity_anchors: List[Dict[str, Any]] = []
    anchor_idx = monthly_indices[-1]
    if int(args.liquidity_anchor_count) > 0:
        ranked_liquidity: List[tuple[str, float]] = []
        for symbol, sym_data in prepared_symbols.items():
            avg_volume = _avg_volume_as_of(sym_data, int(anchor_idx), int(args.liquidity_lookback_days))
            if not math.isfinite(avg_volume):
                continue
            ranked_liquidity.append((str(symbol).upper(), avg_volume))
        ranked_liquidity.sort(key=lambda item: item[1], reverse=True)
        for symbol, avg_volume in ranked_liquidity[: int(args.liquidity_anchor_count)]:
            selected_symbols.add(symbol)
            liquidity_anchors.append({"symbol": symbol, "avg_volume": avg_volume})

    symbol_list = sorted(selected_symbols)
    out_symbols = (ROOT / args.out_symbols).resolve()
    out_summary = (ROOT / args.out_summary).resolve()
    out_symbols.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    out_symbols.write_text("\n".join(symbol_list) + "\n", encoding="utf-8")
    summary = {
        "cache_path": str(cache_path),
        "membership_source": membership_source,
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "monthly_windows": int(len(month_stats)),
        "leader_pct": float(args.leader_pct),
        "monthly_top_n": int(args.monthly_top_n),
        "min_rs_rating": float(args.min_rs_rating),
        "min_volume": float(args.min_volume),
        "liquidity_anchor_count": int(args.liquidity_anchor_count),
        "liquidity_lookback_days": int(args.liquidity_lookback_days),
        "seed_symbols_count": int(len(seed_symbols)),
        "selected_symbol_count": int(len(symbol_list)),
        "selected_symbols_path": str(out_symbols),
        "top_seen_symbols": [
            {"symbol": symbol, "month_count": count}
            for symbol, count in sorted(seen_counts.items(), key=lambda item: (-item[1], item[0]))[:50]
        ],
        "liquidity_anchors": liquidity_anchors[:50],
        "month_stats": month_stats,
    }
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"saved {out_symbols}")
    print(f"saved {out_summary}")


if __name__ == "__main__":
    main()
