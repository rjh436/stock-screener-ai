#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols_pit_window_with_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill local price cache for a PIT universe.")
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3200)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--targets", default="missing,incomplete")
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _resolve_symbols(universe: str, start_date: str, end_date: str) -> Tuple[List[str], str]:
    name = str(universe).strip().upper()
    if name != "RUSSELL3000":
        raise ValueError(f"Unsupported universe for cache backfill: {universe}")
    symbols, source = get_universe_symbols_pit_window_with_meta("RUSSELL3000", start_date, end_date)
    return list(symbols), str(source)


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for raw in items:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _select_target_symbols(quality_report: Mapping[str, Any], target_modes: Sequence[str]) -> List[str]:
    selected: List[str] = []
    modes = {str(mode).strip().lower() for mode in target_modes if str(mode).strip()}
    if "missing" in modes:
        selected.extend(quality_report.get("missing_symbols", []) or [])
    if "incomplete" in modes:
        selected.extend(quality_report.get("incomplete_symbols", []) or [])
    if "stale" in modes:
        selected.extend(quality_report.get("stale_symbols", []) or [])
    return _dedupe(selected)


def _quality_summary(report: Mapping[str, Any]) -> Dict[str, int]:
    return {
        "requested": int(report.get("requested", 0) or 0),
        "loaded": int(report.get("loaded", 0) or 0),
        "missing": int(report.get("missing", 0) or 0),
        "incomplete_history": int(report.get("incomplete_history", 0) or 0),
        "stale": int(report.get("stale", 0) or 0),
    }


def main() -> None:
    args = _parse_args()
    symbols, source = _resolve_symbols(args.universe, args.start_date, args.end_date)
    print(
        f"cache-backfill: universe={args.universe} source={source} "
        f"requested_symbols={len(symbols)}"
    )

    before_quality: Dict[str, Any] = {}
    fetch_data_pack(symbols, days=int(args.days), backtest_mode=True, quality_report=before_quality)
    target_modes = [part.strip().lower() for part in str(args.targets).split(",") if part.strip()]
    targets = _select_target_symbols(before_quality, target_modes)
    print(
        f"before: loaded={before_quality.get('loaded', 0)} "
        f"missing={before_quality.get('missing', 0)} "
        f"incomplete={before_quality.get('incomplete_history', 0)} "
        f"stale={before_quality.get('stale', 0)} "
        f"target_symbols={len(targets)}"
    )

    batch_size = max(1, int(args.batch_size or 1))
    max_batches = max(0, int(args.max_batches or 0))
    batch_reports: List[Dict[str, Any]] = []
    if targets:
        for batch_num, start_idx in enumerate(range(0, len(targets), batch_size), start=1):
            if max_batches and batch_num > max_batches:
                break
            batch = targets[start_idx : start_idx + batch_size]
            q: Dict[str, Any] = {}
            print(f"batch {batch_num}: backfilling {len(batch)} symbols")
            fetch_data_pack(
                batch,
                days=int(args.days),
                backtest_mode=False,
                force_fresh=bool(args.force_fresh),
                max_workers=int(args.max_workers),
                quality_report=q,
            )
            batch_reports.append(
                {
                    "batch": batch_num,
                    "requested": list(batch),
                    "quality": _quality_summary(q),
                }
            )

    after_quality: Dict[str, Any] = {}
    fetch_data_pack(symbols, days=int(args.days), backtest_mode=True, quality_report=after_quality)
    payload = {
        "universe": str(args.universe),
        "universe_source": source,
        "start_date": str(args.start_date),
        "end_date": str(args.end_date),
        "days": int(args.days),
        "targets": target_modes,
        "force_fresh": bool(args.force_fresh),
        "before": _quality_summary(before_quality),
        "after": _quality_summary(after_quality),
        "target_symbols": int(len(targets)),
        "batches_run": int(len(batch_reports)),
        "batch_reports": batch_reports,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
