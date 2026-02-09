#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _normalize_symbol(value: str) -> str:
    return str(value or "").strip().upper().replace(".", "-").replace("/", "-")


def _load_universe(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        raw = payload.get("symbols") or payload.get("tickers") or []
    else:
        raw = []

    out: List[str] = []
    seen = set()
    for item in raw:
        sym = _normalize_symbol(str(item))
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _load_partition_symbols(base_dir: Path) -> set[str]:
    out: set[str] = set()
    if not base_dir.exists():
        return out
    for child in base_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("ticker="):
            continue
        sym = _normalize_symbol(child.name.split("=", 1)[1])
        parquet_path = child / "fundamentals.parquet"
        if sym and parquet_path.exists():
            out.add(sym)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit EDGAR fundamentals coverage against universe file")
    parser.add_argument(
        "--universe-file",
        default="data/cache_indices/russell3000_iwv.json",
        help="Universe symbols JSON file",
    )
    parser.add_argument(
        "--fund-dir",
        default="data/fundamentals/edgar_income",
        help="EDGAR fundamentals partition root",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=25,
        help="Show up to N missing symbols",
    )
    args = parser.parse_args()

    universe_path = Path(args.universe_file)
    fund_dir = Path(args.fund_dir)

    universe = _load_universe(universe_path)
    loaded = _load_partition_symbols(fund_dir)

    if not universe:
        print(f"Universe empty or unreadable: {universe_path}")
        return 2

    covered = [s for s in universe if s in loaded]
    missing = [s for s in universe if s not in loaded]
    coverage_pct = (len(covered) / float(len(universe))) * 100.0

    print(f"Universe symbols: {len(universe)}")
    print(f"Loaded fundamentals: {len(covered)}")
    print(f"Coverage: {coverage_pct:.2f}%")
    print(f"Fundamentals dir: {fund_dir}")

    if missing:
        n = max(0, int(args.show_missing))
        if n > 0:
            sample = ", ".join(missing[:n])
            print(f"Missing sample ({min(n, len(missing))}): {sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
