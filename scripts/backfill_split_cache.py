#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.corporate_actions import backfill_split_cache
from data.custom_universe import list_cached_symbols, load_symbol_file
from data.universe import get_universe_symbols_pit_window_with_meta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill split-history cache for symbols.")
    parser.add_argument("--universe", default="cache_all")
    parser.add_argument("--symbol-file", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    universe = str(args.universe).strip().upper()
    if universe == "CACHE_ALL":
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        return list_cached_symbols(max_symbols=max_symbols)
    if universe == "RUSSELL3000":
        symbols, _ = get_universe_symbols_pit_window_with_meta("RUSSELL3000", args.start_date, args.end_date)
        return list(symbols)
    raise ValueError(f"Unsupported universe: {args.universe}")


def main() -> None:
    args = _parse_args()
    symbols = _resolve_symbols(args)
    if int(args.max_symbols or 0) > 0:
        symbols = symbols[: int(args.max_symbols)]
    payload = backfill_split_cache(
        symbols,
        batch_size=int(args.batch_size),
        force_refresh=bool(args.force_refresh),
    )
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
