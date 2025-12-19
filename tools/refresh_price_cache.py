import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack


def main() -> int:
    p = argparse.ArgumentParser(description="Refresh cached daily OHLCV data with rate-limited API calls.")
    p.add_argument("--universe", default="S&P 1500", help="Universe name (default: S&P 1500)")
    p.add_argument("--days", type=int, default=1260, help="Lookback days (default: 1260)")
    p.add_argument("--max-workers", type=int, default=None, help="Symbol fetch workers (default: DATA_FETCH_WORKERS or 10)")
    p.add_argument("--max-lag-days", type=int, default=None, help="Allowed lag in days before data is considered stale")
    p.add_argument("--force-fresh", action="store_true", help="Always ping API and merge latest data")
    p.add_argument("--require-fresh", action="store_true", help="Drop symbols that can't be refreshed within lag threshold")
    p.add_argument("--require-full-lookback", action="store_true", help="Drop symbols without full lookback coverage")
    p.add_argument("--sample", type=int, default=0, help="Only refresh the first N symbols (for quick tests)")
    args = p.parse_args()

    symbols = get_index_symbols(args.universe)
    if not symbols:
        print("❌ No symbols returned from universe fetch.")
        return 1

    if args.sample and args.sample > 0:
        symbols = symbols[: args.sample]

    data = fetch_data_pack(
        symbols,
        days=args.days,
        max_workers=args.max_workers,
        require_full_lookback=args.require_full_lookback,
        force_fresh=args.force_fresh,
        require_fresh=args.require_fresh,
        max_lag_days=args.max_lag_days,
    )
    print(f"✅ Loaded {len(data)}/{len(symbols)} symbols into cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

