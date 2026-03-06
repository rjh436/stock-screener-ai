#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.fundamental_loader import LoaderConfig, refresh_fundamentals


def _normalize_ticker(raw: object) -> str:
    return str(raw or "").strip().upper()


def _default_output_dir() -> Path:
    return ROOT / "data" / "fundamentals" / "edgar_income"


def _partition_path(root: Path, ticker: str) -> Path:
    return root / f"ticker={ticker}" / "fundamentals.parquet"


def _read_tickers_from_root(root: Path) -> List[str]:
    tickers: List[str] = []
    for child in sorted(root.glob("ticker=*")):
        if not child.is_dir():
            continue
        ticker = _normalize_ticker(child.name.split("=", 1)[1])
        if ticker:
            tickers.append(ticker)
    return tickers


def _partition_missing_shares(path: Path) -> bool:
    if not path.exists():
        return True
    if pd is None:
        return True
    try:
        df = pd.read_parquet(path, engine="pyarrow", columns=["shares_outstanding"])
    except Exception:
        return True
    if "shares_outstanding" not in df.columns:
        return True
    series = pd.to_numeric(df["shares_outstanding"], errors="coerce")
    return bool(series.dropna().empty)


def find_tickers_missing_shares(
    root: Path,
    *,
    tickers: Optional[Sequence[str]] = None,
    limit: int = 0,
) -> List[str]:
    universe = [_normalize_ticker(t) for t in (tickers or _read_tickers_from_root(root))]
    out: List[str] = []
    seen = set()
    for ticker in universe:
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        if _partition_missing_shares(_partition_path(root, ticker)):
            out.append(ticker)
            if limit > 0 and len(out) >= int(limit):
                break
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh EDGAR fundamentals for tickers whose parquet partitions are missing shares_outstanding."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_default_output_dir()),
        help="Partition root containing ticker=* parquet outputs.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional comma-separated ticker override. Defaults to scanning existing partitions.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on tickers to refresh.")
    parser.add_argument(
        "--identity",
        type=str,
        default=os.getenv("SEC_EDGAR_IDENTITY", ""),
        help="SEC identity string: 'Name email@example.com'",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--sec-rate-limit", type=int, default=8)
    parser.add_argument("--max-filings-per-ticker", type=int, default=16)
    parser.add_argument("--max-tasks-per-child", type=int, default=8)
    parser.add_argument("--request-pause-ms", type=int, default=120)
    parser.add_argument("--task-timeout-sec", type=int, default=300)
    parser.add_argument("--heartbeat-sec", type=int, default=15)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually refresh the selected tickers. Without this flag the script only reports the list.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(str(args.output_dir)).resolve()
    tickers: Optional[Iterable[str]] = None
    if args.tickers:
        tickers = [x for x in (_normalize_ticker(p) for p in str(args.tickers).split(",")) if x]

    selected = find_tickers_missing_shares(root, tickers=tickers, limit=int(args.limit or 0))
    payload = {
        "output_dir": str(root),
        "requested_override_count": (0 if tickers is None else len(list(tickers))),
        "selected_count": len(selected),
        "tickers": selected,
        "execute": bool(args.execute),
    }

    if not args.execute:
        print(json.dumps(payload, indent=2))
        return

    if not selected:
        print(json.dumps(payload, indent=2))
        return

    config = LoaderConfig(
        output_dir=root,
        identity=str(args.identity or ""),
        max_workers=max(1, int(args.max_workers or 1)),
        sec_rate_limit=max(1, int(args.sec_rate_limit or 1)),
        max_filings_per_ticker=max(1, int(args.max_filings_per_ticker or 1)),
        max_tasks_per_child=max(0, int(args.max_tasks_per_child or 0)),
        request_pause_sec=max(0.0, float(args.request_pause_ms or 0) / 1000.0),
        task_timeout_sec=max(0.0, float(args.task_timeout_sec or 0)),
        heartbeat_sec=max(1.0, float(args.heartbeat_sec or 15)),
        overwrite=True,
        verbose=not bool(args.quiet),
    )
    summary = refresh_fundamentals(selected, config)
    payload["refresh_summary"] = summary
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
