#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.universe import (  # noqa: E402
    get_russell3000_pit_status,
    get_universe_symbols_pit_with_meta,
)


def _parse_ymd(value: str) -> Optional[date]:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            token = raw[:10] if fmt != "%Y%m%d" else raw
            return datetime.strptime(token, fmt).date()
        except Exception:
            continue
    return None


def _extract_snapshot_date(path: str) -> Optional[date]:
    stem = os.path.splitext(os.path.basename(path))[0]
    match_list = re.findall(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    for yy, mm, dd in match_list:
        try:
            return date(int(yy), int(mm), int(dd))
        except Exception:
            continue
    return None


@dataclass
class RangeValidation:
    path: str
    rows: int
    symbols: int
    errors: List[str]


def validate_membership_ranges_csv(path: str) -> RangeValidation:
    errors: List[str] = []
    rows = 0
    symbols = set()
    if not path or not os.path.exists(path):
        return RangeValidation(path=path, rows=0, symbols=0, errors=["file_missing"])

    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            field_map = {str(c).strip().lower(): c for c in fieldnames if c}
            sym_col = field_map.get("symbol") or field_map.get("ticker")
            start_col = field_map.get("start_date") or field_map.get("from_date")
            end_col = field_map.get("end_date") or field_map.get("to_date")
            if not sym_col:
                errors.append("missing_symbol_column(symbol|ticker)")
            if not start_col:
                errors.append("missing_start_column(start_date|from_date)")
            if errors:
                return RangeValidation(path=path, rows=0, symbols=0, errors=errors)

            for line_no, row in enumerate(reader, start=2):
                rows += 1
                sym = str(row.get(sym_col) or "").strip().upper()
                if not sym:
                    errors.append(f"line_{line_no}:empty_symbol")
                    continue
                symbols.add(sym)
                start_dt = _parse_ymd(str(row.get(start_col) or "").strip())
                if start_dt is None:
                    errors.append(f"line_{line_no}:invalid_start_date")
                    continue
                if end_col:
                    end_raw = str(row.get(end_col) or "").strip()
                    if end_raw:
                        end_dt = _parse_ymd(end_raw)
                        if end_dt is None:
                            errors.append(f"line_{line_no}:invalid_end_date")
                            continue
                        if end_dt < start_dt:
                            errors.append(f"line_{line_no}:end_before_start")
    except Exception as exc:
        errors.append(f"read_error:{exc}")

    return RangeValidation(path=path, rows=rows, symbols=len(symbols), errors=errors)


@dataclass
class SnapshotValidation:
    directory: str
    csv_files: int
    dated_files: int
    valid_symbol_files: int
    sample_errors: List[str]


def validate_snapshot_dir(path: str) -> SnapshotValidation:
    if not path or not os.path.isdir(path):
        return SnapshotValidation(
            directory=path,
            csv_files=0,
            dated_files=0,
            valid_symbol_files=0,
            sample_errors=["snapshot_dir_missing"],
        )

    files = [os.path.join(path, n) for n in os.listdir(path) if n.lower().endswith(".csv")]
    files.sort()
    dated_files = 0
    valid_symbol_files = 0
    sample_errors: List[str] = []

    for csv_path in files:
        if _extract_snapshot_date(csv_path) is not None:
            dated_files += 1
        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                fmap = {str(c).strip().lower(): c for c in fieldnames if c}
                sym_col = fmap.get("symbol") or fmap.get("ticker")
                if not sym_col:
                    sample_errors.append(f"{os.path.basename(csv_path)}:missing_symbol_column")
                    continue
                has_symbol = False
                for row in reader:
                    sym = str(row.get(sym_col) or "").strip().upper()
                    if sym:
                        has_symbol = True
                        break
                if has_symbol:
                    valid_symbol_files += 1
                else:
                    sample_errors.append(f"{os.path.basename(csv_path)}:no_symbols")
        except Exception as exc:
            sample_errors.append(f"{os.path.basename(csv_path)}:read_error:{exc}")

    return SnapshotValidation(
        directory=path,
        csv_files=len(files),
        dated_files=dated_files,
        valid_symbol_files=valid_symbol_files,
        sample_errors=sample_errors[:10],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Russell 3000 point-in-time membership inputs."
    )
    parser.add_argument(
        "--as-of",
        dest="as_of",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="As-of date in YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when PIT data is unavailable or validation errors exist.",
    )
    args = parser.parse_args()

    as_of = args.as_of
    as_of_dt = _parse_ymd(as_of)
    if as_of_dt is None:
        print(f"ERROR: invalid --as-of date: {as_of}")
        return 2

    status: Dict[str, object] = get_russell3000_pit_status(as_of_dt.isoformat())
    source = str(status.get("source", "unavailable") or "unavailable")
    symbol_count = int(status.get("symbol_count", 0) or 0)
    pit_dir = str(status.get("pit_dir", "") or "")
    range_csv = str(status.get("range_csv", "") or "")

    print(f"As-Of Date: {as_of_dt.isoformat()}")
    print(f"PIT Source: {source}")
    print(f"PIT Symbol Count: {symbol_count}")
    print(f"PIT Dir: {pit_dir or '(not set)'}")
    print(f"Range CSV: {range_csv or '(not set)'}")

    snap_validation = validate_snapshot_dir(pit_dir)
    print(
        "Snapshot Files: "
        f"{snap_validation.csv_files} csv, "
        f"{snap_validation.dated_files} dated, "
        f"{snap_validation.valid_symbol_files} with symbols"
    )
    for err in snap_validation.sample_errors:
        print(f" - snapshot_issue: {err}")

    range_validation = validate_membership_ranges_csv(range_csv) if range_csv else None
    if range_validation is not None:
        print(
            "Range CSV Rows/Symbols: "
            f"{range_validation.rows}/{range_validation.symbols}"
        )
        for err in range_validation.errors[:20]:
            print(f" - range_issue: {err}")

    loaded_symbols: List[str] = []
    loaded_source = "not_attempted"
    if source in {"pit_snapshot", "pit_ranges"}:
        loaded_symbols, loaded_source = get_universe_symbols_pit_with_meta(
            "RUSSELL3000",
            as_of_dt.isoformat(),
        )
    elif source == "unavailable":
        loaded_source = "unavailable"
    print(f"Loader Probe Source: {loaded_source}")
    print(f"Loader Probe Symbol Count: {len(loaded_symbols)}")

    errors: List[str] = []
    if source not in {"pit_snapshot", "pit_ranges"} or symbol_count <= 0:
        errors.append("pit_unavailable")
    if range_validation is not None and range_validation.errors:
        errors.append("range_validation_failed")
    if snap_validation.sample_errors and snap_validation.valid_symbol_files == 0:
        errors.append("snapshot_validation_failed")
    if source in {"pit_snapshot", "pit_ranges"} and loaded_source == "fallback_current":
        errors.append("loader_fallback_current")

    if errors:
        print("Validation status: FAIL")
        for err in errors:
            print(f" - {err}")
        return 2 if args.strict else 1

    print("Validation status: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
