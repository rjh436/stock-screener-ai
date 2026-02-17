from __future__ import annotations

import csv
import json
import os
import re
from datetime import date, datetime
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

import requests

try:
    from data.indices import get_index_symbols as _get_index_symbols
except Exception:
    _get_index_symbols = None

_CACHE_DIR = os.path.join("data", "cache_indices")
_RUSSELL3000_PIT_DIR = os.path.join("data", "russell3000_membership")
_RUSSELL3000_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)

_DEFAULT_LARGE_CAPS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "BRK/B",
    "TSLA",
    "JPM",
    "JNJ",
    "V",
    "PG",
    "MA",
    "HD",
    "AVGO",
    "XOM",
    "LLY",
    "UNH",
    "COST",
    "MRK",
    "PEP",
    "ABBV",
    "KO",
    "WMT",
    "BAC",
]


def get_universe_symbols(name: str) -> List[str]:
    key = _normalize_name(name)
    if key == "SP100":
        symbols = _fetch_sp100()
    elif key == "SP500":
        symbols = _fetch_sp500()
    elif key == "SP1500":
        symbols = _fetch_sp1500()
    elif key == "NASDAQ100":
        symbols = _fetch_from_indices("NASDAQ 100")
    elif key == "RUSSELL3000":
        print(
            "⚠️ WARNING: Russell 3000 contains Survivorship Bias. "
            "Backtest results > 5 years are inflated."
        )
        symbols = _fetch_russell_3000()
    else:
        symbols = []

    return symbols or _safe_default()


def get_universe_symbols_pit(name: str, as_of_date: Optional[str] = None) -> List[str]:
    """
    Return point-in-time universe members when local PIT files are available.
    Falls back to get_universe_symbols(name) when PIT data is missing.
    """
    symbols, _ = get_universe_symbols_pit_with_meta(name, as_of_date)
    return symbols


def get_universe_symbols_pit_with_meta(
    name: str,
    as_of_date: Optional[str] = None,
) -> Tuple[List[str], str]:
    """
    Return (symbols, source) where source is one of:
    - pit_snapshot
    - pit_ranges
    - current_index
    - fallback_current
    """
    key = _normalize_name(name)
    as_of = _parse_date_ymd(as_of_date)
    if as_of is None or key != "RUSSELL3000":
        return get_universe_symbols(name), "current_index"

    symbols = _load_russell_3000_pit_from_snapshots(as_of)
    if symbols:
        print(f"   └── Loaded PIT Russell 3000 snapshot for {as_of.isoformat()} ({len(symbols)} symbols)")
        return symbols, "pit_snapshot"

    symbols = _load_russell_3000_pit_from_ranges(as_of)
    if symbols:
        print(f"   └── Loaded PIT Russell 3000 membership ranges for {as_of.isoformat()} ({len(symbols)} symbols)")
        return symbols, "pit_ranges"

    print(
        "   ⚠️ WARNING: No PIT Russell 3000 dataset found. "
        "Falling back to current constituents (survivorship bias remains)."
    )
    return get_universe_symbols(name), "fallback_current"


def get_russell3000_pit_status(as_of_date: Optional[str] = None) -> Dict[str, object]:
    """
    Return PIT availability metadata without falling back to live universe downloads.

    Keys:
    - as_of_date (str)
    - source (str): pit_snapshot | pit_ranges | unavailable
    - symbol_count (int)
    - pit_dir (str)
    - pit_dir_exists (bool)
    - range_csv (str)
    - range_csv_exists (bool)
    """
    as_of = _parse_date_ymd(as_of_date)
    if as_of is None:
        as_of = datetime.utcnow().date()

    pit_dir = os.getenv("RUSSELL3000_PIT_DIR", _RUSSELL3000_PIT_DIR)
    range_csv = os.getenv("RUSSELL3000_PIT_MEMBERSHIP_CSV", "").strip()

    snapshot_symbols = _load_russell_3000_pit_from_snapshots(as_of)
    if snapshot_symbols:
        return {
            "as_of_date": as_of.isoformat(),
            "source": "pit_snapshot",
            "symbol_count": int(len(snapshot_symbols)),
            "pit_dir": pit_dir,
            "pit_dir_exists": bool(pit_dir and os.path.isdir(pit_dir)),
            "range_csv": range_csv,
            "range_csv_exists": bool(range_csv and os.path.exists(range_csv)),
        }

    range_symbols = _load_russell_3000_pit_from_ranges(as_of)
    if range_symbols:
        return {
            "as_of_date": as_of.isoformat(),
            "source": "pit_ranges",
            "symbol_count": int(len(range_symbols)),
            "pit_dir": pit_dir,
            "pit_dir_exists": bool(pit_dir and os.path.isdir(pit_dir)),
            "range_csv": range_csv,
            "range_csv_exists": bool(range_csv and os.path.exists(range_csv)),
        }

    return {
        "as_of_date": as_of.isoformat(),
        "source": "unavailable",
        "symbol_count": 0,
        "pit_dir": pit_dir,
        "pit_dir_exists": bool(pit_dir and os.path.isdir(pit_dir)),
        "range_csv": range_csv,
        "range_csv_exists": bool(range_csv and os.path.exists(range_csv)),
    }


def _fetch_sp100() -> List[str]:
    symbols = _fetch_from_indices("S&P 100")
    if symbols:
        return symbols

    cached = _load_cached_symbols("s&p_100.json")
    if cached:
        return cached

    sp500 = _load_cached_symbols("s&p_500.json")
    if sp500:
        return sp500[:100]

    return list(_DEFAULT_LARGE_CAPS)


def _fetch_sp500() -> List[str]:
    symbols = _fetch_from_indices("S&P 500")
    if symbols:
        return symbols

    cached = _load_cached_symbols("s&p_500.json")
    if cached:
        return cached

    return list(_DEFAULT_LARGE_CAPS)


def _fetch_sp1500() -> List[str]:
    sp500 = _fetch_sp500()
    sp400 = _fetch_from_indices("S&P 400") or _load_cached_symbols("s&p_400.json")
    sp600 = _fetch_from_indices("S&P 600") or _load_cached_symbols("s&p_600.json")
    combined = _normalize_symbols([*sp500, *sp400, *sp600])
    if len(combined) >= 1000:
        return combined

    fallback = _fetch_from_indices("S&P 1500")
    if fallback:
        return fallback

    for cache_name in ("sp1500_sptm.json", "s&p_1500.json"):
        cached = _load_cached_symbols(cache_name)
        if cached:
            return cached

    return []


def _fetch_russell_3000() -> List[str]:
    csv_path = os.path.join("data", "russell3000.csv")
    if os.path.exists(csv_path):
        print(f"   └── Loading from local file: {csv_path}")
        try:
            with open(csv_path, "r", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                field_map = {name.strip().lower(): name for name in fieldnames if name}
                col = field_map.get("ticker") or field_map.get("symbol")
                symbols = []
                if col:
                    for row in reader:
                        raw = row.get(col) or ""
                        sym = str(raw).strip().upper()
                        if sym:
                            symbols.append(sym)
                return _normalize_symbols(symbols)
        except Exception as exc:
            print(f"   ⚠️ WARNING: Failed to read {csv_path}: {exc}")

    try:
        resp = requests.get(
            _RUSSELL3000_URL,
            headers={"User-Agent": "Mozilla/5.0 (universe-fetch)"},
            timeout=30,
        )
        resp.raise_for_status()
        return _parse_ishares_csv(resp.text)
    except Exception:
        pass

    print("   ⚠️ WARNING: Russell 3000 download failed. Using Synthetic Proxy (SP1500 + NASDAQ100).")
    sp1500 = _fetch_sp1500()
    nasdaq = _fetch_from_indices("NASDAQ 100")
    combined = list(set(sp1500 + nasdaq))
    return combined


def _parse_date_ymd(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw[:10] if fmt != "%Y%m%d" else raw, fmt).date()
        except Exception:
            continue
    return None


def _extract_snapshot_date(path: str) -> Optional[date]:
    stem = os.path.splitext(os.path.basename(path))[0]
    matches = re.findall(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", stem)
    for y, m, d in matches:
        try:
            return date(int(y), int(m), int(d))
        except Exception:
            continue
    for token in [stem] + stem.replace("_", "-").split("-"):
        dt = _parse_date_ymd(token)
        if dt is not None:
            return dt
    return None


def _read_symbol_csv(path: str) -> List[str]:
    try:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            field_map = {name.strip().lower(): name for name in fieldnames if name}
            col = field_map.get("ticker") or field_map.get("symbol")
            if not col:
                return []
            out = []
            for row in reader:
                raw = row.get(col) or ""
                sym = str(raw).strip().upper()
                if sym:
                    out.append(sym)
            return _normalize_symbols(out)
    except Exception:
        return []


def _load_russell_3000_pit_from_snapshots(as_of: date) -> List[str]:
    pit_dir = os.getenv("RUSSELL3000_PIT_DIR", _RUSSELL3000_PIT_DIR)
    if not pit_dir or not os.path.isdir(pit_dir):
        return []

    best_path = None
    best_dt = None
    for name in os.listdir(pit_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(pit_dir, name)
        snap_dt = _extract_snapshot_date(path)
        if snap_dt is None or snap_dt > as_of:
            continue
        if best_dt is None or snap_dt > best_dt:
            best_dt = snap_dt
            best_path = path

    if best_path is None:
        return []
    return _read_symbol_csv(best_path)


def _load_russell_3000_pit_from_ranges(as_of: date) -> List[str]:
    path = os.getenv("RUSSELL3000_PIT_MEMBERSHIP_CSV", "").strip()
    if not path or not os.path.exists(path):
        return []

    try:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            field_map = {name.strip().lower(): name for name in fieldnames if name}
            sym_col = field_map.get("ticker") or field_map.get("symbol")
            start_col = field_map.get("start_date") or field_map.get("from_date")
            end_col = field_map.get("end_date") or field_map.get("to_date")
            if not sym_col or not start_col:
                return []

            out = []
            for row in reader:
                sym = str(row.get(sym_col) or "").strip().upper()
                if not sym:
                    continue
                start_dt = _parse_date_ymd(str(row.get(start_col) or "").strip())
                if start_dt is None or start_dt > as_of:
                    continue
                end_dt = _parse_date_ymd(str(row.get(end_col) or "").strip()) if end_col else None
                if end_dt is not None and end_dt < as_of:
                    continue
                out.append(sym)
            return _normalize_symbols(out)
    except Exception:
        return []


def _parse_ishares_csv(text: str) -> List[str]:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,") or line.startswith("Symbol,"):
            header_idx = i
            break
    if header_idx is None:
        return []

    reader = csv.DictReader(StringIO("\n".join(lines[header_idx:])))
    symbols = []
    for row in reader:
        raw = row.get("Ticker") or row.get("Symbol") or ""
        sym = str(raw).strip().upper()
        if sym:
            symbols.append(sym)
    return _normalize_symbols(symbols)


def _fetch_from_indices(index_name: str) -> List[str]:
    if _get_index_symbols is None:
        return []
    try:
        symbols = _get_index_symbols(index_name) or []
    except Exception:
        return []
    return _normalize_symbols(symbols)


def _load_cached_symbols(filename: str) -> List[str]:
    path = os.path.join(_CACHE_DIR, filename)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return _normalize_symbols(data)


def _normalize_symbols(symbols: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for symbol in symbols:
        if not symbol:
            continue
        sym = str(symbol).strip().upper().replace(".", "/")
        if not _is_valid_symbol(sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized


def _is_valid_symbol(symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    if not all(ch.isalnum() or ch in "./-" for ch in sym):
        return False
    # Exclude placeholders like "-" that can leak from holdings files.
    return any(ch.isalnum() for ch in sym)


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in (name or "").upper() if ch.isalnum())


def _safe_default() -> List[str]:
    cached = _load_cached_symbols("s&p_500.json")
    if cached:
        return cached
    fallback = _fetch_from_indices("S&P 500")
    if fallback:
        return fallback
    return list(_DEFAULT_LARGE_CAPS)
