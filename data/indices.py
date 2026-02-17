import csv
import json
import os
from io import BytesIO, StringIO
from typing import Iterable, Optional

import pandas as pd
import requests

CACHE_DIR = "data/cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)
SP1500_CACHE_FILE = os.path.join(CACHE_DIR, "sp1500_sptm.json")
SP500_CACHE_FILE = os.path.join(CACHE_DIR, "sp500_wikipedia.json")
SP400_CACHE_FILE = os.path.join(CACHE_DIR, "sp400_wikipedia.json")
SP600_CACHE_FILE = os.path.join(CACHE_DIR, "sp600_wikipedia.json")
SP100_CACHE_FILE = os.path.join(CACHE_DIR, "sp100_wikipedia.json")
NASDAQ100_CACHE_FILE = os.path.join(CACHE_DIR, "nasdaq100_wikipedia.json")
R3000_CACHE_FILE = os.path.join(CACHE_DIR, "russell3000_iwv.json")
R3000_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
SP100_URL = "https://en.wikipedia.org/wiki/S%26P_100"
NASDAQ100_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"
SP1500_SPTM_URL = (
    "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-sptm.xlsx"
)
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def get_index_symbols(index_name="S&P 1500"):
    key = _normalize_index_name(index_name)
    if key in {"R3000", "RUSSELL3000", "IWV"}:
        return _get_from_cache_or_fetch(
            cache_path=R3000_CACHE_FILE,
            min_len=2000,
            fetcher=_fetch_russell_3000,
        )
    if key in {"SP500"}:
        return _get_from_cache_or_fetch(
            cache_path=SP500_CACHE_FILE,
            min_len=450,
            fetcher=lambda: _fetch_wikipedia_constituents(SP500_URL, expected_count=500, min_count=450),
        )
    if key in {"SP400"}:
        return _get_from_cache_or_fetch(
            cache_path=SP400_CACHE_FILE,
            min_len=350,
            fetcher=lambda: _fetch_wikipedia_constituents(SP400_URL, expected_count=400, min_count=350),
        )
    if key in {"SP600"}:
        return _get_from_cache_or_fetch(
            cache_path=SP600_CACHE_FILE,
            min_len=500,
            fetcher=lambda: _fetch_wikipedia_constituents(SP600_URL, expected_count=600, min_count=500),
        )
    if key in {"SP100", "OEX"}:
        return _get_from_cache_or_fetch(
            cache_path=SP100_CACHE_FILE,
            min_len=80,
            fetcher=lambda: _fetch_wikipedia_constituents(SP100_URL, expected_count=100, min_count=80),
        )
    if key in {"NASDAQ100", "NDX", "QQQ", "NASDAQ100INDEX"}:
        return _get_from_cache_or_fetch(
            cache_path=NASDAQ100_CACHE_FILE,
            min_len=80,
            fetcher=lambda: _fetch_wikipedia_constituents(NASDAQ100_URL, expected_count=100, min_count=80),
        )
    if key in {"SP1500"}:
        return _get_from_cache_or_fetch(
            cache_path=SP1500_CACHE_FILE,
            min_len=1200,
            fetcher=_fetch_sp1500,
        )

    return []


def _get_from_cache_or_fetch(cache_path: str, *, min_len: int, fetcher) -> list[str]:
    cached = _load_cached_symbols(cache_path, min_len=min_len)
    if cached:
        return cached

    symbols = _normalize_symbols(fetcher() or [])
    if len(symbols) >= min_len:
        _save_cached_symbols(cache_path, symbols)
        return symbols

    # Fallback to whatever cache exists if network fetch fails.
    cached_loose = _load_cached_symbols(cache_path, min_len=1)
    if cached_loose:
        return cached_loose
    return symbols


def _save_cached_symbols(path: str, symbols: Iterable[str]) -> None:
    normalized = _normalize_symbols(symbols)
    if not normalized:
        return
    try:
        with open(path, "w") as handle:
            json.dump(normalized, handle)
    except Exception:
        pass


def _fetch_sp1500() -> list[str]:
    print("🌐 Fetching S&P 1500 from State Street (SPTM)...")
    try:
        resp = requests.get(SP1500_SPTM_URL, headers=DEFAULT_HEADERS, timeout=30)
        resp.raise_for_status()
        for header_row in (4, 3, 0):
            try:
                df = pd.read_excel(BytesIO(resp.content), header=header_row)
            except Exception:
                continue
            ticker_col = _find_symbol_column(df)
            if not ticker_col:
                continue
            symbols = _normalize_symbols(df[ticker_col].dropna().astype(str).tolist())
            if len(symbols) >= 1200:
                print(f"✅ Successfully fetched {len(symbols)} symbols from SPTM.")
                return symbols
    except Exception as exc:
        print(f"❌ SPTM fetch failed: {exc}")

    print("⚠️ Falling back to S&P 500 + S&P 400 + S&P 600 composition.")
    combined = _normalize_symbols(
        [
            *get_index_symbols("SP500"),
            *get_index_symbols("SP400"),
            *get_index_symbols("SP600"),
        ]
    )
    return combined


def _fetch_wikipedia_constituents(url: str, *, expected_count: int, min_count: int) -> list[str]:
    try:
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
    except Exception as exc:
        print(f"WARNING: Wikipedia fetch failed for {url}: {exc}")
        return []

    candidates: list[list[str]] = []
    for df in tables:
        symbols = _extract_symbols_from_table(df)
        if len(symbols) >= min_count:
            candidates.append(symbols)

    if not candidates:
        return []

    # Prefer the table closest to the expected constituent count.
    best = min(candidates, key=lambda s: abs(len(s) - expected_count))
    return _normalize_symbols(best)


def _extract_symbols_from_table(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []

    candidates = []
    for col in df.columns:
        col_label = _flatten_column_name(col)
        norm = _normalize_index_name(col_label)
        if norm not in {"SYMBOL", "TICKER", "TICKERSYMBOL"}:
            continue
        raw_values = df[col].tolist()
        symbols = _normalize_symbols(raw_values)
        if symbols:
            candidates.append(symbols)

    if not candidates:
        return []
    return max(candidates, key=len)


def _flatten_column_name(col) -> str:
    if isinstance(col, tuple):
        parts = [str(part).strip() for part in col if str(part).strip()]
        return " ".join(parts)
    return str(col).strip()


def _find_symbol_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in df.columns:
        norm = _normalize_index_name(_flatten_column_name(col))
        if norm in {"SYMBOL", "TICKER", "TICKERSYMBOL"}:
            return col
    return None


def _fetch_russell_3000() -> list[str]:
    csv_path = os.path.join("data", "russell3000.csv")
    if os.path.exists(csv_path):
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
            print(f"WARNING: Failed to read {csv_path}: {exc}")

    try:
        resp = requests.get(
            R3000_URL,
            headers={"User-Agent": "Mozilla/5.0 (index-fetch)"},
            timeout=30,
        )
        if resp.status_code == 200:
            return _parse_ishares_csv(resp.text)
    except Exception as exc:
        print(f"WARNING: Russell 3000 fetch failed: {exc}")
    return []


def _parse_ishares_csv(text: str) -> list[str]:
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


def _load_cached_symbols(path: str, *, min_len: int) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    symbols = _normalize_symbols(data)
    return symbols if len(symbols) >= min_len else []


def _normalize_symbols(symbols) -> list[str]:
    seen = set()
    normalized = []
    for symbol in symbols or []:
        if not symbol:
            continue
        sym = str(symbol).strip().upper().replace(".", "/")
        if len(sym) > 8:
            continue
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
    return any(ch.isalpha() for ch in sym)


def _normalize_index_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").upper() if ch.isalnum())
