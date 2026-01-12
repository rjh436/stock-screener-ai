from __future__ import annotations

import csv
import json
import os
from io import StringIO
from typing import Iterable, List

import requests

try:
    from data.indices import get_index_symbols as _get_index_symbols
except Exception:
    _get_index_symbols = None

_CACHE_DIR = os.path.join("data", "cache_indices")
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
    elif key == "RUSSELL3000":
        print(
            "⚠️ WARNING: Russell 3000 contains Survivorship Bias. "
            "Backtest results > 5 years are inflated."
        )
        symbols = _fetch_russell_3000()
    else:
        symbols = []

    return symbols or _safe_default()


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
    return all(ch.isalnum() or ch in "./-" for ch in symbol)


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
