import csv
import json
import os
from io import BytesIO, StringIO

import pandas as pd
import requests

CACHE_DIR = "data/cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "sp1500_sptm.json")
R3000_CACHE_FILE = os.path.join(CACHE_DIR, "russell3000_iwv.json")
R3000_URL = (
    "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/"
    "1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
)

def get_index_symbols(index_name="S&P 1500"):
    key = _normalize_index_name(index_name)
    if key in {"R3000", "RUSSELL3000", "IWV"}:
        cached = _load_cached_symbols(R3000_CACHE_FILE, min_len=2000)
        if cached:
            return cached
        symbols = _fetch_russell_3000()
        if len(symbols) >= 2000:
            with open(R3000_CACHE_FILE, "w") as f:
                json.dump(symbols, f)
        return symbols

    # 1. Return Cache if recent (prevent spamming SSGA)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                symbols = json.load(f)
            if len(symbols) > 1400: return symbols
        except: pass

    print("🌐 Fetching S&P 1500 from State Street (SPTM)...")
    
    # 2. Fetch Official Holdings (SPTM ETF)
    url = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-sptm.xlsx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200:
            # Skip first few rows of header junk, look for 'Ticker'
            df = pd.read_excel(BytesIO(r.content), header=4) 
            
            # Normalize columns
            df.columns = df.columns.str.strip().str.upper()
            
            # Find Ticker column
            ticker_col = next((c for c in df.columns if 'TICKER' in c), None)
            if ticker_col:
                symbols = df[ticker_col].dropna().astype(str).tolist()
                # Clean (e.g. remove cash/msg)
                symbols = [s.strip().replace('.', '/') for s in symbols if len(s) < 6 and s.isalpha()]
                
                # Verify Count
                if len(symbols) > 1400:
                    print(f"✅ Successfully fetched {len(symbols)} symbols from SPTM.")
                    with open(CACHE_FILE, "w") as f: json.dump(symbols, f)
                    return symbols
    except Exception as e:
        print(f"❌ SPTM Fetch Failed: {e}")

    # 3. Fallback (If SSGA fails, return empty so user knows)
    print("⚠️ Could not fetch S&P 1500. Check internet or URL.")
    return []


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
        if not _is_valid_symbol(sym):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        normalized.append(sym)
    return normalized


def _is_valid_symbol(symbol: str) -> bool:
    return all(ch.isalnum() or ch in "./-" for ch in symbol)


def _normalize_index_name(name: str) -> str:
    return "".join(ch for ch in str(name or "").upper() if ch.isalnum())
