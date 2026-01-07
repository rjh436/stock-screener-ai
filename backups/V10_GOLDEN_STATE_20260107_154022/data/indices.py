import pandas as pd
import requests
import os
import json
from io import BytesIO

CACHE_DIR = "data/cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "sp1500_sptm.json")

def get_index_symbols(index_name="S&P 1500"):
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
