import pandas as pd
import requests
import os
import json
from io import StringIO

# Cache directory
CACHE_DIR = "data/cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)

# Emergency Fallback (Top 20 tickers) to ensure system NEVER returns empty
FALLBACK_TICKERS = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "TSLA",
    "BRK.B",
    "UNH",
    "JNJ",
    "JPM",
    "XOM",
    "V",
    "PG",
    "MA",
    "HD",
    "CVX",
    "ABBV",
    "MRK",
    "PEP",
]


def get_cache_path(index_name):
    clean_name = index_name.replace(" ", "_").lower()
    return os.path.join(CACHE_DIR, f"{clean_name}.json")


def load_from_cache(index_name):
    path = get_cache_path(index_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_to_cache(index_name, symbols):
    path = get_cache_path(index_name)
    with open(path, "w") as f:
        json.dump(symbols, f)


def get_index_symbols(index_name="S&P 100"):
    """
    Robust fetcher with Caching, Headers, and Fallback.
    """
    cached = load_from_cache(index_name)
    if cached and len(cached) > 0:
        return cached

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
    }

    symbols = []
    try:
        if index_name == "S&P 100":
            url = "https://en.wikipedia.org/wiki/S%26P_100"
            r = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(StringIO(r.text))
            df = tables[2]
            symbols = df["Symbol"].tolist()

        elif index_name == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            r = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(StringIO(r.text))
            df = tables[0]
            symbols = df["Symbol"].tolist()

        elif index_name == "S&P 1500":
            s500 = get_index_symbols("S&P 500")

            url_400 = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
            r400 = requests.get(url_400, headers=headers, timeout=15)
            df400 = pd.read_html(StringIO(r400.text))[0]
            s400 = df400["Symbol"].tolist()

            url_600 = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
            r600 = requests.get(url_600, headers=headers, timeout=15)
            df600 = pd.read_html(StringIO(r600.text))[0]
            s600 = df600["Symbol"].tolist()

            symbols = s500 + s400 + s600

        elif index_name == "Nasdaq 100":
            url = "https://en.wikipedia.org/wiki/Nasdaq-100"
            r = requests.get(url, headers=headers, timeout=15)
            tables = pd.read_html(StringIO(r.text))
            df = tables[4]
            symbols = df["Ticker"].tolist()

    except Exception as e:
        print(f"❌ Index fetch failed for {index_name}: {e}")
        if index_name in ["S&P 100", "S&P 500"]:
            return FALLBACK_TICKERS
        return []

    cleaned_symbols = [s.replace(".", "/") for s in symbols]

    if len(cleaned_symbols) > 10:
        save_to_cache(index_name, cleaned_symbols)

    return cleaned_symbols
