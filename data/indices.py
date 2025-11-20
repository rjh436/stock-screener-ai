import requests
import time
from bs4 import BeautifulSoup

def _extract_symbols_from_wiki_tables(html_text, prefer_headers=("Symbol", "Ticker", "Ticker symbol")):
    soup = BeautifulSoup(html_text, "html.parser")
    tables = soup.find_all("table", class_="wikitable")
    symbols = []
    for table in tables:
        header = table.find("tr")
        if not header:
            continue
        headers = [hc.get_text(strip=True) for hc in header.find_all(["th", "td"])]
        col_idx = None
        for pref in prefer_headers:
            if pref in headers:
                col_idx = headers.index(pref)
                break
        if col_idx is None:
            for i, h in enumerate(headers):
                if "symbol" in h.lower():
                    col_idx = i
                    break
        if col_idx is None:
            continue
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= col_idx:
                continue
            raw = cells[col_idx].get_text(" ", strip=True)
            if not raw:
                continue
            sym = (
                raw.upper()
                .replace("\u200a", "").replace(" ", "").replace("\n", "").replace("\t", "")
            )
            sym = sym.split("[")[0].strip()
            if len(sym) > 0 and all(ch.isalnum() or ch in ".-/" for ch in sym):
                symbols.append(sym)
    seen = set()
    uniq = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


# Use a simple cache decorator if streamlit is available, otherwise no-op
try:
    import streamlit as st
    cache_data = st.cache_data(ttl=86400, show_spinner=False)
except ImportError:
    def cache_data(func):
        return func

@cache_data
def get_index_symbols(index_name: str):
    idx = index_name.strip().upper()
    try:
        if idx == "S&P 100":
            url = "https://en.wikipedia.org/wiki/S%26P_100"
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (index-fetch)"})
            r.raise_for_status()
            syms = _extract_symbols_from_wiki_tables(r.text)
            return [s for s in syms if len(s) <= 6][:200]
        elif idx == "S&P 500":
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (index-fetch)"})
            r.raise_for_status()
            syms = _extract_symbols_from_wiki_tables(r.text)
            return [s for s in syms if len(s) <= 6][:600]
        elif idx == "S&P 1500":
            urls = [
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
                "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
            ]
            all_syms = []
            for url in urls:
                r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (index-fetch)"})
                r.raise_for_status()
                all_syms.extend(_extract_symbols_from_wiki_tables(r.text))
                time.sleep(0.4)
            seen = set()
            uniq = []
            for s in all_syms:
                if s not in seen:
                    seen.add(s)
                    uniq.append(s)
            return [s for s in uniq if len(s) <= 6][:1700]
        else:
            return []
    except Exception as e:
        print(f"Index fetch failed for {index_name}: {e}")
        return []
