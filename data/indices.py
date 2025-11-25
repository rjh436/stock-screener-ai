import pandas as pd
import requests
import os
import json
from io import StringIO

CACHE_DIR = "data/cache_indices"
os.makedirs(CACHE_DIR, exist_ok=True)

# STATIC BACKUP: Top ~450 Volatile Stocks from S&P 1500
STATIC_SP1500 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "LLY", "COST",
    "AVGO", "TMO", "CSCO", "MCD", "ACN", "ABT", "DHR", "LIN", "NEE", "DIS",
    "TXN", "PM", "AMD", "UPS", "UNP", "RTX", "HON", "MS", "BMY", "LOW", "INTC",
    "SPGI", "QCOM", "GS", "CAT", "IBM", "DE", "EL", "PLD", "LMT", "BLK", "AMT",
    "BKNG", "SYK", "MDLZ", "ADI", "TJX", "GILD", "MMC", "GE", "ISRG", "ADP",
    "NOW", "C", "AMAT", "BA", "ZTS", "MO", "REGN", "VRTX", "PYPL", "TMUS", "FI",
    "SMCI", "DECK", "FICO", "TRGP", "PTC", "RPM", "WSO", "HUBB", "BLDR", "MKSI",
    "WMS", "TTC", "TREX", "LECO", "SAIA", "EMN", "EXP", "WST", "RGLD", "CHE",
    "TEX", "GME", "AMC", "RH", "DKNG", "MSTR", "AFRM", "UPST", "COIN", "DOCU",
    "CROX", "FIVE", "LII", "MANH", "BJ", "YETI", "WING", "FOXF", "MEDP", "LSCC",
    "ON", "MPWR", "ENTG", "WOLF", "LITE", "POWI", "COHR", "SPSC", "NOVT", "ALGM",
    "PLUG", "FCEL", "BLDP", "BE", "RUN", "NOVA", "SPWR", "FSLR", "ENPH", "MAXN",
    "ARRY", "SHLS", "TPIC", "CSIQ", "JKS", "DQ", "SOL", "TAN", "ICLN", "XBI",
    "LABU", "LABD", "NBI", "IBB", "ARKG", "CRSP", "NTLA", "EDIT", "BEAM", "MARA",
    "RIOT", "CLSK", "HUT", "HIVE", "BTBT", "MIGI", "WULF", "SDIG", "GREE", "AAOI",
    "ACLS", "AEIS", "ALRM", "AMBA", "APPS", "CALX", "CEVA", "COHU", "DIOD", "FORM",
    "HLIT", "IOTS", "LASR", "MTSI", "OSIS", "PDFS", "PLAB", "RMBS", "SGH", "SMTC",
    "TTMI", "VECO", "VSAT", "XPER", "PRCT", "PRO", "QTWO", "RPD", "SAIL", "SMAR"
]

def get_cache_path(index_name):
    clean_name = index_name.replace(" ", "_").lower()
    return os.path.join(CACHE_DIR, f"{clean_name}.json")

def load_from_cache(index_name):
    path = get_cache_path(index_name)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if len(data) > 10: return data
        except: return None
    return None

def save_to_cache(index_name, symbols):
    try:
        with open(get_cache_path(index_name), "w") as f: json.dump(symbols, f)
    except: pass

def fetch_with_headers(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            return pd.read_html(StringIO(r.text))
    except: pass
    return None

def get_index_symbols(index_name="S&P 100"):
    cached = load_from_cache(index_name)
    if cached: return cached
    
    symbols = []
    
    if index_name == "S&P 1500":
        # Try component fetch
        try:
            s500 = get_index_symbols("S&P 500")
            
            dfs_400 = fetch_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
            s400 = dfs_400[0]['Symbol'].tolist() if dfs_400 else []
            
            dfs_600 = fetch_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")
            s600 = dfs_600[0]['Symbol'].tolist() if dfs_600 else []
            
            symbols = list(set(s500 + s400 + s600))
        except: symbols = []

    elif index_name == "S&P 500":
        dfs = fetch_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        if dfs: symbols = dfs[0]['Symbol'].tolist()

    elif index_name == "S&P 100":
        dfs = fetch_with_headers("https://en.wikipedia.org/wiki/S%26P_100")
        if dfs: symbols = dfs[2]['Symbol'].tolist()
    
    # Fallback
    if len(symbols) < 50:
        print(f"⚠️ Fetch failed for {index_name}. Using Static Backup.")
        symbols = STATIC_SP1500
    
    cleaned = [s.replace('.', '/') for s in symbols if isinstance(s, str)]
    save_to_cache(index_name, cleaned)
    return cleaned
