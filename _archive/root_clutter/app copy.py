import datetime as dt
import os as os_mod, math, time, requests, pandas as pd, streamlit as st
from datetime import datetime, timedelta, timezone, date
from dotenv import load_dotenv
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, KeltnerChannel
from urllib.parse import quote_plus
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup

# --- NEW Backtest Engine Imports ---
from backtest_runner import run_compare
import backtrader as bt
# -----------------------------------

# ==============================
# App Config
# ==============================
st.set_page_config(page_title="RJH Custom Strategy Screener & Backtester", layout="wide")
st.title("🚀 RJH Custom Scanner & Backtester")

# ==============================
# Scanner Output
# ==============================
# Container placeholder. The content will be added at the end.
scan_out_container = st.container()

# ==============================
# Auth / Schwab wrapper
# ==============================
load_dotenv(override=True)
try:
    from schwab.auth import easy_client  # from schwab-py
except ImportError:
    st.error("Missing `schwab-py` library. Please install it: pip install schwab-py")
    st.stop()


# === Sidebar: Historical Data Export ===
def render_historical_export_sidebar(index_choice: str):
    import io, csv, zipfile
    from datetime import datetime, timedelta
    import streamlit as st

    st.sidebar.markdown("### 📊 Historical Data Export")

    export_choice = st.sidebar.radio(
        "Choose format",
        ("Single CSV (All Symbols)", "ZIP (One CSV per Symbol)"),
        index=1,
        key="hist_export_choice",
    )

    # --- Inserted: History window selection ---
    history_choice = st.sidebar.selectbox(
        "History window",
        [
            "1 year (~252 trading days)",
            "5 years (~1,260 trading days)",
            "10 years (~2,520 trading days)",
            "20 years (~5,040 trading days)",
        ],
        index=1,
        key="hist_window_choice",
    )

    if history_choice.startswith("1 year"):
        days_back = 365
    elif history_choice.startswith("5 years"):
        days_back = 365 * 5
    elif history_choice.startswith("10 years"):
        days_back = 365 * 10
    elif history_choice.startswith("20 years"):
        days_back = 365 * 20
    else:
        days_back = 365 * 5  # sensible fallback

    if st.sidebar.button("Run Export", key="hist_run_export"):
        symbols = get_index_symbols(index_choice)
        if not symbols:
            st.sidebar.warning("No symbols found for the selected universe.")
            return

        os_mod.makedirs("exports", exist_ok=True)
        ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
        base = index_choice.replace(" ", "").replace("&", "and").lower()
        failed = []

        progress = st.sidebar.progress(0.0, text="Starting export…")
        total = len(symbols)

        def fetch_all(sym):
            """Fetch historical daily bars for a symbol over the selected history window.

            Uses chunked requests (~400 days each) starting `days_back` days ago
            up to now, similar to the Backtrader data provider.
            If Schwab returns an error, we record it so we can see what went wrong
            instead of silently treating it as no data.
            """
            try:
                # Use timezone-aware datetimes, consistent with the rest of the app
                end_dt = datetime.now(timezone.utc)
                start_dt = end_dt - timedelta(days=days_back)

                all_bars = []
                chunk_start = start_dt

                while chunk_start < end_dt:
                    chunk_end = min(chunk_start + timedelta(days=400), end_dt)
                    try:
                        bars = sd.price_daily(
                            symbol=sym,
                            start_datetime=chunk_start,
                            end_datetime=chunk_end,
                        )
                        if isinstance(bars, dict):
                            bars = bars.get("candles", [])

                        if bars:
                            all_bars.extend(bars)
                    except Exception as inner_e:
                        # Log per-chunk errors but keep whatever we've collected so far
                        st.sidebar.write(f"Historical export error for {sym} (chunk): {inner_e}")
                        break

                    chunk_start += timedelta(days=400)
                    time.sleep(0.25)  # gentle rate limiting

                return all_bars or []
            except Exception as e:
                # Surface the error in the sidebar so the user can see why export failed
                st.sidebar.write(f"Historical export error for {sym}: {e}")
                return []

        if export_choice.startswith("Single CSV"):
            rows = []
            for i, sym in enumerate(symbols, 1):
                bars = fetch_all(sym)
                if bars:
                    for b in bars:
                        rows.append({
                            "symbol": sym,
                            "datetime": b.get("datetime") or b.get("date"),
                            "open": b.get("open"),
                            "high": b.get("high"),
                            "low": b.get("low"),
                            "close": b.get("close"),
                            "volume": b.get("volume"),
                        })
                else:
                    failed.append(sym)
                if i % 5 == 0 or i == total:
                    progress.progress(min(1.0, i / total), text=f"Fetching {i}/{total}…")

            if not rows:
                st.sidebar.info("No historical data returned.")
                return

            out_csv = f"exports/{ts}_historical_{base}.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            fail_log = f"exports/{ts}_historical_{base}_failed.txt"
            with open(fail_log, "w", encoding="utf-8") as lf:
                for s in failed:
                    lf.write(s + "\n")

            with open(out_csv, "rb") as f:
                st.sidebar.download_button(
                    "📦 Download Historical CSV", data=f,
                    file_name=out_csv.split("/")[-1], mime="text/csv",
                )
            with open(fail_log, "rb") as f:
                st.sidebar.download_button(
                    "🧾 Download Fail Log", data=f,
                    file_name=fail_log.split("/")[-1], mime="text/plain",
                )
            ok = total - len(failed)
            st.sidebar.success(f"Export complete: {ok} succeeded, {len(failed)} failed.")

        else: # ZIP
            zip_path = f"exports/{ts}_historical_{base}.zip"
            fail_log_name = f"{ts}_historical_{base}_failed.txt"

            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, sym in enumerate(symbols, 1):
                    bars = fetch_all(sym)
                    if bars:
                        mem = io.StringIO()
                        writer = csv.writer(mem)
                        writer.writerow(["datetime", "open", "high", "low", "close", "volume"])
                        for b in bars:
                            writer.writerow([
                                b.get("datetime") or b.get("date"),
                                b.get("open"), b.get("high"),
                                b.get("low"), b.get("close"),
                                b.get("volume")
                            ])
                        zf.writestr(f"{sym}.csv", mem.getvalue())
                    else:
                        failed.append(sym)
                    if i % 5 == 0 or i == total:
                        progress.progress(min(1.0, i / total), text=f"Fetching {i}/{total}…")
                zf.writestr(fail_log_name, "\n".join(failed))

            fail_log_disk = f"exports/{fail_log_name}"
            with open(fail_log_disk, "w", encoding="utf-8") as lf:
                for s in failed:
                    lf.write(s + "\n")

            with open(zip_path, "rb") as f:
                st.sidebar.download_button(
                    "📦 Download Historical ZIP", data=f,
                    file_name=zip_path.split("/")[-1], mime="application/zip",
                )
            with open(fail_log_disk, "rb") as f:
                st.sidebar.download_button(
                    "🧾 Download Fail Log", data=f,
                    file_name=fail_log_disk.split("/")[-1], mime="text/plain",
                )
            ok = total - len(failed)
            st.sidebar.success(f"Export complete: {ok} succeeded, {len(failed)} failed.")


def _first_env(*keys, default=""):
    for k in keys:
        v = os_mod.getenv(k, "")
        if v and v.strip():
            return v.strip()
    return default


class SchwabData:
    def __init__(self):
        self.cid = _first_env("SCHWAB_CLIENT_ID", "SCHWAB_APP_KEY")
        self.ck = _first_env("SCHWAB_CLIENT_KEY", "SCHWAB_CLIENT_SECRET")
        self.redir = _first_env(
            "SCHWAB_REDIRECT_URI", "SCHWAB_CALLBACK_URL", "CALLBACK_URL", "REDIRECT_URI",
            default="http://127.0.0.1:8000",
        )
        self.creds = _first_env("SCHWAB_CRED_PATH", default="data/.schwab_creds.json")
        self.port = int(_first_env("SCHWAB_PORT", default="8182"))
        os_mod.makedirs("data", exist_ok=True)

        if not self.cid or not self.ck:
            st.warning(
                "⚠️ Missing Schwab credentials. Set SCHWAB_CLIENT_ID and either "
                "SCHWAB_CLIENT_KEY or SCHWAB_CLIENT_SECRET in your .env"
            )
        self._cli = None
        self._tok = "data/schwab_tokens.json"
        self.signature_used = None

    def _ensure(self):
        if self._cli is not None:
            return
        errors = []
        def _try(fn, label):
            try:
                cli = fn()
                self.signature_used = label
                return cli
            except Exception as e:
                errors.append(f"{label}: {e}")
                return None

        # Try all known schwab-py signatures
        self._cli = _try(
            lambda: easy_client(
                api_key=self.cid, client_secret=self.ck, redirect_uri=self.redir,
                credentials_path=self.creds, token_path=self._tok,
                make_webdriver=lambda: None, headless=True, port=self.port,
            ), "v1:new-keywords")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(
                    app_key=self.cid, app_secret=self.ck, callback_url=self.redir,
                    creds_path=self.creds), "v2:classic-kw")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(
                    app_key=self.cid, app_secret=self.ck, redirect_uri=self.redir,
                    credentials_path=self.creds), "v3:classic-alt-kw")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(self.cid, self.ck, self.redir, self.creds),
                "v4:positional")

        if self._cli is None:
            st.error("Failed to initialize Schwab client. Tried multiple signatures:\n• "
                + "\n• ".join(errors))
            raise RuntimeError("Schwab easy_client initialization failed.")

    def price_daily(self, symbol, start_datetime=None, end_datetime=None):
        self._ensure()
        r = self._cli.get_price_history_every_day(
            symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=False,
            need_previous_close=False,
        )
        j = r.json()
        return j["candles"] if isinstance(j, dict) and "candles" in j else j

    def health_check(self, symbol="VOO"):
        try:
            self._ensure()
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=10)
            r = self.price_daily(symbol, start_datetime=start, end_datetime=end)
            n = len(r) if isinstance(r, list) else 0
            last_dt = None
            if n:
                ts = r[-1].get("datetime")
                if ts is not None:
                    last_dt = pd.to_datetime(ts, unit="ms", utc=True)
            return {
                "ok": True, "signature_used": self.signature_used, "symbol": symbol,
                "candles": n, "last_bar_utc": str(last_dt) if last_dt is not None else None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "signature_used": self.signature_used}

# Initialize one global client
sd = SchwabData()

# ==============================
# Index fetchers (Wikipedia) + caching
# ==============================
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


@st.cache_data(ttl=86400, show_spinner=False)
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
        st.warning(f"Index fetch failed for {index_name}: {e}")
        return []


# ==============================
# Sidebar Controls
# ==============================
with st.sidebar:
    st.header("Scan Options")
    index_choice = st.selectbox("Select Index Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=0) # S&P 100 default

    strategy = st.selectbox(
        "Strategy",
        [
            "All (run every strategy)",
            "RHCTS (recommended)",
            "ConnorsRSI Standalone",
            "CBC v1 (Volatility Breakout)",
            "Raptor II (Regime MR)",
            "High-Momentum Pullback",
            "CGM-10 (CustomGPT-ready)",
            "RJH Adaptive R2-X",
            "Mindful Trader",
        ],
        index=0,
    )
    
    RunScanClicked = st.button("Run Scan", type="primary", key="run_scan_sidebar")
    if RunScanClicked:
        st.session_state["__do_scan__"] = True

    st.sidebar.divider()
    render_historical_export_sidebar(index_choice)

    # Strategy Rules
    with st.expander("📘 Strategy Rules", expanded=False):
        if strategy.startswith("RHCTS") or strategy.startswith("All"):
            st.markdown("""**RHCTS**
- **Pre-filters**: ≥ 60 daily bars; 20-day avg volume **≥ 1,000,000**; Close **above** 50-SMA and **SMA50 rising**; RSI(14) **< 70**
- **Trigger**: **EMA20 touch + reclaim** OR **EMA20 reclaim from below**
- **Extra filter**: **CRSI ≤ 20**
- **Risk/Targets**: Entry: **EMA20**, Stop: **min(5-bar low, EMA20−1×ATR)**""")
        if strategy.startswith("ConnorsRSI") or strategy.startswith("All"):
            st.markdown("""**ConnorsRSI Standalone**
- **Pre-filters**: ≥ 60 bars; **CRSI ≤ 15**; Close **above** 200-SMA
- **Risk/Targets**: Entry: **close**, Stop: **2-ATR**, Target: **SMA20**""")
        if strategy.startswith("CBC v1") or strategy.startswith("All"):
            st.markdown("""**CBC v1 (Volatility Breakout)**
- **Pre-filters**: ≥ 60 bars; **ATR14 < ATR14’s 20-day avg** (volatility contraction); Yesterday close **within 2% below** 20-day high
- **Trigger**: Today close **> 20-day high** AND volume **≥ 1.3×** 20-day avg
- **Risk/Targets**: Entry: **close**, Stop: **max(recent swing low, breakout level − 1×ATR)**""")
        if strategy.startswith("Raptor II") or strategy.startswith("All"):
            st.markdown("""**Raptor II (Regime Mean-Reversion)**
- **Regime**: Uptrend: price **above** rising **100-EMA**
- **Trigger**: **RSI(2) ≤ 10** AND **Lower Bollinger(20,2) touch** AND **ATR squeeze**
- **Risk/Targets**: Entry: **close**, Stop: **recent 5-bar low**, Target: **middle band touch**""")
        if strategy.startswith("High-Momentum") or strategy.startswith("All"):
            st.markdown("""**High-Momentum Pullback (HMP)**
- **Trend**: **EMA50 > EMA200**, price **above** EMA50
- **Trigger**: **RSI(5) ≤ 35**
- **Risk/Targets**: Entry: **close**, Stop: **EMA50 − 1×ATR**""")
        if strategy.startswith("CGM-10") or strategy.startswith("All"):
            st.markdown("""**CGM-10 (Dual Strategy)**
**1. Breakaway Expansion (BX)**
- Price breaks **≥ 55-day high**
- ATR(14) **>** 20-day avg(ATR14)
- Volume **≥ 2.0 ×** 20-day avg volume
**2. Power Trend Pullback (PTP)**
- Trend: **EMA20 > SMA50** and **SMA50 rising**
- Pullback: **RSI(2) ≤ 5**
- Touch: Prior day low **≤** prior day EMA20
- Volume: Today's volume **≥ 1.5 ×** 20-day avg volume""")
        if strategy.startswith("RJH Adaptive") or strategy.startswith("All"):
            st.markdown("""**RJH Adaptive R2-X**
- **Regime**: **Close > SMA200**
- **Trigger**: Dynamic RSI(2) threshold: `IF ATR%20 > 2.5`: **RSI(2) < 5** `ELSE`: **RSI(2) < 10**
- **Risk/Targets**: Entry: **close**, Stop: **entry − 2×ATR10**, Time Stop: **10 bars**""")
        if strategy.startswith("Mindful Trader") or strategy.startswith("All"):
            st.markdown("""**Mindful Trader**
- **Trend Filter**: Stock must close **above its SMA20** for **10 consecutive** days.
- **Keltner Pierce**: High price must **exceed upper Keltner Channel** (SMA20 + 2.25 * ATR14) on at least one of those 10 days.
- **Entry**: Limit order at tomorrow's projected SMA20
- **Risk/Targets**: Stop: **Entry - 2 * ATR(14)**, Target: **Entry + 2 * ATR(14)**, Time Stop: **9 trading days**""")

# ==============================
# Data Helpers & Indicators
# ==============================
@st.cache_data(ttl=3600, show_spinner=False)
def _daily_cached(symbol, start, end):
    return sd.price_daily(symbol, start_datetime=start, end_datetime=end) or []

def _daily(symbol, start, end=None):
    return _daily_cached(symbol, start, end or datetime.now(timezone.utc))

def _safe_float(series, idx):
    try:
        v = float(series.iloc[idx])
        return v if pd.notna(v) else None
    except Exception:
        return None

def _atr14_manual(df):
    hl = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()

def _atr10_manual(df):
    hl = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift(1)).abs()
    l_pc = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(10, min_periods=10).mean()

def _features(df):
    df = df.copy()
    if "datetime" not in df.columns:
        return None # Bad data
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    
    if df.empty or len(df) < 20: # Need some data
        return None

    # Trend & momentum
    df["ema20"] = EMAIndicator(df["close"], 20).ema_indicator()
    df["sma5"] = SMAIndicator(df["close"], 5).sma_indicator()
    df["sma20"] = SMAIndicator(df["close"], 20).sma_indicator()
    df["sma50"] = SMAIndicator(df["close"], 50).sma_indicator()
    df["sma200"] = SMAIndicator(df["close"], 200).sma_indicator()
    df["ema50"] = EMAIndicator(df["close"], 50).ema_indicator()
    df["ema100"] = EMAIndicator(df["close"], 100).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], 200).ema_indicator()

    # Volatility & bands
    df["atr14"] = _atr14_manual(df)
    df["atr14_ma20"] = df["atr14"].rolling(20, min_periods=1).mean()
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_high"] = bb.bollinger_hband()

    # Keltner Channel for Mindful Trader
    df["keltner_upper"] = df["sma20"] + (2.25 * df["atr14"])

    # RSI
    df["rsi14"] = RSIIndicator(df["close"], 14).rsi()
    df["rsi5"] = RSIIndicator(df["close"], 5).rsi()
    df["rsi2"] = RSIIndicator(df["close"], 2).rsi()
    df["rsi3"] = RSIIndicator(df["close"], 3).rsi()

    # Rolling
    df["high20"] = df["high"].rolling(20).max()
    df["high55"] = df["high"].rolling(55).max()
    df["low20"] = df["low"].rolling(20).min()
    df["avgvol20"] = df["volume"].rolling(20).mean()

    # R2-X
    df["atr10"] = _atr10_manual(df)
    # --- MODIFIED: Added epsilon (1e-9) to prevent division by zero ---
    df["atr_pct10"] = (df["atr10"] / (df["close"] + 1e-9)) * 100.0
    # --- END MODIFIED ---
    df["atr_pct20"] = df["atr_pct10"].rolling(20, min_periods=20).mean()

    return df


def _crsi_series(df):
    return df["rsi2"] * 0.4 + df["rsi3"] * 0.4 + df["rsi14"] * 0.2


# ==============================
# Strategy Signals
# ==============================
def _signal_rhcts(d):
    if len(d) < 60: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    if not (pd.notna(last["avgvol20"]) and last["avgvol20"] >= 1_000_000): return None
    if not (last["close"] > last["sma50"] and d["sma50"].iloc[-1] > d["sma50"].iloc[-6]): return None
    if not (last["rsi14"] < 70): return None
    touch = (last["low"] <= last["ema20"]) and (last["close"] >= last["ema20"])
    reclaim = (prev["close"] < prev["ema20"]) and (last["close"] >= last["ema20"])
    if not (touch or reclaim): return None
    crsi = _crsi_series(d).iloc[-1]
    if not (pd.notna(crsi) and crsi <= 20): return None
    return "BUY"

def _signal_crsi(d):
    if len(d) < 200: return None
    last = d.iloc[-1]
    crsi = _crsi_series(d).iloc[-1]
    if not (pd.notna(crsi) and crsi <= 15): return None
    if not (last["close"] > last["sma200"]): return None
    return "BUY"

def _signal_cbc_v1(d):
    if len(d) < 60: return None
    last, prev = d.iloc[-1], d.iloc[-2]
    if not (pd.notna(last["atr14"]) and pd.notna(last["atr14_ma20"]) and last["atr14"] < last["atr14_ma20"]): return None
    high20_prev = d["high20"].iloc[-2]
    if not pd.notna(high20_prev): return None
    if not (prev["close"] >= 0.98 * high20_prev and prev["close"] <= high20_prev): return None
    if not (last["close"] > high20_prev): return None # Use high20_prev (yesterday's high)
    if not (pd.notna(last["avgvol20"]) and pd.notna(last["volume"]) and last["volume"] >= 1.3 * last["avgvol20"]): return None
    if pd.notna(d["rsi2"].iloc[-2]) and d["rsi2"].iloc[-2] > 20: return None # Check rsi *before*
    return "BUY"

def _signal_raptor2(d):
    if len(d) < 120: return None
    last = d.iloc[-1]
    ema100_prev5 = d["ema100"].iloc[-6] if len(d) >= 106 else None
    if not (pd.notna(last["ema100"]) and pd.notna(ema100_prev5) and last["ema100"] > ema100_prev5): return None
    if not (last["close"] > last["ema100"]): return None
    if not (pd.notna(last["rsi2"]) and last["rsi2"] <= 10): return None
    if not (pd.notna(last["bb_low"]) and last["close"] <= last["bb_low"]): return None
    if not (pd.notna(last["atr14"]) and pd.notna(last["atr14_ma20"]) and last["atr14"] < last["atr14_ma20"]): return None
    return "BUY"

def _signal_hmp(d):
    if len(d) < 200: return None
    last = d.iloc[-1]
    if not (pd.notna(last["ema50"]) and pd.notna(last["ema200"]) and last["ema50"] > last["ema200"]): return None
    if not (last["close"] > last["ema50"]): return None
    if not (pd.notna(last["rsi5"]) and last["rsi5"] <= 35): return None
    return "BUY"

def _signal_cgm10(d):
    if len(d) < 60: return None
    last, prev = d.iloc[-1], d.iloc[-2]

    # --- Strategy 1: Breakaway Expansion (BX) ---
    high55_prev = d["high55"].iloc[-2]
    bx_cond_1 = pd.notna(high55_prev) and last["high"] > high55_prev
    bx_cond_2 = pd.notna(last["atr14"]) and pd.notna(last["atr14_ma20"]) and last["atr14"] > last["atr14_ma20"]
    bx_cond_3 = pd.notna(last["volume"]) and pd.notna(last["avgvol20"]) and last["volume"] >= 2.0 * last["avgvol20"]
    if bx_cond_1 and bx_cond_2 and bx_cond_3:
        return "BUY"

    # --- Strategy 2: Power Trend Pullback (PTP) ---
    ptp_cond_1 = pd.notna(last["ema20"]) and pd.notna(last["sma50"]) and last["ema20"] > last["sma50"] and d["sma50"].iloc[-1] > d["sma50"].iloc[-6]
    ptp_cond_2 = pd.notna(last["rsi2"]) and last["rsi2"] <= 5
    ptp_cond_3 = pd.notna(prev["low"]) and pd.notna(prev["ema20"]) and prev["low"] <= prev["ema20"]
    ptp_cond_4 = pd.notna(last["volume"]) and pd.notna(last["avgvol20"]) and last["volume"] >= 1.5 * last["avgvol20"]
    if ptp_cond_1 and ptp_cond_2 and ptp_cond_3 and ptp_cond_4:
        return "BUY"
    
    return None

def _signal_rjh_r2x(d):
    if len(d) < 200: return None
    last = d.iloc[-1]
    if not (pd.notna(last["sma200"]) and last["close"] > last["sma200"]): return None
    atr_pct20 = last.get("atr_pct20")
    if not pd.notna(atr_pct20): return None
    threshold = 5.0 if atr_pct20 > 2.5 else 10.0
    if not (pd.notna(last["rsi2"]) and last["rsi2"] < threshold): return None
    return "BUY"

def _signal_mindful_trader(d):
    if len(d) < 34: return None
    trend_window = d.iloc[-10:]
    if len(trend_window) < 10: return None
    required_cols = ["close", "sma20", "high", "keltner_upper"]
    if not all(col in trend_window.columns for col in required_cols): return None
    if trend_window[required_cols].isna().any().any(): return None
    trend_ok = (trend_window["close"] > trend_window["sma20"]).all()
    if not trend_ok: return None
    pierce_ok = (trend_window["high"] > trend_window["keltner_upper"]).any()
    if not pierce_ok: return None
    return "BUY"

# ==============================
# Entry/Target/Stop
# ==============================
def _rank_and_targets(d, mode: str):
    last = d.iloc[-1]
    close = float(last["close"])
    atr = _safe_float(d["atr14"], -1) or 1.0

    if mode == "RHCTS":
        ema20 = _safe_float(d["ema20"], -1) or close
        mid = round(ema20, 2)
        recent_low = _safe_float(d["low"].rolling(5).min(), -2) or close
        stop = round(min(recent_low, (ema20 - 1.0 * atr)), 2)
        risk = max(0.01, mid - stop)
        t1 = round(mid + 2.0 * risk, 2)
        strength = round(max(0.0, 1.5 - min(1.5, abs(close - ema20) / max(atr, 1e-9))), 2)
        eta = int(math.ceil(max(0.0, t1 - close) / max(atr, 1e-9))) if (atr and t1 > close) else 0
        apct = round((t1 - mid) / max(1e-9, mid) * 100.0, 2)
        return mid, t1, apct, eta, strength

    if mode == "CRSI":
        mid = round(close, 2)
        stop = round(mid - 2 * atr, 2) # 2-ATR stop
        risk = max(0.01, mid - stop)
        t1 = round(mid + 2.0 * risk, 2)
        eta = int(math.ceil(max(0.0, t1 - close) / max(atr, 1e-9))) if (atr and t1 > close) else 0
        apct = round((t1 - mid) / max(1e-9, mid) * 100.0, 2)
        return mid, t1, apct, eta, 1.0

    if mode == "CBC":
        breakout = _safe_float(d["high20"], -2) or close # Breakout from *yesterday's* high
        entry = round(close, 2)
        stop = round(max(_safe_float(d["low"].rolling(20).min(), -2) or close, breakout - 1.0 * atr), 2)
        risk = max(0.01, entry - stop)
        t1 = round(entry + 2.0 * risk, 2)
        eta = int(math.ceil(max(0.0, t1 - entry) / max(atr, 1e-9))) if (atr and t1 > entry) else 0
        apct = round((t1 - entry) / max(1e-9, entry) * 100.0, 2)
        strength = round(min(1.5, max(0.0, (close - breakout) / max(atr, 1e-9))), 2)
        return entry, t1, apct, eta, strength

    if mode == "RAPTOR":
        entry = round(close, 2)
        stop = round(_safe_float(d["low"].rolling(5).min(), -2) or close, 2)
        risk = max(0.01, entry - stop)
        t1 = round(_safe_float(d["bb_mid"], -1) or (entry + 2.0 * risk), 2) # Target middle band
        eta = int(math.ceil(max(0.0, t1 - entry) / max(atr, 1e-9))) if (atr and t1 > entry) else 0
        apct = round((t1 - entry) / max(1e-9, entry) * 100.0, 2)
        lb = _safe_float(d["bb_low"], -1) or close
        strength = round(min(1.5, max(0.0, (lb - close) / max(atr, 1e-9) + 1.0)), 2)
        return entry, t1, apct, eta, strength

    if mode == "HMP":
        ema50 = _safe_float(d["ema50"], -1) or close
        entry = round(close, 2)
        stop = round(ema50 - 1.0 * atr, 2)
        risk = max(0.01, entry - stop)
        t1 = round(entry + 2.0 * risk, 2)
        eta = int(math.ceil(max(0.0, t1 - entry) / max(atr, 1e-9))) if (atr and t1 > entry) else 0
        apct = round((t1 - entry) / max(1e-9, entry) * 100.0, 2)
        strength = 1.0
        return entry, t1, apct, eta, strength

    if mode == "CGM10":
        entry = round(close, 2)
        stop = entry - 2.5 * atr
        risk = max(0.01, entry - stop)
        target = round(entry + 2.0 * risk, 2)
        eta = int(math.ceil(max(0.0, target - entry) / max(atr, 1e-9)))
        apct = round((target - entry) / max(1e-9, entry) * 100.0, 2)
        strength = 1.0
        return entry, target, apct, eta, strength

    if mode == "R2X":
        atr10 = _safe_float(d["atr10"], -1) or 1.0
        entry = round(close, 2)
        stop = round(entry - 2.0 * atr10, 2)
        risk = max(0.01, entry - stop)
        t1 = round(entry + 2.0 * risk, 2)
        eta = int(math.ceil(max(0.0, t1 - entry) / max(atr10, 1e-9)))
        apct = round((t1 - entry) / max(1e-9, entry) * 100.0, 2)
        strength = (10.0 - min(10.0, _safe_float(d["rsi2"], -1) or 10.0)) / 10.0
        return entry, t1, apct, eta, round(strength, 2)

    if mode == "MINDFUL":
        sma20_t0 = _safe_float(d["sma20"], -1)
        sma20_t1 = _safe_float(d["sma20"], -2)
        if sma20_t0 is None or sma20_t1 is None:
            entry = round(close, 2)
        else:
            projected_entry = sma20_t0 + (sma20_t0 - sma20_t1)
            entry = round(projected_entry, 2)
        stop = round(entry - (2.0 * atr), 2)
        target = round(entry + (2.0 * atr), 2)
        apct = round((target - entry) / max(1e-9, entry) * 100.0, 2)
        eta = 9  # Per rules
        strength = 1.0
        return entry, target, apct, eta, strength

    return round(close, 2), round(close + atr, 2), 2.0, 0, 1.0


# ==============================
# Scan Function
# ==============================
def _resolve_index_universe():
    syms = get_index_symbols(index_choice) or []
    seen = set()
    ordered = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def _signal_for_strategy(d, name: str):
    if name.startswith("RHCTS"): return _signal_rhcts(d), "RHCTS"
    elif name.startswith("ConnorsRSI"): return _signal_crsi(d), "CRSI"
    elif name.startswith("CBC v1"): return _signal_cbc_v1(d), "CBC"
    elif name.startswith("Raptor II"): return _signal_raptor2(d), "RAPTOR"
    elif name.startswith("High-Momentum"): return _signal_hmp(d), "HMP"
    elif name.startswith("CGM-10"): return _signal_cgm10(d), "CGM10"
    elif name.startswith("RJH Adaptive"): return _signal_rjh_r2x(d), "R2X"
    elif name.startswith("Mindful Trader"): return _signal_mindful_trader(d), "MINDFUL"
    else: return None, "CRSI"


def _all_strategy_signals(d):
    results = []
    for label in [
        "RHCTS (recommended)", "ConnorsRSI Standalone", "CBC v1 (Volatility Breakout)",
        "Raptor II (Regime MR)", "High-Momentum Pullback", "CGM-10 (CustomGPT-ready)",
        "RJH Adaptive R2-X", "Mindful Trader",
    ]:
        sig, mode = _signal_for_strategy(d, label)
        if sig == "BUY":
            results.append(mode)
    return results


def run_scan():
    symbols = _resolve_index_universe()
    if not symbols:
        st.warning(f"No symbols found for {index_choice}. Check your internet connection.")
        return

    st.write(f"**Universe: {index_choice}** — {len(symbols)} symbols")
    
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=420)
    rows = []
    scanned = 0
    qualified = 0
    errors = []
    
    progress_bar = st.progress(0, "Scanning...")

    for i, sym in enumerate(symbols):
        scanned += 1
        try:
            bars = _daily(sym, start)
            if not bars: continue
            df = pd.DataFrame(bars)
            if not {"open", "high", "low", "close", "volume", "datetime"}.issubset(df.columns):
                continue
            fx = _features(df)
            if fx is None or fx.empty:
                continue

            modes_to_add = []
            if strategy.startswith("All"):
                modes_to_add = _all_strategy_signals(fx)
            else:
                sig, mode = _signal_for_strategy(fx, strategy)
                if sig == "BUY":
                    modes_to_add = [mode]

            for mode in modes_to_add:
                entry, target, prof, eta, strength = _rank_and_targets(fx, mode)
                qualified += 1
                rows.append({
                    "symbol": sym, "mode": mode,
                    "date": str(pd.Timestamp.now(tz="UTC").normalize().date()),
                    "entry": entry, "exit": target, "anticipated_profit_%": prof,
                    "eta_days": eta, "strength": strength,
                    "avgvol20": float(fx["avgvol20"].iloc[-1]) if pd.notna(fx["avgvol20"].iloc[-1]) else 0.0,
                })
        except Exception as e:
            errors.append(f"{sym}: {e}")
            continue
        finally:
            progress_bar.progress((i + 1) / len(symbols), f"Scanning {sym} ({i+1}/{len(symbols)})")

    progress_bar.empty()
    st.write(f"**Scan summary:** scanned {scanned} symbols • qualifying setups: {qualified}")

    if errors:
        with st.expander(f"Diagnostics (errors on {len(errors)} symbols)", expanded=False):
            st.code("\n".join(errors[-100:]))

    if not rows:
        st.info("No qualifying setups today.")
        st.session_state["base_df"] = pd.DataFrame()
        return

    base_df = pd.DataFrame(rows)
    st.session_state["base_df"] = base_df.copy() # Persist for download
    
    # Rank
    sort_cols = ["strength", "anticipated_profit_%", "eta_days", "avgvol20"]
    sort_asc = [False, False, True, False]
    base_df = base_df.sort_values(sort_cols, ascending=sort_asc).set_index("symbol")
    st.session_state["base_df"] = base_df.copy() # Persist sorted

    st.subheader("🏁 Ranked BUY Setups (shown above in Scanner Output)")

    os_mod.makedirs("data", exist_ok=True)
    out_csv = "data/signals_ranked.csv"
    base_df.reset_index().to_csv(out_csv, index=False)
    with open(out_csv, "rb") as f:
        st.download_button("⬇️ Download CSV", data=f, file_name="signals_ranked.csv", mime="text/csv")


# ==============================
# API Health Check UI
# ==============================
st.divider()
st.subheader("🔎 Schwab API Health Check")
colA, colB = st.columns([1, 2])
with colA:
    test_symbol = st.text_input("Test symbol", value="VOO")
if st.button("Run API Check"):
    with st.spinner("Verifying Schwab credentials and fetching recent data…"):
        info = sd.health_check(test_symbol.strip().upper() or "VOO")
    if info.get("ok"):
        st.success("API OK")
        st.json(info)
    else:
        st.error("API check failed")
        st.json(info)

# ==============================
# NEW Backtest Engine UI
# ==============================
st.divider()
st.subheader("🔁 Backtest Engine (Backtrader + Schwab API)")

# --- Data Provider Function (FIXED) ---
@st.cache_data(ttl=3600, show_spinner=False) # Cache data loads
def schwab_data_provider_for_backtrader(symbol: str, start_date: date):
    """
    Returns a cleaned pandas DataFrame of OHLCV for `symbol`
    suitable for Backtrader, using the authenticated Schwab client `sd`.
    
    CRITICAL FIX: Adds 'openinterest' column to satisfy Backtrader.
    """
    try:
        user_requested_start = datetime.combine(
            start_date, datetime.min.time(), tzinfo=timezone.utc
        )
        # Need 300 days of data *before* the start_date for indicators to warm up
        warmup_start = user_requested_start - timedelta(days=300)
        now_utc = datetime.now(timezone.utc)
        all_bars = []
        chunk_start = warmup_start
        symbol_for_api = symbol.replace(".", "/")

        while chunk_start < now_utc:
            chunk_end = min(chunk_start + timedelta(days=400), now_utc)
            try:
                bars = sd.price_daily(
                    symbol=symbol_for_api,
                    start_datetime=chunk_start,
                    end_datetime=chunk_end,
                )
                if not bars and symbol_for_api != symbol:
                    bars = sd.price_daily(
                        symbol=symbol,
                        start_datetime=chunk_start,
                        end_datetime=chunk_end,
                    )
                if bars:
                    if isinstance(bars, dict):
                        bars = bars.get("candles", [])
                    all_bars.extend(bars)
            except Exception as e:
                msg = str(e)
                if "404" in msg or "Not Found" in msg:
                    print(f"No data for {symbol} (404)")
                else:
                    print(f"Error fetching chunk for {symbol}: {e}")
                break
            
            chunk_start += timedelta(days=400)
            time.sleep(0.3) # Rate limit

        if not all_bars:
            print(f"No data returned for {symbol}")
            return None

        df = pd.DataFrame(all_bars)
        if df.empty:
            print(f"{symbol} DataFrame empty after construction")
            return None

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
            time_col = "datetime"
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], unit="ms", utc=True)
            time_col = "date"
        else:
            print(f"{symbol} missing datetime column")
            return None

        df = df.drop_duplicates(subset=[time_col])
        df = df.set_index(time_col)
        
        wanted_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in wanted_cols if col not in df.columns]
        if missing:
            print(f"{symbol} missing one of OHLCV columns: {missing}")
            return None

        df = df[wanted_cols].astype(float)
        
        # --- MODIFIED: Data Validation ---
        # Filter out zero volume bars (you already had this, which is great)
        df = df[df["volume"] > 0]
        # --- ADDED: Filter out "flat" bars (high == low) ---
        # This is a primary cause of ZeroDivisionError in ATR calculations
        df = df[df['high'] > df['low']]
        # --- END MODIFIED ---
        
        df = df.dropna()
        df = df.sort_index(ascending=True)

        # --- CRITICAL FIX ---
        # Add openinterest column, Backtrader feed requires it
        df['openinterest'] = 0.0
        # --------------------

        # Enforce warmup window
        df = df[df.index >= warmup_start]
        
        if df.empty:
            print(f"{symbol} cleaned down to 0 rows")
            return None
        if df.index.max() < user_requested_start:
            print(f"{symbol} has no data on/after backtest start")
            return None

        return df

    except Exception as e:
        print(f"Error in schwab_data_provider_for_backtrader for {symbol}: {e}")
        return None


# --- Backtest UI ---

# Advanced performance settings (Apple M3 Max optimization)
cpu_count = os_mod.cpu_count() or 1
default_workers = min(max(1, cpu_count - 1), 8)

col_bt1, col_bt2 = st.columns(2)
with col_bt1:
    bt_use_parallel = st.checkbox(
        "Use parallel execution (faster, recommended)",
        value=True,
        help="Run each strategy on a separate worker thread so NumPy/Pandas can fully use your M3 Max cores.",
    )
with col_bt2:
    bt_max_workers = st.slider(
        "Max worker threads",
        min_value=1,
        max_value=8,
        value=default_workers,
        help="Upper limit on concurrent strategy runs. Defaults to using most of your performance cores while keeping 1 core free.",
    )

bt_start_date = st.date_input("Start Date", value=date(2014, 1, 1)) # Changed to 2014
bt_start_cash = 100000.0

if st.button("Run Backtest", key="run_backtest_engine"):
    symbols = _resolve_index_universe()
    if not symbols:
        st.error("No symbols found for the selected universe. Cannot run backtest.")
    else:
        st.write(f"Backtesting on **{len(symbols)}** symbols from **{index_choice}**...")

        # 2. Build data_dict
        data_dict = {}
        progress_bar = st.progress(0, "Loading historical data...")
        with st.spinner(f"Loading data for {len(symbols)} symbols..."):
            for i, symbol in enumerate(symbols):
                try:
                    df = schwab_data_provider_for_backtrader(symbol, bt_start_date)
                    if df is not None and not df.empty:
                        data_dict[symbol] = df
                    else:
                        print(f"No data loaded for {symbol}, skipping.")
                except Exception as e:
                    print(f"Error loading {symbol}: {e}")
                
                progress_bar.progress((i + 1) / len(symbols), f"Loading data... ({i+1}/{len(symbols)})")
        
        progress_bar.empty()

        if not data_dict:
            st.error("Failed to load historical data for any symbols. Aborting backtest.")
        else:
            # 3. Provide strategy names
            strategy_names = [
                "RHCTS", "CRSI", "CBC", "RAPTOR",
                "HMP", "CGM10", "R2X", "MINDFUL",
            ]

            # 4. Run the comparison
            try:
                with st.spinner("Running Backtrader engine... This may take several minutes."):
                    results_df = run_compare(
                        strategy_names=strategy_names,
                        data_dict=data_dict,
                        symbol_universe=symbols,
                        start_cash=bt_start_cash,
                        start_date=bt_start_date,  # Pass start_date to runner
                        use_parallel=bt_use_parallel,
                        max_workers=bt_max_workers,
                    )

                if results_df is None or results_df.empty:
                    st.warning("Backtest ran but produced no results.")
                else:
                    # 5. Fix returned result columns (FIX 5)
                    st.subheader("Strategy Performance")
                    
                    display_df = pd.DataFrame()
                    display_df["Strategy"] = results_df["strategy"]
                    # cagr is decimal, convert to %
                    display_df["CAGR %"] = (results_df["cagr"].fillna(0) * 100.0).round(2)
                    display_df["Sharpe"] = results_df["sharpe"].fillna(0).round(3)
                    display_df["Max DD %"] = results_df["max_drawdown_pct"].fillna(0).round(2)
                    display_df["Hit %"] = results_df["hit_rate"].fillna(0).round(2)
                    display_df["Total Trades"] = results_df["total_trades"].fillna(0).astype(int)
                    display_df["Score"] = results_df["Score"].fillna(0).round(3)
                    display_df["End Value"] = results_df["final_value"].fillna(bt_start_cash).round(2)

                    # 6. Ensure Streamlit UI uses new fields
                    display_df = display_df.sort_values(by="Score", ascending=False)
                    
                    st.dataframe(
                        display_df.set_index("Strategy"),
                        use_container_width=True,
                    )
                    
                    # Also show raw results
                    with st.expander("Show Raw Metrics DataFrame"):
                        st.dataframe(results_df)

            except Exception as e:
                st.error(f"Backtest engine failed: {e}")
                import traceback
                st.code(traceback.format_exc())

# --- Deferred Run Scan trigger ---
if st.session_state.get("__do_scan__"):
    try:
        run_scan()
    finally:
        st.session_state["__do_scan__"] = False

# --- Render the Scanner Output ---
with scan_out_container:
    if (
        "base_df" in st.session_state
        and isinstance(st.session_state["base_df"], pd.DataFrame)
        and not st.session_state["base_df"].empty
    ):
        col_header, col_actions = st.columns([4, 1])
        with col_header:
            st.subheader("Scanner Output")
        with col_actions:
            df = st.session_state["base_df"]
            df_csv = df.reset_index() # For export

            # Copy button
            csv_for_js = df_csv.to_csv(index=False).replace("`", "\\`").replace("\n", "\\n")
            st.markdown(
                f"""
            <script>
            function copyToClipboard() {{
                const text = `{csv_for_js}`;
                try {{
                    navigator.clipboard.writeText(text);
                    alert("✅ Scan results copied to clipboard!");
                }} catch (err) {{
                    alert("❌ Copy failed. Please use the Download button.");
                }}
            }}
            </script>
            <button onclick="copyToClipboard()" style="width:100%;border-radius:0.5rem;padding:0.25rem 0.75rem;border:1px solid #ddd;background:white;">
                📋 Copy
            </button>
            """,
                unsafe_allow_html=True,
            )

            # Download button
            st.download_button(
                label="📥 Download CSV",
                data=df_csv.to_csv(index=False).encode("utf-8"),
                file_name=f"RJH_Scan_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.dataframe(df, use_container_width=True)
        st.success(f"✅ Scan complete — {len(df)} qualifying setups found.")
    else:
        st.caption("Run a scan to see results here.")