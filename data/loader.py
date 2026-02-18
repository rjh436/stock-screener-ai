import pandas as pd
import numpy as np
import os
import time
import io
import json
from datetime import datetime, timedelta, timezone
import pytz
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
from .schwab_client import sd
from .cache_manager import DataCache
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

try:
    import requests
except Exception:
    requests = None

_MISS_CACHE_DIR = os.path.join("data", "cache", "_miss")
_SYMBOL_ALIAS_CACHE: Optional[Dict[str, List[str]]] = None
_SYMBOL_ALIAS_LOCK = threading.Lock()
_FALLBACK_HTTP = requests.Session() if requests is not None else None


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _fallback_enabled() -> bool:
    # Default ON for better historical coverage; can be disabled via env.
    return _truthy_env("DATA_ENABLE_FALLBACK_HISTORY", default=True)


def _fallback_timeout_seconds() -> float:
    try:
        v = float(os.getenv("DATA_FALLBACK_TIMEOUT_SEC", "8.0") or "8.0")
    except Exception:
        v = 8.0
    return max(2.0, min(v, 30.0))


def _fallback_max_workers() -> int:
    try:
        v = int(os.getenv("DATA_FALLBACK_MAX_WORKERS", "8") or "8")
    except Exception:
        v = 8
    return max(1, min(v, 24))


_FALLBACK_SEM = threading.BoundedSemaphore(_fallback_max_workers())


def _fallback_min_bars() -> int:
    try:
        v = int(os.getenv("DATA_FALLBACK_MIN_BARS", "40") or "40")
    except Exception:
        v = 40
    return max(20, min(v, 252))


def _fallback_user_agent() -> str:
    return os.getenv("DATA_FALLBACK_USER_AGENT", "Mozilla/5.0 (ApexSniper fallback loader)")


def _load_symbol_aliases() -> Dict[str, List[str]]:
    global _SYMBOL_ALIAS_CACHE
    if _SYMBOL_ALIAS_CACHE is not None:
        return _SYMBOL_ALIAS_CACHE

    with _SYMBOL_ALIAS_LOCK:
        if _SYMBOL_ALIAS_CACHE is not None:
            return _SYMBOL_ALIAS_CACHE
        path = os.getenv(
            "DATA_SYMBOL_ALIASES_PATH",
            os.path.join("data", "russell3000_membership", "symbol_aliases.json"),
        )
        out: Dict[str, List[str]] = {}
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    for k, v in payload.items():
                        key = str(k or "").strip().upper()
                        if not key:
                            continue
                        vals: List[str] = []
                        if isinstance(v, list):
                            vals = [str(x).strip().upper() for x in v if str(x).strip()]
                        elif isinstance(v, str):
                            vv = v.strip().upper()
                            vals = [vv] if vv else []
                        if vals:
                            out[key] = vals
        except Exception:
            out = {}
        _SYMBOL_ALIAS_CACHE = out
    return out


def _symbol_aliases(sym: str) -> List[str]:
    s = str(sym or "").strip().upper()
    if not s:
        return []
    aliases = [s]

    def _add(val: str) -> None:
        v = str(val or "").strip().upper()
        if not v:
            return
        if v not in aliases:
            aliases.append(v)

    for mapped in _load_symbol_aliases().get(s, []):
        _add(mapped)

    # Common punctuation variants across providers.
    _add(s.replace(".", "/"))
    _add(s.replace("/", "."))
    _add(s.replace("/", "-"))
    _add(s.replace("-", "/"))
    _add(s.replace(".", "-"))
    _add(s.replace("-", "."))
    _add(s.replace(".", "").replace("/", "").replace("-", ""))
    return aliases


def _missing_symbol_path(sym: str, *, scope: str = "base") -> str:
    scope_key = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(scope or "base").lower()) or "base"
    key = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(sym or "").upper())
    if not key:
        key = "UNKNOWN"
    return os.path.join(_MISS_CACHE_DIR, f"{scope_key}__{key}.miss")


def _missing_symbol_ttl_seconds() -> int:
    try:
        ttl_hours = float(os.getenv("DATA_MISSING_RETRY_HOURS", "24") or "24")
    except Exception:
        ttl_hours = 24.0
    ttl_hours = max(1.0, min(ttl_hours, 24.0 * 14.0))
    return int(ttl_hours * 3600)


def _is_recent_missing_symbol(sym: str, *, scope: str = "base") -> bool:
    path = _missing_symbol_path(sym, scope=scope)
    if not os.path.exists(path):
        return False
    ttl = _missing_symbol_ttl_seconds()
    try:
        age = max(0.0, time.time() - os.path.getmtime(path))
    except Exception:
        return False
    if age <= ttl:
        return True
    try:
        os.remove(path)
    except Exception:
        pass
    return False


def _mark_missing_symbol(sym: str, *, scope: str = "base") -> None:
    try:
        os.makedirs(_MISS_CACHE_DIR, exist_ok=True)
        path = _missing_symbol_path(sym, scope=scope)
        with open(path, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass


def _clear_missing_symbol(sym: str, *, scope: str = "base") -> None:
    path = _missing_symbol_path(sym, scope=scope)
    if not os.path.exists(path):
        return
    try:
        os.remove(path)
    except Exception:
        pass


def clean_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        if df is None or df.empty:
            return None
        df.columns = df.columns.str.lower()

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
            elif "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df.set_index("datetime", inplace=True)
            else:
                df.index = pd.to_datetime(df.index)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df = df.sort_index()
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df[df["high"] > df["low"]]
        if len(df) < 20:
            return None
        return df
    except Exception:
        return None


def _as_price(val) -> Optional[float]:
    try:
        val = float(val)
    except Exception:
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    return val


def _extract_quote_fields(quote: Dict, sym: str) -> Tuple[Optional[float], Optional[float], float]:
    if not isinstance(quote, dict):
        return None, None, 0.0
    sym_data = quote.get(sym) or quote.get(sym.upper())
    if not isinstance(sym_data, dict):
        return None, None, 0.0
    q_data = sym_data.get("quote") if isinstance(sym_data.get("quote"), dict) else sym_data
    if not isinstance(q_data, dict):
        return None, None, 0.0
    open_price = _as_price(q_data.get("openPrice") or q_data.get("open"))
    last_price = _as_price(
        q_data.get("lastPrice")
        or q_data.get("mark")
        or q_data.get("markPrice")
        or q_data.get("closePrice")
    )
    try:
        volume = float(q_data.get("totalVolume") or q_data.get("volume") or 0.0)
    except Exception:
        volume = 0.0
    return open_price, last_price, volume


def inject_live_quote(df: pd.DataFrame, sym: str, prefetched_quote: Optional[Dict] = None) -> pd.DataFrame:
    """Append a synthetic bar using live quote data when today's bar is missing."""
    if df is None or df.empty:
        return df
    q = prefetched_quote
    if q is None:
        try:
            q = sd.get_quote(sym)
        except Exception:
            return df
    if q is None:
        return df
    if isinstance(q, dict) and sym not in q and sym.upper() not in q:
        q = {sym: q}

    open_price, last_price, volume = _extract_quote_fields(q, sym)
    if open_price is None and last_price is None:
        return df

    try:
        last_date = df.index.max().date()
    except Exception:
        return df

    # FIX: Use NY time to prevent UTC rollover issues
    today = datetime.now(ZoneInfo("America/New_York")).date()
    if last_date == today:
        return df

    open_val = open_price or last_price
    close_val = last_price or open_val
    if open_val is None or close_val is None:
        return df

    high_val = max(open_val, close_val)
    low_val = min(open_val, close_val)
    row_ts = pd.Timestamp(today)

    # FIX: Add is_live_bar flag
    live_row = pd.DataFrame(
        {
            "open": [open_val],
            "high": [high_val],
            "low": [low_val],
            "close": [close_val],
            "volume": [volume],
            "is_live_bar": [True]
        },
        index=[row_ts],
    )
    df = pd.concat([df, live_row])
    df = df[~df.index.duplicated(keep="last")]
    return df.sort_index()


def _required_history_start(days: int, *, buffer_days: int = 100) -> datetime:
    try:
        day_count = int(days)
    except Exception:
        day_count = 1260
    day_count = max(1, day_count)
    return (datetime.now(timezone.utc) - timedelta(days=day_count + max(0, int(buffer_days)))).replace(tzinfo=None)


def _has_required_history(
    df: Optional[pd.DataFrame],
    required_start: datetime,
    *,
    tolerance_days: int = 30,
) -> bool:
    if df is None or df.empty:
        return False
    if not isinstance(df.index, pd.DatetimeIndex):
        return False
    try:
        first_date = pd.Timestamp(df.index.min()).tz_localize(None)
    except Exception:
        return False
    return first_date <= (required_start + timedelta(days=max(0, int(tolerance_days))))


_SOURCE_HITS_LOCK = threading.Lock()
_SOURCE_HITS: Dict[str, int] = {}


def _note_source_hit(source: str) -> None:
    key = str(source or "").strip().lower()
    if not key:
        return
    with _SOURCE_HITS_LOCK:
        _SOURCE_HITS[key] = int(_SOURCE_HITS.get(key, 0)) + 1


def _source_hits_snapshot() -> Dict[str, int]:
    with _SOURCE_HITS_LOCK:
        return dict(_SOURCE_HITS)


def _source_hits_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    keys = set(before.keys()) | set(after.keys())
    out: Dict[str, int] = {}
    for k in keys:
        d = int(after.get(k, 0)) - int(before.get(k, 0))
        if d > 0:
            out[k] = d
    return out


def _schwab_alias_attempts() -> int:
    try:
        v = int(os.getenv("DATA_SCHWAB_ALIAS_ATTEMPTS", "4") or "4")
    except Exception:
        v = 4
    return max(1, min(v, 10))


def _to_frame_from_candles(candles: List[Dict]) -> Optional[pd.DataFrame]:
    if not candles:
        return None
    df = pd.DataFrame(candles)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
        df = df.set_index("datetime")
    return clean_dataframe(df)


def _fetch_schwab_candles_with_aliases(
    sym: str,
    *,
    start_datetime: datetime,
    end_datetime: datetime,
) -> Tuple[Optional[List[Dict]], str]:
    aliases = _symbol_aliases(sym)
    max_tries = _schwab_alias_attempts()
    aliases = aliases[:max_tries]
    last_exc: Optional[Exception] = None
    for alias in aliases:
        try:
            candles = sd.price_daily(alias, start_datetime=start_datetime, end_datetime=end_datetime)
            if candles:
                if alias != str(sym).upper():
                    _note_source_hit("schwab_alias")
                _note_source_hit("schwab")
                return candles, alias
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise RuntimeError(str(last_exc))
    return None, ""


def _safe_epoch_seconds(ts: datetime) -> int:
    try:
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return int(t.timestamp())
    except Exception:
        return int(time.time()) - 86400


def _normalize_ohlcv_frame(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    cols = {str(c).lower(): c for c in df.columns}
    mapped = {}
    for want in ("open", "high", "low", "close", "volume"):
        src = cols.get(want)
        if src is not None:
            mapped[want] = df[src]
    if len(mapped) < 4:
        return None
    out = pd.DataFrame(mapped)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in out.columns:
            out[c] = np.nan if c != "volume" else 0.0
    return clean_dataframe(out)


def _download_yahoo_history(alias: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    if _FALLBACK_HTTP is None:
        return None
    symbol = str(alias or "").strip().upper().replace("/", "-")
    if not symbol:
        return None
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": "1d",
        "period1": str(_safe_epoch_seconds(start)),
        "period2": str(max(_safe_epoch_seconds(end), _safe_epoch_seconds(start) + 86400)),
        "events": "history",
        "includeAdjustedClose": "true",
    }
    try:
        with _FALLBACK_SEM:
            resp = _FALLBACK_HTTP.get(
                url,
                params=params,
                headers={"User-Agent": _fallback_user_agent()},
                timeout=_fallback_timeout_seconds(),
            )
        if resp.status_code != 200:
            return None
        payload = resp.json()
        chart = (payload or {}).get("chart") or {}
        if chart.get("error"):
            return None
        result = chart.get("result") or []
        if not result:
            return None
        item = result[0] or {}
        timestamps = item.get("timestamp") or []
        if not timestamps:
            return None
        quote = ((item.get("indicators") or {}).get("quote") or [{}])[0] or {}
        n = len(timestamps)

        def _fit_len(values):
            vals = list(values or [])
            if len(vals) < n:
                vals.extend([None] * (n - len(vals)))
            elif len(vals) > n:
                vals = vals[:n]
            return vals

        df = pd.DataFrame(
            {
                "open": _fit_len(quote.get("open")),
                "high": _fit_len(quote.get("high")),
                "low": _fit_len(quote.get("low")),
                "close": _fit_len(quote.get("close")),
                "volume": _fit_len(quote.get("volume")),
            },
            index=pd.to_datetime(pd.Series(timestamps), unit="s", utc=True),
        )
        return _normalize_ohlcv_frame(df)
    except Exception:
        return None


def _download_stooq_history(alias: str) -> Optional[pd.DataFrame]:
    if _FALLBACK_HTTP is None:
        return None
    symbol = str(alias or "").strip().lower().replace("/", "-").replace(".", "-")
    if not symbol:
        return None
    url = f"https://stooq.com/q/d/l/?s={symbol}.us&i=d"
    try:
        with _FALLBACK_SEM:
            resp = _FALLBACK_HTTP.get(
                url,
                headers={"User-Agent": _fallback_user_agent()},
                timeout=_fallback_timeout_seconds(),
            )
        if resp.status_code != 200:
            return None
        text = resp.text
        if "Date,Open,High,Low,Close,Volume" not in text:
            return None
        parsed = pd.read_csv(io.StringIO(text))
        if parsed is None or parsed.empty:
            return None
        if "Date" not in parsed.columns:
            return None
        parsed["Date"] = pd.to_datetime(parsed["Date"], errors="coerce")
        parsed = parsed.dropna(subset=["Date"]).set_index("Date")
        return _normalize_ohlcv_frame(parsed)
    except Exception:
        return None


def _fallback_providers() -> List[str]:
    raw = str(os.getenv("DATA_FALLBACK_PROVIDERS", "yahoo,stooq") or "yahoo,stooq")
    out: List[str] = []
    for item in raw.split(","):
        p = item.strip().lower()
        if p and p not in out:
            out.append(p)
    if not out:
        out = ["yahoo", "stooq"]
    return out


def _fetch_fallback_history(sym: str, *, start: datetime, end: datetime) -> Tuple[Optional[pd.DataFrame], str]:
    if not _fallback_enabled():
        return None, ""
    providers = _fallback_providers()
    aliases = _symbol_aliases(sym)
    min_bars = _fallback_min_bars()

    for provider in providers:
        for alias in aliases:
            if provider == "yahoo":
                df = _download_yahoo_history(alias, start, end)
            elif provider == "stooq":
                df = _download_stooq_history(alias)
            else:
                df = None
            if df is None or df.empty:
                continue
            if len(df) < min_bars:
                continue
            _note_source_hit(provider)
            return df, provider
    return None, ""


def fetch_single_symbol(
    sym: str,
    days: int = 1260,
    force_fresh: bool = False,
    *,
    cache_only: bool = False,
    require_fresh: bool = False,
    inject_live: bool = False,
    prefetched_quote: Optional[Dict] = None,
    max_lag_days: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a single symbol.
    If force_fresh=True, it will ALWAYS ping the API for the latest data and merge it.
    If inject_live=True, it appends a synthetic bar using live quotes when today's bar is missing.
    """
    cache_dir = os.path.join("data", "cache")
    symbol = str(sym or "").strip().upper()
    if not symbol:
        return None

    required_start = _required_history_start(days)
    recent_primary_miss = _is_recent_missing_symbol(symbol) or _is_recent_missing_symbol(symbol, scope="schwab")
    recent_fallback_miss = _is_recent_missing_symbol(symbol, scope="fallback")
    if (
        not force_fresh
        and not cache_only
        and not os.path.exists(os.path.join(cache_dir, f"{symbol}.parquet"))
        and recent_primary_miss
        and (not _fallback_enabled() or recent_fallback_miss)
    ):
        return None

    # Legacy CSV cache compatibility.
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")
    if not force_fresh and os.path.exists(cache_path):
        try:
            if (time.time() - os.path.getmtime(cache_path)) < 43200:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                df = clean_dataframe(df)
                if cache_only or _has_required_history(df, required_start):
                    return df
        except Exception:
            pass

    # Prefer very fresh parquet cache for hot reruns.
    try:
        cache_path = os.path.join(cache_dir, f"{symbol}.parquet")
        if not force_fresh and os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) < 43200:
                df = pd.read_parquet(cache_path)
                df = clean_dataframe(df)
                if cache_only or _has_required_history(df, required_start):
                    return df
    except Exception:
        pass

    if cache_only:
        try:
            df = DataCache.get_cached_data(symbol, allow_stale=True, validate=False)
            df = clean_dataframe(df)
            if df is not None and not df.empty:
                first_date = df.index.min()
                if first_date > (required_start + timedelta(days=30)):
                    print(
                        f"⚠️ Cache shallow for {symbol} (Starts {first_date.date()}). "
                        "Using available data (Backtest Mode)."
                    )
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    return df[df.index >= first_date]
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                return df[df.index >= required_start]
        except Exception as e:
            print(f"❌ Cache read error ({symbol}): {e}")
            df = None

    if max_lag_days is None:
        if inject_live or force_fresh or require_fresh:
            max_lag_days = 0
        else:
            try:
                max_lag_days = int(os.getenv("DATA_MAX_LAG_DAYS", "4"))
            except Exception:
                max_lag_days = 4
    max_lag_days = max(0, int(max_lag_days))

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 100)

    # Keep stale cache available so we can incrementally refresh instead of discarding
    # a full 5-year history (which dramatically increases API load and rate-limits).
    df = DataCache.get_cached_data(symbol, allow_stale=True)
    df = clean_dataframe(df)

    start_naive = required_start

    fetch_start = start
    if df is not None and not df.empty:
        # If cached history doesn't go back far enough for the requested window,
        # pull the full range to backfill older data.
        if df.index.min() <= start_naive:
            fetch_start = df.index.max().replace(tzinfo=timezone.utc)
        else:
            fetch_start = start

    if not force_fresh and df is not None:
        last_date = df.index.max().date()
        today = end.date()
        if (today - last_date).days <= max_lag_days:
            if df.index.min() <= start_naive:
                out = df[df.index >= start_naive]
                return inject_live_quote(out, symbol, prefetched_quote=prefetched_quote) if inject_live else out

    if cache_only:
        if df is None:
            return None

    schwab_error: Optional[Exception] = None
    candles: Optional[List[Dict]] = None
    try:
        candles, _ = _fetch_schwab_candles_with_aliases(
            symbol,
            start_datetime=fetch_start,
            end_datetime=end,
        )
    except Exception as e:
        schwab_error = e

    if candles:
        df_new = _to_frame_from_candles(candles)
        if df_new is not None:
            if df is not None and not df.empty:
                df = pd.concat([df, df_new])
                df = df[~df.index.duplicated(keep="last")]
            else:
                df = df_new
            DataCache.save_to_cache(symbol, df)
            _clear_missing_symbol(symbol)
            _clear_missing_symbol(symbol, scope="schwab")
            _clear_missing_symbol(symbol, scope="fallback")
    elif not cache_only and (df is None or df.empty):
        _mark_missing_symbol(symbol)
        _mark_missing_symbol(symbol, scope="schwab")
        if schwab_error is not None:
            print(f"⚠️ API Fetch failed for {symbol}: {schwab_error}")

    # Backfill with fallback sources when Schwab is unavailable or insufficient.
    needs_backfill = (df is None or df.empty or not _has_required_history(df, start_naive))
    can_try_fallback = (
        not cache_only
        and _fallback_enabled()
        and (force_fresh or not _is_recent_missing_symbol(symbol, scope="fallback"))
    )
    if needs_backfill and can_try_fallback:
        fb_df, fb_source = _fetch_fallback_history(symbol, start=start, end=end)
        if fb_df is not None and not fb_df.empty:
            if df is not None and not df.empty:
                df = pd.concat([df, fb_df])
                df = df[~df.index.duplicated(keep="last")]
                df = clean_dataframe(df)
            else:
                df = fb_df
            if df is not None and not df.empty:
                DataCache.save_to_cache(symbol, df)
                _clear_missing_symbol(symbol)
                _clear_missing_symbol(symbol, scope="schwab")
                _clear_missing_symbol(symbol, scope="fallback")
                _note_source_hit(f"fallback_{fb_source}")
        else:
            _mark_missing_symbol(symbol, scope="fallback")

    if df is not None:
        if require_fresh:
            try:
                last_date = df.index.max().date()
                today = end.date()
                if (today - last_date).days > max_lag_days:
                    return None
            except Exception:
                return None
        df = df[df.index >= start_naive]
        return inject_live_quote(df, symbol, prefetched_quote=prefetched_quote) if inject_live else df
    if not cache_only:
        _mark_missing_symbol(symbol)
        _mark_missing_symbol(symbol, scope="schwab")
    return None


def fetch_data_pack(
    symbols: List[str],
    days: int = 1260,
    *,
    max_workers: Optional[int] = None,
    require_full_lookback: bool = False,
    force_fresh: bool = False,
    require_fresh: bool = False,
    inject_live: bool = False,
    backtest_mode: bool = False,
    max_lag_days: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """Bulk fetch for Backtester (Threaded for speed)."""
    data: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    incomplete: List[str] = []
    stale: List[str] = []

    end = datetime.now(timezone.utc)
    # Must match `fetch_single_symbol` windowing (days + 100 buffer).
    start_naive = (end - timedelta(days=days + 100)).replace(tzinfo=None)
    min_history_start = start_naive + timedelta(days=10)  # weekend/holiday tolerance
    today = end.date()

    if backtest_mode:
        force_fresh = False
        require_fresh = False
        inject_live = False
        max_lag_days = 99999

    if max_lag_days is None:
        try:
            max_lag_days = int(os.getenv("DATA_MAX_LAG_DAYS", "4"))
        except Exception:
            max_lag_days = 4
    max_lag_days = max(0, int(max_lag_days))

    live_map: Dict[str, Dict] = {}
    if inject_live:
        try:
            live_map = sd.get_quotes(symbols)
        except Exception as e:
            print(f"❌ CRITICAL: Batch Quote Fetch Failed! {e}")
            print("⚠️ Disabling Live Injection to prevent deadlock.")
            live_map = {}
            inject_live = False
        if inject_live and not live_map:
            print("⚠️ WARNING: Batch fetch returned 0 quotes. Disabling Live Injection to prevent deadlock.")
            inject_live = False
        print(f"🚀 Batch fetched {len(live_map)} live quotes.")

    def load(sym: str):
        quote = None
        if inject_live and live_map:
            quote = live_map.get(sym) or live_map.get(sym.upper())
        return sym, fetch_single_symbol(
            sym,
            days,
            force_fresh=force_fresh,
            cache_only=backtest_mode,
            require_fresh=require_fresh,
            inject_live=inject_live,
            prefetched_quote=quote,
            max_lag_days=max_lag_days,
        )

    if max_workers is None:
        raw_workers = str(os.getenv("DATA_FETCH_WORKERS", "") or "").strip()
        if raw_workers:
            try:
                max_workers = int(raw_workers)
            except Exception:
                max_workers = 10
        else:
            cpu = os.cpu_count() or 8
            if len(symbols) >= 2000:
                max_workers = min(24, max(8, cpu * 2))
            elif len(symbols) >= 800:
                max_workers = min(18, max(8, int(cpu * 1.5)))
            elif len(symbols) >= 250:
                max_workers = min(14, max(6, cpu))
            else:
                max_workers = min(10, max(4, cpu // 2))
    max_workers = max(1, min(int(max_workers), 32))

    sources_before = _source_hits_snapshot()
    count = 0
    total = len(symbols)
    print(f"📉 Starting History Download for {total} symbols...")
    try:
        heartbeat_sec = float(os.getenv("DATA_FETCH_HEARTBEAT_SEC", "8.0") or "8.0")
    except Exception:
        heartbeat_sec = 8.0
    heartbeat_sec = max(2.0, min(heartbeat_sec, 60.0))

    try:
        stall_timeout_sec = float(os.getenv("DATA_FETCH_STALL_TIMEOUT_SEC", "180.0") or "180.0")
    except Exception:
        stall_timeout_sec = 180.0
    stall_timeout_sec = max(20.0, min(stall_timeout_sec, 900.0))

    last_completion = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=max_workers)
    future_to_sym = {executor.submit(load, sym): sym for sym in symbols}
    pending = set(future_to_sym.keys())

    try:
        while pending:
            done, pending = wait(
                pending,
                timeout=heartbeat_sec,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                stalled_for = time.monotonic() - last_completion
                print(
                    f"   ⏳ Waiting on workers... {count}/{total} completed, "
                    f"{len(pending)} pending (idle {stalled_for:.0f}s)"
                )
                if stalled_for >= stall_timeout_sec:
                    print(
                        "⚠️ Data download stall timeout reached; returning partial coverage "
                        f"after {stall_timeout_sec:.0f}s without completions."
                    )
                    for fut in list(pending):
                        sym = future_to_sym.get(fut)
                        if sym:
                            missing.append(sym)
                        fut.cancel()
                    pending.clear()
                    break
                continue

            for fut in done:
                sym = future_to_sym.get(fut, "")
                try:
                    sym_res, df = fut.result()
                    if sym_res:
                        sym = sym_res
                except Exception:
                    missing.append(sym)
                    count += 1
                    continue

                count += 1
                last_completion = time.monotonic()
                if count % 50 == 0 or count == total:
                    print(f"   ⏳ Downloaded {count}/{total} symbols ({(count/total)*100:.1f}%)")

                if df is not None and not df.empty:
                    # Guardrail: don't silently accept "short" cached series when a longer lookback
                    # was requested (common when caches were built with fewer days).
                    try:
                        if isinstance(df.index, pd.DatetimeIndex) and df.index.min() > min_history_start:
                            incomplete.append(sym)
                            if require_full_lookback:
                                continue
                    except Exception:
                        pass

                    try:
                        if isinstance(df.index, pd.DatetimeIndex):
                            last_date = df.index.max().date()
                            if (today - last_date).days > max_lag_days:
                                stale.append(sym)
                                if require_fresh:
                                    continue
                    except Exception:
                        stale.append(sym)
                        if require_fresh:
                            continue

                    data[sym] = df
                else:
                    missing.append(sym)
    finally:
        # Never block forever on hung workers when the app is interactive.
        executor.shutdown(wait=False, cancel_futures=True)

    source_delta = _source_hits_delta(sources_before, _source_hits_snapshot())
    if source_delta:
        parts = [f"{k}={v}" for k, v in sorted(source_delta.items(), key=lambda kv: kv[0])]
        print(f"🧭 Data sources used: {', '.join(parts)}")

    if missing or incomplete or stale:
        msg = []
        if missing:
            msg.append(f"missing={len(missing)}")
        if incomplete:
            msg.append(f"incomplete_history={len(incomplete)}")
        if stale:
            msg.append(f"stale={len(stale)}")
        print(f"⚠️ Data quality issues: {', '.join(msg)} (requested={len(symbols)}, loaded={len(data)}).")
        if os.getenv("DATA_EXPORT_MISSING", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}:
            os.makedirs("exports", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join("exports", f"{ts}_data_quality_issues.txt")
            with open(path, "w") as f:
                if missing:
                    f.write("# Missing symbols (no usable data)\n")
                    f.write("\n".join(missing) + "\n\n")
                if incomplete:
                    f.write("# Incomplete lookback (shorter history than requested)\n")
                    f.write("\n".join(incomplete) + "\n")
                    f.write("\n")
                if stale:
                    f.write(f"# Stale data (last bar older than {max_lag_days} days)\n")
                    f.write("\n".join(stale) + "\n")
            print(f"   📝 Saved details to {path}")

    return data
