import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta, timezone
import pytz
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
from .schwab_client import sd
from .cache_manager import DataCache
from concurrent.futures import ThreadPoolExecutor

_MISS_CACHE_DIR = os.path.join("data", "cache", "_miss")


def _missing_symbol_ttl_seconds() -> int:
    try:
        ttl_hours = float(os.getenv("DATA_MISSING_RETRY_HOURS", "24") or "24")
    except Exception:
        ttl_hours = 24.0
    ttl_hours = max(1.0, min(ttl_hours, 24.0 * 14.0))
    return int(ttl_hours * 3600)


def _missing_symbol_path(sym: str) -> str:
    key = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(sym or "").upper())
    if not key:
        key = "UNKNOWN"
    return os.path.join(_MISS_CACHE_DIR, f"{key}.miss")


def _is_recent_missing_symbol(sym: str) -> bool:
    path = _missing_symbol_path(sym)
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


def _mark_missing_symbol(sym: str) -> None:
    try:
        os.makedirs(_MISS_CACHE_DIR, exist_ok=True)
        path = _missing_symbol_path(sym)
        with open(path, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass


def _clear_missing_symbol(sym: str) -> None:
    path = _missing_symbol_path(sym)
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
    symbol = sym
    required_start = _required_history_start(days)
    if (
        not force_fresh
        and not cache_only
        and not os.path.exists(os.path.join(cache_dir, f"{symbol}.parquet"))
        and _is_recent_missing_symbol(symbol)
    ):
        return None
    # --- TURBO CACHE FIX: Trust fresh files for IPOs ---
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")
    if not force_fresh and os.path.exists(cache_path):
        try:
            # If file is < 12 hours old, use it regardless of start date
            if (time.time() - os.path.getmtime(cache_path)) < 43200:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                df = clean_dataframe(df)
                if cache_only or _has_required_history(df, required_start):
                    return df
        except: pass
    # ---------------------------------------------------
    # --- TURBO CACHE: Trust fresh files (12 hours) ---
    # Prevents infinite redownload of IPOs/Short-history stocks
    try:
        cache_path = os.path.join("data", "cache", f"{sym}.parquet")
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) < 43200: # 12 hours
                # If we just downloaded it, use it. Don't check depth.
                df = pd.read_parquet(cache_path)
                df = clean_dataframe(df)
                if cache_only or _has_required_history(df, required_start):
                    return df
    except Exception:
        pass
    # -------------------------------------------------
    # --- SMART TURBO MODE (Auto-Backfill) ---
    if cache_only:
        try:
            # 1. Attempt load
            df = DataCache.get_cached_data(sym, allow_stale=True, validate=False)
            df = clean_dataframe(df)

            if df is not None and not df.empty:
                # 3. DEPTH CHECK: Does cache go back far enough?
                # We allow a 30-day buffer. If cache starts AFTER the required date, it's a miss.
                first_date = df.index.min()
                if first_date > (required_start + timedelta(days=30)):
                    if cache_only:
                        print(f"⚠️ Cache shallow for {sym} (Starts {first_date.date()}). Using available data (Backtest Mode).")
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        df = df[df.index >= first_date] 
                        return df
                    
                    print(
                        f"⚠️ Cache shallow for {sym} (Starts {first_date.date()}, "
                        f"Need {required_start.date()}). Auto-downloading..."
                    )
                    # IMPORTANT: Set df to None so we fall through to the download logic below
                    df = None
                else:
                    # Cache is good! Slice and return.
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df[df.index >= required_start]
                    return df

            # If we get here, df is None (either missing or rejected for depth).
            # We explicitly PASS to allow the function to continue to the 'sd.price_daily' download block.

        except Exception as e:
            print(f"❌ Cache read error ({sym}): {e}")
            df = None
            # Fall through to download

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
    df = DataCache.get_cached_data(sym, allow_stale=True)
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
                return inject_live_quote(df[df.index >= start_naive], sym, prefetched_quote=prefetched_quote) if inject_live else df[df.index >= start_naive]

    if cache_only:
        if df is None:
            # print(f"⚠️ {sym} not found in cache. Skipping (Backtest Mode).")
            return None
            
    try:
        candles = sd.price_daily(sym, start_datetime=fetch_start, end_datetime=end)
        if candles:
            df_new = pd.DataFrame(candles)
            if "datetime" in df_new.columns:
                df_new["datetime"] = pd.to_datetime(df_new["datetime"], unit="ms", utc=True)
                df_new = df_new.set_index("datetime")

            if df is not None:
                df_new = clean_dataframe(df_new)
                if df_new is not None:
                    df = pd.concat([df, df_new])
                    df = df[~df.index.duplicated(keep="last")]
            else:
                df = clean_dataframe(df_new)

            if df is not None:
                DataCache.save_to_cache(sym, df)
                _clear_missing_symbol(sym)
        elif not cache_only and (df is None or df.empty):
            _mark_missing_symbol(sym)

    except Exception as e:
        print(f"⚠️ API Fetch failed for {sym}: {e}")
        if not cache_only and (df is None or df.empty):
            _mark_missing_symbol(sym)
        if require_fresh:
            return None
        # Preserve cached data (even if slightly stale) instead of dropping the symbol.
        if df is None:
            return None
        df = df[df.index >= start_naive]
        return inject_live_quote(df, sym, prefetched_quote=prefetched_quote) if inject_live else df

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
        return inject_live_quote(df, sym, prefetched_quote=prefetched_quote) if inject_live else df
    if not cache_only:
        _mark_missing_symbol(sym)
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

    count = 0
    total = len(symbols)
    print(f"📉 Starting History Download for {total} symbols...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym, df in executor.map(load, symbols):
            count += 1
            if count % 50 == 0:
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
