import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
import pytz
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
from .schwab_client import sd
from .cache_manager import DataCache
from concurrent.futures import ThreadPoolExecutor


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
    # --- TURBO CACHE: Trust fresh files (12 hours) ---
    # Prevents infinite redownload of IPOs/Short-history stocks
    try:
        cache_path = os.path.join("data", "cache", f"{sym}.parquet")
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (time.time() - mtime) < 43200: # 12 hours
                # If we just downloaded it, use it. Don't check depth.
                df = pd.read_parquet(cache_path)
                return clean_dataframe(df)
    except Exception:
        pass
    # -------------------------------------------------
    # --- SMART TURBO MODE (Auto-Backfill) ---
    if cache_only:
        try:
            # 1. Attempt load
            df = DataCache.get_cached_data(sym, allow_stale=True, validate=False)
            df = clean_dataframe(df)

            # 2. Calculate Required Start Date (Trading Days -> Calendar Days)
            calendar_days = int(days * 1.6)
            start_cutoff = datetime.now(timezone.utc) - timedelta(days=calendar_days)
            start_naive = start_cutoff.replace(tzinfo=None)

            if df is not None and not df.empty:
                # 3. DEPTH CHECK: Does cache go back far enough?
                # We allow a 30-day buffer. If cache starts AFTER the required date, it's a miss.
                first_date = df.index.min()
                if first_date > (start_naive + timedelta(days=30)):
                    print(
                        f"⚠️ Cache shallow for {sym} (Starts {first_date.date()}, "
                        f"Need {start_naive.date()}). Auto-downloading..."
                    )
                    # IMPORTANT: Set df to None so we fall through to the download logic below
                    df = None
                else:
                    # Cache is good! Slice and return.
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df[df.index >= start_naive]
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

    start_naive = start.replace(tzinfo=None)

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

    except Exception as e:
        print(f"⚠️ API Fetch failed for {sym}: {e}")
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
        try:
            max_workers = int(os.getenv("DATA_FETCH_WORKERS", "10"))
        except Exception:
            max_workers = 10
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
