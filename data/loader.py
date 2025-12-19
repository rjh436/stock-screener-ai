import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
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


def fetch_single_symbol(
    sym: str,
    days: int = 1260,
    force_fresh: bool = False,
    *,
    require_fresh: bool = False,
    max_lag_days: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a single symbol.
    If force_fresh=True, it will ALWAYS ping the API for the latest data and merge it.
    """
    if max_lag_days is None:
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
                return df[df.index >= start_naive]

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

    except Exception:
        if require_fresh:
            return None
        # Preserve cached data (even if slightly stale) instead of dropping the symbol.
        return df[df.index >= start_naive] if df is not None else None

    if df is not None:
        if require_fresh:
            try:
                last_date = df.index.max().date()
                today = end.date()
                if (today - last_date).days > max_lag_days:
                    return None
            except Exception:
                return None
        return df[df.index >= start_naive]
    return None


def fetch_data_pack(
    symbols: List[str],
    days: int = 1260,
    *,
    max_workers: Optional[int] = None,
    require_full_lookback: bool = False,
    force_fresh: bool = False,
    require_fresh: bool = False,
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

    if max_lag_days is None:
        try:
            max_lag_days = int(os.getenv("DATA_MAX_LAG_DAYS", "4"))
        except Exception:
            max_lag_days = 4
    max_lag_days = max(0, int(max_lag_days))

    def load(sym: str):
        return sym, fetch_single_symbol(
            sym,
            days,
            force_fresh=force_fresh,
            require_fresh=require_fresh,
            max_lag_days=max_lag_days,
        )

    if max_workers is None:
        try:
            max_workers = int(os.getenv("DATA_FETCH_WORKERS", "10"))
        except Exception:
            max_workers = 10
    max_workers = max(1, min(int(max_workers), 32))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sym, df in executor.map(load, symbols):
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
