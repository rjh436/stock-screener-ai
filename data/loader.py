import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from .schwab_client import sd
from .cache_manager import DataCache


def clean_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Standardize DataFrame: Lowercase columns, timezone-naive index, numeric types."""
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


def fetch_data_pack(symbols: List[str], days: int = 1260) -> Dict[str, pd.DataFrame]:
    """Fetch and clean data for a list of symbols. Uses cache first."""
    print(f"Loader: Fetching {len(symbols)} symbols ({days} days)...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 100)
    data: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            df = DataCache.get_cached_data(sym)
            if df is not None:
                df = clean_dataframe(df)
                start_naive = start.replace(tzinfo=None)
                if df is not None and df.index.min() <= start_naive:
                    data[sym] = df[df.index >= start_naive]
                    continue

            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df_new = pd.DataFrame(candles)
                if "datetime" in df_new.columns:
                    df_new["datetime"] = pd.to_datetime(df_new["datetime"], unit="ms", utc=True)
                    df_new = df_new.set_index("datetime")
                DataCache.save_to_cache(sym, df_new)
                df_clean = clean_dataframe(df_new)
                if df_clean is not None:
                    data[sym] = df_clean
        except Exception:
            continue

    return data
