import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from .schwab_client import sd
from .cache_manager import DataCache


def clean_dataframe(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Standardize DataFrame: Lowercase columns, Timezone-Naive Index, Numeric Types."""
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

        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])
        df = df[df["high"] > df["low"]]

        if len(df) < 20:
            return None
        return df
    except Exception:
        return None


def fetch_single_symbol(sym: str, days: int = 1260) -> Optional[pd.DataFrame]:
    """Fetch/Cache a single symbol (UI Helper)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 100)

    df = DataCache.get_cached_data(sym)
    if df is not None:
        df = clean_dataframe(df)
        start_naive = start.replace(tzinfo=None)
        if df is not None and df.index.min() <= start_naive:
            return df[df.index >= start_naive]

    try:
        candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
        if candles:
            df_new = pd.DataFrame(candles)
            if "datetime" in df_new.columns:
                df_new["datetime"] = pd.to_datetime(df_new["datetime"], unit="ms", utc=True)
                df_new = df_new.set_index("datetime")
            DataCache.save_to_cache(sym, df_new)
            return clean_dataframe(df_new)
    except Exception:
        pass
    return None


def fetch_data_pack(symbols: List[str], days: int = 1260) -> Dict[str, pd.DataFrame]:
    """Bulk fetch for Backtests/Optimizer."""
    print(f"Loader: Fetching {len(symbols)} symbols...")
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = fetch_single_symbol(sym, days)
        if df is not None:
            data[sym] = df
    return data
