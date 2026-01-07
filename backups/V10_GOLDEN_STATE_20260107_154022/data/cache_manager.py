import os
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from zoneinfo import ZoneInfo
    _ET_ZONE = ZoneInfo("America/New_York")
except Exception:
    _ET_ZONE = None

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache is considered stale if older than this many days
CACHE_STALENESS_DAYS = 7

def _cache_verbose() -> bool:
    return os.getenv("DATA_CACHE_VERBOSE", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}


def _cache_log(msg: str) -> None:
    if _cache_verbose():
        print(msg)

def _now_et() -> datetime:
    if _ET_ZONE is None:
        return datetime.now()
    return datetime.now(_ET_ZONE)


def _is_market_hours(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = _now_et()
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return start <= now <= end


class DataCache:
    @staticmethod
    def get_cached_data(
        symbol: str,
        validate: bool = True,
        *,
        allow_stale: bool = False,
        max_content_lag_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load cached data for a symbol if it exists.
        
        Args:
            symbol: Stock symbol
            validate: If True, check for data quality issues
            allow_stale: If True, return cached data even if cache file is older than
                CACHE_STALENESS_DAYS. This enables incremental refresh without discarding
                otherwise good historical data.
            max_content_lag_days: Overrides cache staleness window based on the last bar date.
            
        Returns:
            DataFrame or None if cache is invalid/missing
        """
        path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                
                # Check staleness
                if DataCache.is_cache_stale(symbol, max_content_lag_days=max_content_lag_days, df=df) and not allow_stale:
                    _cache_log(f"{symbol} cache is stale, will refresh")
                    return None
                
                # Validate data integrity
                if validate and not DataCache.validate_data(df, symbol):
                    _cache_log(f"{symbol} cache failed validation, will refresh")
                    return None
                    
                return df
            except Exception as e:
                _cache_log(f"Error reading cache for {symbol}: {e}")
        return None

    @staticmethod
    def save_to_cache(symbol: str, df: pd.DataFrame):
        """Save DataFrame to cache."""
        if df is None or df.empty:
            return
        path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        # Ensure index is datetime and sorted
        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.set_index("date")
        
        df = df.sort_index()
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        try:
            df.to_parquet(path)
        except Exception as e:
            _cache_log(f"Error saving cache for {symbol}: {e}")

    @staticmethod
    def get_last_date(symbol: str) -> date:
        """Get the last available date in the cache."""
        df = DataCache.get_cached_data(symbol, validate=False, allow_stale=True)
        if df is not None and not df.empty:
            return df.index.max().date()
        return None
    
    @staticmethod
    def _extract_last_bar_date(df: pd.DataFrame) -> Optional[date]:
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
        else:
            idx = None
            for col in ("date", "datetime", "timestamp", "__index_level_0__"):
                if col in df.columns:
                    idx = pd.to_datetime(df[col], errors="coerce")
                    break
            if idx is None:
                try:
                    idx = pd.to_datetime(df.index, errors="coerce")
                except Exception:
                    idx = None
        if idx is None:
            return None
        try:
            idx = pd.DatetimeIndex(idx).tz_localize(None)
        except Exception:
            pass
        try:
            last_ts = pd.DatetimeIndex(idx).max()
        except Exception:
            return None
        if pd.isna(last_ts):
            return None
        return last_ts.date()

    @staticmethod
    def is_cache_stale(
        symbol: str,
        *,
        max_content_lag_days: Optional[int] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> bool:
        """Check if cached content is stale based on the last bar date."""
        path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        if not os.path.exists(path):
            return True

        if df is None:
            try:
                df = pd.read_parquet(path)
            except Exception as e:
                _cache_log(f"{symbol} cache read failed: {e}")
                return True

        last_date = DataCache._extract_last_bar_date(df)
        if last_date is None:
            _cache_log(f"{symbol} cache missing last bar date")
            return True

        now = _now_et()
        today = now.date()
        if _is_market_hours(now) and last_date != today:
            return True

        if max_content_lag_days is None:
            max_content_lag_days = CACHE_STALENESS_DAYS
        try:
            lag_days = (today - last_date).days
        except Exception:
            return True
        return lag_days > max_content_lag_days
    
    @staticmethod
    def validate_data(df: pd.DataFrame, symbol: str) -> bool:
        """Validate data integrity.
        
        Checks for:
        - NaN values in critical columns
        - Zero/negative prices
        - Massive gaps in dates
        
        Returns:
            True if data is valid, False otherwise
        """
        if df is None or df.empty:
            return False
            
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for NaN values
        for col in critical_cols:
            if col in df.columns and df[col].isna().any():
                _cache_log(f"{symbol}: Found NaN values in {col}")
                return False
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                _cache_log(f"{symbol}: Found zero/negative values in {col}")
                return False
        
        # Check for massive date gaps (>30 days, indicating missing data)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            date_diffs = df.index.to_series().diff()
            max_gap = date_diffs.max()
            if max_gap > timedelta(days=30):
                _cache_log(f"{symbol}: Found suspicious date gap of {max_gap.days} days")
                return False
        
        return True

    @staticmethod
    def validate_manifest(symbols, min_bars: int = 200) -> list:
        """
        Fast manifest validation using parquet metadata only.

        Returns a list of symbols that are missing, corrupt, or have insufficient history.
        """
        if not symbols:
            return []

        bad = []
        for sym in symbols:
            if not sym:
                continue
            path = os.path.join(CACHE_DIR, f"{sym}.parquet")
            if not os.path.exists(path):
                bad.append(sym)
                continue
            try:
                meta = pq.read_metadata(path)
                if meta is None or meta.num_rows < int(min_bars):
                    bad.append(sym)
            except Exception:
                bad.append(sym)
        return bad
    
    @staticmethod
    def clear_cache(symbol: str = None):
        """Clear cache for a specific symbol or all symbols.
        
        Args:
            symbol: If provided, clear only this symbol. If None, clear all.
        """
        if symbol:
            path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
            if os.path.exists(path):
                os.remove(path)
                print(f"Cleared cache for {symbol}")
        else:
            # Clear all cache files
            import glob
            cache_files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
            for f in cache_files:
                os.remove(f)
            print(f"Cleared {len(cache_files)} cache files")
