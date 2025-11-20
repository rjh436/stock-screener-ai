import os
import pandas as pd
from datetime import datetime, date, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Cache is considered stale if older than this many days
CACHE_STALENESS_DAYS = 7

class DataCache:
    @staticmethod
    def get_cached_data(symbol: str, validate: bool = True) -> pd.DataFrame:
        """Load cached data for a symbol if it exists.
        
        Args:
            symbol: Stock symbol
            validate: If True, check for data quality issues
            
        Returns:
            DataFrame or None if cache is invalid/missing
        """
        path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                
                # Check staleness
                if DataCache.is_cache_stale(symbol):
                    print(f"{symbol} cache is stale (>7 days old), will refresh")
                    return None
                
                # Validate data integrity
                if validate and not DataCache.validate_data(df, symbol):
                    print(f"{symbol} cache failed validation, will refresh")
                    return None
                    
                return df
            except Exception as e:
                print(f"Error reading cache for {symbol}: {e}")
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
            print(f"Error saving cache for {symbol}: {e}")

    @staticmethod
    def get_last_date(symbol: str) -> date:
        """Get the last available date in the cache."""
        df = DataCache.get_cached_data(symbol, validate=False)
        if df is not None and not df.empty:
            return df.index.max().date()
        return None
    
    @staticmethod
    def is_cache_stale(symbol: str) -> bool:
        """Check if cache file is older than CACHE_STALENESS_DAYS."""
        path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
        if not os.path.exists(path):
            return True
            
        file_mod_time = os.path.getmtime(path)
        file_mod_date = datetime.fromtimestamp(file_mod_time)
        age_days = (datetime.now() - file_mod_date).days
        
        return age_days > CACHE_STALENESS_DAYS
    
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
                print(f"{symbol}: Found NaN values in {col}")
                return False
        
        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                print(f"{symbol}: Found zero/negative values in {col}")
                return False
        
        # Check for massive date gaps (>30 days, indicating missing data)
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            date_diffs = df.index.to_series().diff()
            max_gap = date_diffs.max()
            if max_gap > timedelta(days=30):
                print(f"{symbol}: Found suspicious date gap of {max_gap.days} days")
                return False
        
        return True
    
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
