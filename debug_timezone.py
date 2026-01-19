
import pandas as pd
from data.loader import fetch_data_pack

def debug_timezone():
    print("Fetching SPY and AAPL to check timezone alignment...")
    
    # Fetch SPY (Global)
    g_data = fetch_data_pack(["SPY"], days=200, force_fresh=True)
    spy_df = g_data.get("SPY")
    
    # Fetch AAPL (Symbol)
    data = fetch_data_pack(["AAPL"], days=200, force_fresh=True)
    aapl_df = data.get("AAPL")
    
    if spy_df is None or aapl_df is None:
        print("Failed to fetch data.")
        return

    print("\n--- SPY INDEX INFO ---")
    print(spy_df.index.dtype)
    print(spy_df.index[0])
    print(f"Timezone: {getattr(spy_df.index, 'tz', 'None')}")
    
    print("\n--- AAPL INDEX INFO ---")
    print(aapl_df.index.dtype)
    print(aapl_df.index[0])
    print(f"Timezone: {getattr(aapl_df.index, 'tz', 'None')}")
    
    # Simulate the Reindex Operation in engine.py
    print("\n--- SIMULATING REINDEX ---")
    
    # Simulate engine.py cleaning of AAPL
    if aapl_df.index.tz is not None:
        print("Stripping TZ from AAPL (Engine Logic)...")
        aapl_df.index = aapl_df.index.tz_localize(None)
        
    print(f"AAPL TZ after strip: {getattr(aapl_df.index, 'tz', 'None')}")
    
    # Now try to reindex SPY to AAPL
    try:
        spy_aligned = spy_df["Close"].reindex(aapl_df.index)
        print("\nReindex Result Head:")
        print(spy_aligned.head())
        
        nan_count = spy_aligned.isna().sum()
        print(f"\nNaN Count: {nan_count} / {len(spy_aligned)}")
        
        if nan_count == len(spy_aligned):
            print("CRITICAL FAILURE: Reindex resulted in ALL NaNs! Timezone mismatch confirmed.")
        else:
            print("Reindex seems okay.")
            
    except Exception as e:
        print(f"Reindex CRASHED: {e}")

if __name__ == "__main__":
    debug_timezone()
