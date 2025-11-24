
import sys
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime, timezone

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.cache_manager import DataCache
from execution.engine import _compute_indicators, run_backtest
from strategies.generic import GenericStrategy

def debug_with_real_data(symbol="AAPL"):
    print(f"🔍 Debugging with REAL DATA for {symbol}...")
    
    # 1. Load Data
    print("1️⃣  Loading Symbol Data...")
    df = DataCache.get_cached_data(symbol, validate=False)
    if df is None:
        print(f"❌ No data found for {symbol}")
        return
    
    # 2. Load VIX Data
    print("2️⃣  Loading VIX Data...")
    vix_df = DataCache.get_cached_data("$VIX", validate=False)
    if vix_df is None:
        print("❌ No VIX data found!")
        vix_data = None
    else:
        print(f"✅ VIX data loaded: {len(vix_df)} rows")
        vix_data = {"close": vix_df["close"]}
        global_data = {"VIX": vix_df}

    # 3. Load Strategies
    print("\n3️⃣  Loading Strategies...")
    try:
        with open("config/generated_strategies.json", "r") as f:
            strategies = json.load(f)
        print(f"✅ Loaded {len(strategies)} strategies")
    except Exception as e:
        print(f"❌ Failed to load strategies: {e}")
        return

    # 4. Run Backtest
    print("\n4️⃣  Running Backtest...")
    data_map = {symbol: df}
    
    # 4. Run Backtest & Live Scan Check
    print("\n4️⃣  Running Checks...")
    data_map = {symbol: df}
    
    for strat_config in strategies:
        print(f"\n👉 Strategy: {strat_config['name']}")
        strat = GenericStrategy(strat_config)
        
        # Check Live Scan Signal
        last_idx = len(df) - 1
        try:
            signal = strat.entry(df, last_idx)
            if signal:
                print(f"   🚀 LIVE SIGNAL FOUND for {symbol}!")
                print(f"      Stop: {signal['stop_price']}")
            else:
                print(f"   ❌ No Live Signal for {symbol}")
        except Exception as e:
            print(f"   ⚠️ Live Scan Error: {e}")

        try:
            # Simulate 5 Year Backtest
            start_date = (datetime.now(timezone.utc) - pd.Timedelta(days=1260)).date()
            # print(f"   📅 Start Date: {start_date}")
            
            result = run_backtest(
                strat, 
                data_map, 
                symbol_universe=[symbol], 
                start_cash=100000.0, 
                start_date=start_date,
                global_data=global_data
            )
            print(f"   Backtest Trades: {result['total_trades']}")
            
        except Exception as e:
            print(f"   ❌ Backtest failed: {e}")

if __name__ == "__main__":
    symbols = ["AAPL", "SPY", "NVDA", "TSLA", "AMD"]
    for sym in symbols:
        debug_with_real_data(sym)
        print("-" * 50)
