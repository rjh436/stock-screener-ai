
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

def debug_backtest(symbol="AAPL"):
    print(f"🔍 Debugging Backtest for {symbol}...")
    
    # 1. Load Data
    print("1️⃣  Loading Data...")
    df = DataCache.get_cached_data(symbol, validate=False)
    if df is None:
        print(f"❌ No data found for {symbol}")
        return
    
    # Mock Global Data (VIX)
    print("   Creating mock VIX data...")
    vix_data = pd.DataFrame({
        "close": [20.0] * len(df)
    }, index=df.index)
    global_data = {"VIX": vix_data}

    # 2. Load Strategies
    print("\n2️⃣  Loading Strategies...")
    try:
        with open("config/generated_strategies.json", "r") as f:
            strategies = json.load(f)
        print(f"✅ Loaded {len(strategies)} strategies")
    except Exception as e:
        print(f"❌ Failed to load strategies: {e}")
        return

    # 3. Run Backtest
    print("\n3️⃣  Running Backtest...")
    data_map = {symbol: df}
    
    for strat_config in strategies:
        print(f"\n👉 Strategy: {strat_config['name']}")
        strat = GenericStrategy(strat_config)
        
        try:
            result = run_backtest(
                strat, 
                data_map, 
                symbol_universe=[symbol], 
                start_cash=100000.0, 
                global_data=global_data
            )
            print("   Result:")
            print(json.dumps(result, indent=2, default=str))
            
            if result['total_trades'] == 0:
                print("   ⚠️  WARNING: 0 Trades found!")
        except Exception as e:
            print(f"   ❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    target_symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    debug_backtest(target_symbol)
