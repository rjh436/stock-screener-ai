import sys
import os
import pandas as pd
import numpy as np
from strategies.ai_v3 import StrategyAIV3

def run_debug():
    print("--- STARTING DEBUG BACKTEST FOR AI V3 ---")
    
    # 1. Load Strategy
    strat = StrategyAIV3({})
    print(f"Strategy Loaded: {strat.name}")
    
    # 2. Create Mock Data
    # Scenario: Long term uptrend (Close > SMA200), but deep pullback (EMA100 > Highest55)
    periods = 300
    dates = pd.date_range(start="2023-01-01", periods=periods)
    
    data = {
        "date": dates,
        "open": [100.0] * periods,
        "high": [105.0] * periods,
        "low": [90.0] * periods,
        "close": [95.0] * periods, # Current Price
        "volume": [1000000] * periods,
        
        # Indicators
        "sma200": [90.0] * periods,      # Trend Filter: Close (95) > SMA200 (90) -> PASS
        "ema100": [110.0] * periods,     # Rule: EMA100 (110) > Highest55 (105) -> PASS
        "highest55": [105.0] * periods,  # 55-day High
        "plus_di": [20.0] * periods,     # Rule: Plus_DI (20) < Highest55 (105) -> PASS
        "rsi14": [50.0] * periods,       # Rule: RSI14 (50) > CCI (40) -> PASS
        "cci": [40.0] * periods,
        "atr": [2.0] * periods,
        "ema20": [98.0] * periods,
        "ema50": [100.0] * periods
    }
    
    df = pd.DataFrame(data)
    
    # Set specific values at index 250 to ensure exact match
    i = 250
    df.loc[i, "close"] = 95.0
    df.loc[i, "sma200"] = 90.0
    df.loc[i, "ema100"] = 110.0
    df.loc[i, "highest55"] = 105.0
    df.loc[i, "plus_di"] = 20.0
    df.loc[i, "rsi14"] = 50.0
    df.loc[i, "cci"] = 40.0
    
    print(f"\n--- CHECKING ENTRY AT INDEX {i} ---")
    print(f"Close: {df.loc[i, 'close']}")
    print(f"SMA200: {df.loc[i, 'sma200']} (Check: Close > SMA200)")
    print(f"EMA100: {df.loc[i, 'ema100']}")
    print(f"Highest55: {df.loc[i, 'highest55']} (Check: EMA100 > Highest55)")
    print(f"Plus_DI: {df.loc[i, 'plus_di']} (Check: Plus_DI < Highest55)")
    print(f"RSI14: {df.loc[i, 'rsi14']}")
    print(f"CCI: {df.loc[i, 'cci']} (Check: RSI14 > CCI)")
    
    entry_signal = strat.entry(df, i)
    
    if entry_signal:
        print(f"\n✅ ENTRY TRIGGERED: {entry_signal}")
    else:
        print("\n❌ NO ENTRY TRIGGERED")

if __name__ == "__main__":
    run_debug()
