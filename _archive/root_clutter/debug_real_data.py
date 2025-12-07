import pandas as pd
import datetime
from execution.engine import run_backtest
from strategies.ai_optimized import StrategyAIOptimized
from data.schwab_client import sd

def debug_real_data():
    print("--- DEBUGGING REAL DATA (AAPL) ---")
    
    # 1. Fetch Data
    symbol = "AAPL"
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime.now()
    
    print(f"Fetching data for {symbol}...")
    try:
        bars = sd.price_daily(symbol, start_datetime=start_date, end_datetime=end_date)
        if isinstance(bars, dict):
            bars = bars.get("candles", [])
            
        df = pd.DataFrame(bars)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
            df = df.set_index("datetime")
        
        # Add openinterest for engine compatibility if needed (engine adds it, but let's be safe)
        df["openinterest"] = 0.0
        
        print(f"Loaded {len(df)} bars.")
        
    except Exception as e:
        print(f"Data fetch failed: {e}")
        return

    # 2. Run Strategy
    strat = StrategyAIOptimized()
    print(f"Running {strat.name}...")
    
    # We need to manually compute indicators because engine.run_backtest expects 'enriched' data 
    # OR the engine computes them? 
    # Checking engine.py: run_backtest calls _compute_indicators internally.
    
    res = run_backtest(strat, {symbol: df})
    
    print("\n--- RESULTS ---")
    print(f"Total Trades: {res['total_trades']}")
    print(f"Hit Rate: {res['hit_rate']}%")
    print(f"PnL: {res['pnl']}")
    
    if res['total_trades'] == 0:
        print("\n--- DIAGNOSTIC: CHECKING INDICATORS ---")
        # Let's look at the dataframe inside the engine... 
        # We can't easily. But we can compute them here and check conditions.
        from execution.engine import _compute_indicators
        df_ind = _compute_indicators(df)
        
        print(f"Last 5 RSI2 values:\n{df_ind['rsi2'].tail(5)}")
        print(f"Last 5 SMA200 values:\n{df_ind['sma200'].tail(5)}")
        
        # Check if ANY bar met condition
        # Cond: Close > SMA200 AND (RSI2 < 10 OR Close < BB_Lower)
        
        cond_trend = df_ind["close"] > df_ind["sma200"]
        cond_rsi = df_ind["rsi2"] < 10
        cond_bb = df_ind["close"] < df_ind["bb_lower"]
        
        valid_entries = df_ind[cond_trend & (cond_rsi | cond_bb)]
        
        print(f"\nManually found {len(valid_entries)} potential entry setups.")
        if not valid_entries.empty:
            print("First 3 setups:")
            print(valid_entries[["close", "sma200", "rsi2", "bb_lower"]].head(3))
        else:
            print("No setups found manually either.")

if __name__ == "__main__":
    debug_real_data()
