import sys
import os
import datetime
import pandas as pd
import json
import logging

# Ensure we can import from root
sys.path.append(os.getcwd())

from strategies.strategy_loader import load_strategies
from execution.engine import run_backtest
from data.loader import fetch_data_pack
from data.indices import get_index_symbols

# Set logging to minimal
logging.basicConfig(level=logging.ERROR)

def debug_2008():
    # Load Universe - Small subset for speed
    print("Fetching symbols (limited subset)...")
    # Manually defined small universe for speed
    universe_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "JPM", "XOM", "PG", "JNJ", "HD"]
 

    print("Loading data (this may take a moment)...")
    # Load 20 years of data to cover 2008
    days_needed = 365 * 20 
    
    global_data = fetch_data_pack(["SPY", "VIX"], days=days_needed)
    data = fetch_data_pack(universe_symbols, days=days_needed)
    
    if not data:
        print("No stock data loaded.")
        return
        
    global_context = {
        "SPY": global_data.get("SPY"),
        "VIX": global_data.get("VIX") or global_data.get("$VIX")
    }
    
    if global_context["SPY"] is None or global_context["SPY"].empty:
        print("CRITICAL: SPY data missing or empty!")
        return
    else:
        print(f"SPY Data Range: {global_context['SPY'].index.min()} to {global_context['SPY'].index.max()}")

    # Load Strategies
    print("Loading strategies...")
    try:
        with open("config/generated_strategies.json", "r") as f:
            configs = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    strategies = load_strategies(configs)
    
    # Run Backtest for 2008 specifically
    # Start date needs to be slightly before 2008 to allow for indicator warmup if `run_backtest` cuts strictly.
    # But `run_backtest` usually runs on all available data if date not specified, or filters.
    # We will pass start_date="2007-06-01" to ensure indicators are ready for Jan 1 2008.
    
    start_date = "2007-06-01"
    
    print("Running Backtest (VALID SPY DATA)...")
    results = run_backtest(
        strategies, 
        data, 
        start_date=start_date, 
        global_data=global_context, # Use Real SPY
        start_cash=100000.0
    )
    
    if not isinstance(results, list):
        results = [results]
        
    print("\n" + "="*50)
    print("2008 BACKTEST RESULTS")
    print("="*50)

    for res in results:
        strat_name = res.get('strategy', 'Unknown')
        curve = res.get('equity_curve', [])
        trades = res.get('trades_list', [])
        
        if not curve:
            print(f"Strategy: {strat_name} - NO EQUITY CURVE")
            continue
            
        # Create DF
        df = pd.DataFrame(curve)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Slice 2008
        df_2008 = df['2008-01-01':'2008-12-31']
        
        if df_2008.empty:
            print(f"Strategy: {strat_name} - NO DATA FOR 2008")
            continue
            
        start_val = df_2008.iloc[0]['Equity']
        end_val = df_2008.iloc[-1]['Equity']
        min_val = df_2008['Equity'].min()
        max_val = df_2008['Equity'].max() # Approx peak
        
        # Calculate Max Drawdown WITHIN 2008
        # (Simplified: min / peak_before_min)
        # More accurately: rolling max
        rolling_max = df_2008['Equity'].cummax()
        drawdown = (df_2008['Equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # Filter trades
        trades_2008 = []
        for t in trades:
            try:
                entry_dt = pd.to_datetime(t['Entry Date'])
                if entry_dt.year == 2008:
                    trades_2008.append(t)
            except:
                pass

        print(f"\nStrategy: {strat_name}")
        print(f"  Start: ${start_val:,.2f}")
        print(f"  End:   ${end_val:,.2f}")
        print(f"  Return: {((end_val/start_val)-1)*100:.2f}%")
        print(f"  Max Drawdown (2008): {max_dd*100:.2f}%")
        print(f"  Trade Count: {len(trades_2008)}")
        
        if len(trades_2008) > 0:
            print("  Sample Trades:")
            for t in trades_2008[:3]:
                print(f"    {t['Entry Date']} {t['Symbol']} {t['Entry']} -> {t['Exit']}")

if __name__ == "__main__":
    debug_2008()
