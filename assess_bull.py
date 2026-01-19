import sys
import os
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

def assess_bull():
    # Load Universe - Concentrated 2020 Growth Basket for speed & relevance
    print("Fetching symbols for Growth Basket...")
    universe_symbols = ["TSLA", "NVDA", "AAPL", "AMD", "MSFT", "AMZN", "META", "GOOGL", "ZM", "PTON", "MRNA", "ENPH", "SEDG", "FSLR", "CRWD", "DDOG", "NET", "SHOP"]
    
    if not universe_symbols:

        print("Failed to get symbols.")
        return

    print(f"Loaded {len(universe_symbols)} symbols.")

    print("Loading data for 2019-2022...")
    # Load enough data for 2020-2021
    # Start 2019 for warmup. End 2021.
    days_needed = 365 * 10 
    
    global_data = fetch_data_pack(["SPY", "VIX"], days=days_needed)
    data = fetch_data_pack(universe_symbols, days=days_needed)
    
    if not data:
        print("No stock data loaded.")
        return
        
    global_context = {
        "SPY": global_data.get("SPY"),
        "VIX": global_data.get("VIX") or global_data.get("$VIX")
    }

    # Load Strategies
    print("Loading strategies...")
    try:
        with open("config/generated_strategies.json", "r") as f:
            configs = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    strategies = load_strategies(configs)
    
    start_date = "2019-06-01" 
    
    print("Running Backtest for Bull Market Assessment (2020-2021)...")
    results = run_backtest(
        strategies, 
        data, 
        start_date=start_date, 
        global_data=global_context,
        start_cash=100000.0
    )
    
    if not isinstance(results, list):
        results = [results]
        
    print("\n" + "="*50)
    print("2020-2021 BULL MARKET RESULTS")
    print("="*50)

    for res in results:
        strat_name = res.get('strategy', 'Unknown')
        curve = res.get('equity_curve', [])
        
        if not curve:
            print(f"Strategy: {strat_name} - NO EQUITY CURVE")
            continue
            
        # Create DF
        df = pd.DataFrame(curve)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Slice 2020-2021
        df_bull = df['2020-01-01':'2021-12-31']
        
        if df_bull.empty:
            print(f"Strategy: {strat_name} - NO DATA FOR 2020-2021")
            continue
            
        start_val = df_bull.iloc[0]['Equity']
        end_val = df_bull.iloc[-1]['Equity']
        
        # Calculate CAGR approx for 2 years
        total_ret = (end_val / start_val) - 1.0
        cagr = ((end_val / start_val) ** (1/2)) - 1.0
        
        trade_count = res.get('total_trades', 0)
        win_rate = res.get('hit_rate', 0.0)
        avg_profit = res.get('avg_profit_pct', 0.0)

        print(f"\nStrategy: {strat_name}")
        print(f"  Start: ${start_val:,.2f}")
        print(f"  End:   ${end_val:,.2f}")
        print(f"  Total Return (2yr): {total_ret*100:.2f}%")
        print(f"  CAGR (Bull): {cagr*100:.2f}%")
        print(f"  Trades: {trade_count}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Avg Profit: {avg_profit:.2f}%")
        
        if cagr < 0.25:
             print("  [WARNING] UNDERPERFORMING: CAGR < 25%")
        else:
             print("  [SUCCESS] TARGET MET: CAGR > 25%")

if __name__ == "__main__":
    assess_bull()
