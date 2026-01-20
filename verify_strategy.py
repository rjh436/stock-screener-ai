
import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from data.indices import get_index_symbols
from execution.engine import (
    run_backtest,
    prepare_backtest_data,
)
from strategies.strategy_loader import load_strategies

def main():
    print("--- HEADLESS BACKTEST VERIFICATION ---")
    
    # 1. Load Strategy
    CONFIG_PATH = "config/generated_strategies.json"
    with open(CONFIG_PATH, "r") as f:
        strategies_config = json.load(f)
    
    # Filter for Target Strategies
    target_names = [
        "Apex Qullamaggie V10 (Nitro - Max)"
    ]
    strat_configs = [s for s in strategies_config if s.get("name") in target_names]
    
    if not strat_configs:
        print(f"ERROR: No target strategies found in config!")
        # Fallback to first strat if specific name matching fails (e.g. slight typo)
        print("Available strategies:", [s.get("name") for s in strategies_config])
        return

    strategies = load_strategies(strat_configs)
    print(f"Loaded Strategies: {[s.name for s in strategies]}")

    # 2. Fetch Data (Russell 3000, 20 Years)
    print("Fetching Data (Russell 3000)...")
    days = 5040 
    
    # Use existing caching logic from loader if available, or just fetch.
    symbols = get_universe_symbols("RUSSELL3000")
    print(f"Universe Size: {len(symbols)}")
    
    data = fetch_data_pack(symbols, days=days + 200, backtest_mode=True)
    if not data:
        print("ERROR: No data returned.")
        return

    # Fetch global context
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days + 200, backtest_mode=True) or {}
    vix_df = g_data.get("$VIX")
    if vix_df is None:
        vix_df = g_data.get("VIX")
        
    global_data = {
        "SPY": g_data.get("SPY"),
        "VIX": vix_df
    }

    # 3. Prepare Data
    print("Preparing Data...")
    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=None,
        global_data=global_data,
    )

    # 4. Run Backtest
    print("Running Backtest...")
    from execution.engine import run_backtest
    
    results = run_backtest(
        strategies,
        prepared,
        start_cash=100000.0,
        start_date=None,
        global_data=global_data,
    )
    
    # Ensure results is a list
    if not isinstance(results, list):
        results = [results]

    # 5. Output Metrics
    for res in results:
        print("\n" + "="*40)
        print(f"RESULTS: {res.get('strategy_name', 'Unknown')}")
        print("="*40)
        print(f"CAGR:           {res.get('cagr', 0):.2%}")
        print(f"Win Rate:       {res.get('hit_rate', 0):.1f}%")
        print(f"Total Trades:   {res.get('total_trades', 0)}")
        print(f"Avg Profit:     {res.get('avg_profit_pct', 0):.2f}%")
        print(f"Max Drawdown:   {res.get('max_drawdown_pct', 0):.2f}%")
        print(f"Final Equity:   ${res.get('final_value', 0):,.2f}")
        print("="*40)

if __name__ == "__main__":
    main()
