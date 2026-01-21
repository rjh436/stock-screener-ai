
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
from execution.engine import (
    run_backtest,
    prepare_backtest_data,
)
from strategies.strategy_loader import load_strategies

def main():
    print("--- APEX V17 2020 STRESS TEST ---")
    
    # 1. Load Strategy
    CONFIG_PATH = "config/generated_strategies.json"
    with open(CONFIG_PATH, "r") as f:
        strategies_config = json.load(f)
    
    target_name = "Apex Qullamaggie V17 (High Octane)"
    strat_configs = [s for s in strategies_config if s.get("name") == target_name]
    
    # Patch: Flatten nested parameters so engine.py can see them
    for cfg in strat_configs:
        if "risk_parameters" in cfg:
            cfg.update(cfg.pop("risk_parameters"))
        if "execution_parameters" in cfg:
            cfg.update(cfg.pop("execution_parameters"))

    
    if not strat_configs:
        print(f"ERROR: Strategy '{target_name}' not found!")
        print("Available:", [s.get("name") for s in strategies_config])
        return

    strategies = load_strategies(strat_configs)
    print(f"Loaded: {[s.name for s in strategies]}")

    # 2. Fetch Data (Enough to cover 2020 + warmup)
    # 2020-01-01 is roughly 6 years ago. 252 * 6 = 1512 trading days.
    # We want data starting from e.g. 2018.
    print("Fetching Data (Russell 3000)...")
    # Fetching last 8 years to be safe and filtering later, or just verify_v11 approach
    # verify_v11 uses 5040 days. Let's stick to that to ensure indicators are identical.
    days = 5040 
    symbols = get_universe_symbols("RUSSELL3000")
    print(f"Universe Size: {len(symbols)}")
    
    data = fetch_data_pack(symbols, days=days + 200, backtest_mode=True)
    if not data:
        print("ERROR: No data returned.")
        return

    g_data = fetch_data_pack(["SPY", "VIX"], days=days + 200, backtest_mode=True) or {}
    global_data = {
        "SPY": g_data.get("SPY"),
        "VIX": g_data.get("VIX")
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
    # Note: run_backtest might not take end_date, so we run full and filter results
    results = run_backtest(
        strategies,
        prepared,
        start_cash=100000.0,
        start_date=None, 
        global_data=global_data,
    )
    
    if not isinstance(results, list):
        results = [results]

    # 5. Output Metrics focusing on 2020
    for res in results:
        print("\n" + "="*40)
        print(f"RESULTS: {res.get('strategy', 'Unknown')}")
        print("="*40)
        
        curve = res.get('equity_curve', [])
        if curve:
            # 2020 Return filtering
            try:
                start_2020 = next((x for x in curve if x['Date'].year == 2020), None)
                end_2020 = next((x for x in reversed(curve) if x['Date'].year == 2020), None)
                if start_2020 and end_2020:
                     val_start = start_2020['Equity']
                     val_end = end_2020['Equity']
                     ret_2020 = (val_end - val_start) / val_start
                     print(f"2020 Return:    {ret_2020:.2%}")
                     print(f"2020 Start Eq:  ${val_start:,.2f}")
                     print(f"2020 End Eq:    ${val_end:,.2f}")
                else:
                    print("2020 Data Not Found in Equity Curve")
            except Exception as e:
                print(f"2020 Check Err: {e}")
        
        print(f"Total Trades:   {res.get('total_trades', 0)}")
        print(f"Final Equity:   ${res.get('final_value', 0):,.2f}")
        print("="*40)

if __name__ == "__main__":
    main()
