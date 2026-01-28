
import json
import os
import sys
import csv
import copy
import numpy as np
import pandas as pd
from itertools import product

# --- PATH SETUP ---
sys.path.append(os.getcwd())

# --- IMPORTS ---
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from strategies.strategy_loader import load_strategies
from execution.engine import run_backtest, prepare_backtest_data

# --- CONFIG ---
STRATEGY_NAME = "Apex SEPA Champion (2026)"
CONFIG_PATH = os.path.join("config", "generated_strategies.json")
RESULTS_CSV = "optimization_results_v2.csv"
START_DATE = "2015-01-01"
END_DATE = "2021-01-01"

# --- SEARCH GRID ---
GRID = {
    "rs_floor": [90.0, 93.0, 95.0, 97.0],
    "vol_mult": [1.5, 2.0, 2.5, 3.0],
    "adx_min": [15, 20, 25],
    "stop_loss_atr": [2.0, 2.5, 2.75, 3.0]
}

def load_base_config():
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    for s in data:
        if s["name"] == STRATEGY_NAME:
            return s
    raise ValueError(f"Strategy {STRATEGY_NAME} not found!")

def main():
    print(f"🚀 Starting In-Process Optimization for {STRATEGY_NAME}")
    
    # 1. LOAD DATA (ONCE)
    print("📦 Loading Unified Data Pack (This happens only once)...")
    symbols = get_universe_symbols("RUSSELL3000")
    # Limit universe slightly for speed if needed, but let's try full
    # symbols = symbols[:500] 
    
    data = fetch_data_pack(symbols, days=5040, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=5040, backtest_mode=True)
    
    print("⚙️ Preparing Backtest Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    print("✅ Data Ready.")

    # 2. SETUP CSV
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["RS_Floor", "Vol_Mult", "ADX_Min", "Stop_ATR", "Final_Value", "Trades", "Win_Rate"])

    # 3. LOOPS
    combinations = list(product(GRID["rs_floor"], GRID["vol_mult"], GRID["adx_min"], GRID["stop_loss_atr"]))
    total_runs = len(combinations)
    
    base_strat_config = load_base_config()
    best_val = 0.0
    best_cfg = None

    print(f"🔎 Testing {total_runs} combinations...")
    
    for i, (rs, vol, adx, stop) in enumerate(combinations):
        # Update Config in Memory
        current_config = copy.deepcopy(base_strat_config)
        
        # Inject Parameters
        # Note: We inject into 'execution_parameters' as a generic container derived from Step 1 logic
        if "execution_parameters" not in current_config:
            current_config["execution_parameters"] = {}
        if "risk_parameters" not in current_config:
            current_config["risk_parameters"] = {}
            
        current_config["execution_parameters"]["rs_floor"] = rs
        current_config["execution_parameters"]["vol_mult"] = vol
        current_config["execution_parameters"]["adx_min"] = adx
        current_config["risk_parameters"]["stop_loss_atr"] = stop
        
        # Load Strategy Object
        strategies = load_strategies([current_config])
        
        # Run Backtest
        # Suppress stdout for the run? engine.py run_backtest might print "Avg Universe..."
        # We can accept some noise or suppress it.
        try:
            res = run_backtest(
                strategies, 
                prepared, 
                start_cash=100000.0, 
                start_date=START_DATE, 
                end_date=END_DATE, 
                global_data=g_data
            )
            
            if isinstance(res, list) and res:
                res = res[0]
            
            val = res["final_value"]
            trades = res["total_trades"]
            wr = res["hit_rate"]
            
            # Print minimal status
            print(f"[{i+1}/{total_runs}] {rs}/{vol}/{adx}/{stop} -> ${val:,.0f} ({trades} trts, {wr:.1f}%)")
            
            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([rs, vol, adx, stop, val, trades, wr])
                
            if trades > 80 and val > best_val:
                best_val = val
                best_cfg = (rs, vol, adx, stop)
                
        except Exception as e:
            print(f"❌ Error run {i}: {e}")

    print("\n🏆 OPTIMIZATION COMPLETE 🏆")
    if best_cfg:
        print(f"Top Result: ${best_val:,.2f}")
        print(f"Parameters: RS={best_cfg[0]}, Vol={best_cfg[1]}, ADX={best_cfg[2]}, Stop={best_cfg[3]}")
    else:
        print("No valid result found.")

if __name__ == "__main__":
    main()
