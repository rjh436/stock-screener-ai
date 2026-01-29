
import sys
import os
import json
import itertools
import multiprocessing
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

# --- FAT TAIL GRID ITERATION 2 ---
STOP_LOSS_GRID = [0.05, 0.07, 0.10]
FAIL_FAST_GRID = [True, False]
# Locking these to widest settings to force volume
RS_LOCKED = 80.0
VOL_LOCKED = 1.0

def main():
    print("🚀 Starting Operation Fat Tail (Phase 8)...")
    print("🎯 Target: >15% CAGR | >300 Trades")

    # 1. Load Data
    print("📦 Loading Data Pack...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=252*7, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=252*7, backtest_mode=True) or {}
    
    print("⚙️  Preparing Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 2. Base Configuration (Swing Mode Baseline)
    base_config = {
        "name": "Apex SEPA Champion (2026)",
        "use_fundamentals": False,
        "entry_rules": [
             {"col": "rs_rating", "op": ">", "val": 90}, 
             {"col": "bb_width", "op": "<", "val": 0.25},
             {"col": "close", "op": ">", "ref": "sma50"},
             {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99}
        ],
        "risk_parameters": {
             "stop_loss_type": "atr",
             "stop_loss_atr": 2.75,
             "max_positions": 8, 
             "max_pos_size_pct": 0.125
        },
        "execution_parameters": {
             "time_stop": 30,
             "partial_profit_day": 5,
             "partial_profit_r": 2.0,
             "vol_mult": 1.5,
             "rs_floor": 80.0, # Widest
             "adx_min": 15.0, # SLASHED from 20 -> 15
             "fail_fast": False # Override me
        },
        "exit_rules": [{"col": "close", "op": "<", "ref": "sma20"}] # Locked to HYBRID/SMA20 for now
    }
    
    # WIDEN VCP FILTER
    for rule in base_config["entry_rules"]:
        if rule["col"] == "bb_width":
             rule["val"] = 0.40 # WIDENED from 0.25

    results = []
    
    combinations = list(itertools.product(STOP_LOSS_GRID, FAIL_FAST_GRID))
    total_runs = len(combinations)
    print(f"🔬 Testing {total_runs} Combinations (Iteration 2)...")

    for i, (stop_loss, fail_fast) in enumerate(combinations):
        config = json.loads(json.dumps(base_config))
        params = config.setdefault("execution_parameters", {})
        risk = config.setdefault("risk_parameters", {})
        
        # Apply Grid
        params["rs_floor"] = RS_LOCKED
        params["vol_mult"] = VOL_LOCKED
        params["fail_fast"] = fail_fast
        
        # FIXED STOP Override
        risk["stop_loss_type"] = "fixed"
        risk["stop_loss_pct"] = stop_loss
        
        # Hybrid Exit Fixed
        params["partial_profit_r"] = 3.0
        
        # Instantiate
        strat = SEPAChampionStrategy(config)
        
        # Run
        res = run_backtest(
            strat,
            data=prepared,
            start_cash=100000.0,
            start_date="2015-01-01",
            end_date="2021-01-01",
            global_data=g_data
        )
        
        if not res: continue
        res = res if isinstance(res, dict) else res[0]
        
        cagr = res.get("cagr", 0.0) * 100
        fv = res.get("final_value", 0)
        tr = res.get("total_trades", 0)
        max_dd = res.get("max_drawdown_pct", 0.0) * 100
        
        print(f"[{i+1}/{total_runs}] Stop:{stop_loss} FF:{fail_fast} -> ${fv:,.0f} (CAGR: {cagr:.1f}%) | {tr} Tr")
        
        results.append({
            "StopLoss": stop_loss,
            "FailFast": fail_fast,
            "CAGR": cagr,
            "FinalValue": fv,
            "Trades": tr,
            "MaxDD": max_dd
        })

    # Report
    df_res = pd.DataFrame(results)
    df_res.sort_values("CAGR", ascending=False, inplace=True)
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    print(df_res.head(10).to_string(index=False))
    
    df_res.to_csv("fat_tail_results.csv", index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()
