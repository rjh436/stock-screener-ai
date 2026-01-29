
import sys
import os
import json
import itertools
import multiprocessing
import pandas as pd
import numpy as np

# Ensure we can import from the project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.strategy_loader import load_strategies
from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

# --- CONFIGURATION GRID ---
EXIT_MODES = ["TRAIL_20SMA", "TRAIL_50SMA", "HYBRID"]
MAX_POSITIONS_GRID = [4, 6, 8, 10]
RISK_PER_TRADE_GRID = [0.01, 0.015, 0.02, 0.025]

def main():
    print("🚀 Starting Growth Optimization (Phase 5)...")

    # 1. Load Data (Once)
    print("📦 Loading Data Pack...")
    symbols = get_universe_symbols("RUSSELL3000")
    # For speed, we might want to filter symbols, but for accuracy we keep R3000
    # The engine handles "prepared_data" reuse, so we load once.
    data = fetch_data_pack(symbols, days=252*7, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=252*7, backtest_mode=True) or {}
    
    print("⚙️  Preparing Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 2. Base Strategy Config
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
             "stop_loss_atr": 2.75, # Fixed from previous win
             "max_pos_size_pct": 0.25 # Allow up to 25% for concentrated bets
        },
        "execution_parameters": {
             "time_stop": 30,
             "rs_floor": 90.0,
             "vol_mult": 1.5,
             "adx_min": 25.0, # Fixed from previous win
             # These will be overwritten by the grid
             "partial_profit_r": 2.0 
        },
        "exit_rules": [] 
    }

    results = []
    
    # 3. Optimization Loop
    combinations = list(itertools.product(EXIT_MODES, MAX_POSITIONS_GRID, RISK_PER_TRADE_GRID))
    total_runs = len(combinations)
    print(f"🔬 Testing {total_runs} Combinations...")

    for i, (exit_mode, max_pos, risk) in enumerate(combinations):
        # -- Configure Variant --
        config = pd.json_normalize(base_config).to_dict(orient='records')[0] # Deep copy-ish hack or just manual dict copy?
        # Manual deep copy to be safe
        config = json.loads(json.dumps(base_config))
        
        params = config.setdefault("execution_parameters", {})
        risk_params = config.setdefault("risk_parameters", {})
        
        # Apply Grid
        risk_params["max_positions"] = max_pos
        risk_params["risk_per_trade"] = risk
        
        # Auto-sizing: If we have 4 positions, max size is 25%.
        # If we have 10 positions, max size is 10%.
        # Let's align max_pos_size_pct to (1/max_pos) roughly, slightly higher to allow cash use.
        risk_params["max_pos_size_pct"] = min(1.0, (1.0 / max_pos) * 1.1)

        # Exit Mode Logic
        if exit_mode == "TRAIL_20SMA":
             config["exit_rules"] = [{"col": "close", "op": "<", "ref": "sma20"}]
             params["partial_profit_r"] = 100.0 # Disable
        elif exit_mode == "TRAIL_50SMA":
             config["exit_rules"] = [{"col": "close", "op": "<", "ref": "sma50"}]
             params["partial_profit_r"] = 100.0 # Disable
        elif exit_mode == "HYBRID":
             config["exit_rules"] = [{"col": "close", "op": "<", "ref": "sma20"}]
             params["partial_profit_r"] = 3.0 # Take half at +3R
        
        # -- Instantiate Strategy --
        # We pass the config to the class directly
        strat = SEPAChampionStrategy(config)
        
        # -- Run Backtest --
        res = run_backtest(
            strat,
            data=prepared, # Pass pre-calculated data!
            start_cash=100000.0,
            start_date="2015-01-01",
            end_date="2021-01-01",
            global_data=g_data
        )
        
        if not res: continue
        res = res if isinstance(res, dict) else res[0]
        
        # -- Log --
        cagr = res.get("cagr", 0.0) * 100
        dd = res.get("max_drawdown_pct", 0.0) # Assume engine calculates this, or we rely on final val?
        # Engine output doesn't seem to have DD metric computed in the snippet I saw?
        # It had "final_value", "total_trades", "hit_rate", "cagr".
        # I'll rely on CAGR and Equity Curve Inspection (or just CAGR/Trades for now).
        
        fv = res.get("final_value", 0)
        tr = res.get("total_trades", 0)
        
        print(f"[{i+1}/{total_runs}] {exit_mode} | Pos:{max_pos} | Risk:{risk*100:.1f}% -> ${fv:,.0f} (CAGR: {cagr:.1f}%) | {tr} Trades")
        
        results.append({
            "Exit": exit_mode,
            "MaxPos": max_pos,
            "Risk": risk,
            "CAGR": cagr,
            "FinalValue": fv,
            "Trades": tr
        })

    # 4. Save & Report
    df_res = pd.DataFrame(results)
    df_res.sort_values("CAGR", ascending=False, inplace=True)
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    print(df_res.head(10).to_string(index=False))
    
    df_res.to_csv("growth_optimization_results.csv", index=False)
    print("✅ Done. Results saved.")

if __name__ == "__main__":
    main()
