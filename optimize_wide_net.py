
import sys
import os
import json
import itertools
import multiprocessing
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.strategy_loader import load_strategies
from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

# --- WIDE NET GRID ---
RS_FLOOR_GRID = [80.0, 85.0, 90.0]
BB_WIDTH_MAX_GRID = [0.25, 0.35, 0.45]
ADX_MIN_GRID = [20.0, 25.0]

def main():
    print("🚀 Starting Wide Net Optimization (Phase 5b)...")

    # 1. Load Data
    print("📦 Loading Data Pack...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=252*7, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=252*7, backtest_mode=True) or {}
    
    print("⚙️  Preparing Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 2. Base Config
    base_config = {
        "name": "Apex SEPA Champion (2026)",
        "use_fundamentals": False,
        "entry_rules": [
             # These will be updated dynamically in the loop?
             # Actually, engine.py respects 'execution_parameters' overrides for rs_floor/adx.
             # But 'bb_width' entry rule hardcoded in entry_rules dict needs update.
             # And 'rs_rating' entry rule hardcoded needs update.
             
             # Placeholder rules
             {"col": "rs_rating", "op": ">", "val": 90}, 
             {"col": "bb_width", "op": "<", "val": 0.25},
             {"col": "close", "op": ">", "ref": "sma50"},
             {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99}
        ],
        "risk_parameters": {
             "stop_loss_type": "atr",
             "stop_loss_atr": 2.75,
             "max_positions": 8, # Diversify slightly
             "max_pos_size_pct": 0.125
        },
        "execution_parameters": {
             "time_stop": 30,
             "partial_profit_day": 5,
             "partial_profit_r": 2.0,
             "vol_mult": 1.5,
             
             # Overrides
             "rs_floor": 90.0,
             "adx_min": 25.0,
             "bb_width_max": 0.25 # Engine reads this in the loop?
        },
        "exit_rules": [{"col": "close", "op": "<", "ref": "sma10"}] # Standard exit
    }

    results = []
    
    combinations = list(itertools.product(RS_FLOOR_GRID, BB_WIDTH_MAX_GRID, ADX_MIN_GRID))
    total_runs = len(combinations)
    print(f"🔬 Testing {total_runs} Combinations...")

    for i, (rs, bb, adx) in enumerate(combinations):
        # Deep copy
        config = json.loads(json.dumps(base_config))
        params = config.setdefault("execution_parameters", {})
        
        # Update Params
        params["rs_floor"] = rs
        params["adx_min"] = adx
        params["bb_width_max"] = bb
        
        # Update Entry Rules Config (Critical for rule_pass check)
        entry_rules = config.setdefault("entry_rules", [])
        for rule in entry_rules:
            if rule.get("col") == "rs_rating":
                rule["val"] = rs
            if rule.get("col") == "bb_width":
                rule["val"] = bb
        
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
        
        print(f"[{i+1}/{total_runs}] RS:{rs} | BB:{bb} | ADX:{adx} -> ${fv:,.0f} (CAGR: {cagr:.1f}%) | {tr} Trades")
        
        results.append({
            "RS": rs,
            "BB": bb,
            "ADX": adx,
            "CAGR": cagr,
            "FinalValue": fv,
            "Trades": tr
        })

    # Report
    df_res = pd.DataFrame(results)
    df_res.sort_values("CAGR", ascending=False, inplace=True)
    print("\n🏆 TOP 5 CONFIGURATIONS:")
    print(df_res.head(10).to_string(index=False))
    
    df_res.to_csv("wide_net_results.csv", index=False)
    print("✅ Done.")

if __name__ == "__main__":
    main()
