import os
import json
import itertools
import pandas as pd
import numpy as np
import sys
import random
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack, clean_dataframe
from data.indices import get_index_symbols
from execution.engine import prepare_backtest_data, run_backtest
from strategies.generic import GenericStrategy

# ============================================================================
# CONFIGURATION: QULLAMAGGIE V2.1 (M3 MAX OPTIMIZED)
# ============================================================================
STRATEGY_TEMPLATE = {
    "name": "Apex Kinetic VCP (Optimizer)",
    "type": "breakout",
    "entry_rules": [
        {"col": "rsi14", "op": ">", "val": 55},
        {"col": "rs_rating", "op": ">", "val": 80},
        {"col": "adr_pct", "op": ">", "val": 2.5}, # Lowered floor per GPT audit
        {"col": "bb_width", "op": "<", "val": 0.25},
        {"col": "close", "op": ">", "ref": "donchian_20"},
        {"col": "volume", "op": ">", "ref": "vol_ma20", "mult": 1.5}
    ],
    "market_filter_mode": "traffic_light",
    "yellow_rs_floor": 85,
    "red_bypass_rs": 95,
    "risk_per_trade": 0.015, # 1.5% Risk
    "max_positions": 10,
    "scoring_type": "breakout",
    "min_entry_score": 0.0
}

# --- SEARCH GRID (Qullamaggie Pillars) ---
PARAM_GRID = {
    "stop_loss_atr": [1.0, 1.5, 2.0],        # Tight stops
    "adr_pct":       [3.0, 4.0, 5.0],        # High Fuel
    "exit_mode":     ["sma10", "sma20", "trail2.5", "trail3.5"], # ISOLATED EXITS
    "rs_rating":     [85, 92],               # Top 15% vs Top 8%
    "bb_width":      [0.15, 0.22],           # Squeeze tightness
    "time_stop":     [20, 45],               # 1 mo vs 2 mo
    "limit_ratio":   [1.0, 1.0005]           # Market vs Buffer
}

# Hardware Optimization
WORKER_COUNT = 10 # Optimized for M3 Max 36GB RAM
_WORKER_CACHE = None

# ============================================================================
# WORKER LOGIC
# ============================================================================
def worker_init(pickled_data):
    """Unpickle the heavy universe data once per process (Efficiency)"""
    global _WORKER_CACHE
    _WORKER_CACHE = pickle.loads(pickled_data)

def worker(params):
    try:
        # 1. Clone & Setup Strategy
        strat = STRATEGY_TEMPLATE.copy()
        strat["name"] = f"QM_{params['exit_mode']}_Stop{params['stop_loss_atr']}_ADR{params['adr_pct']}"
        
        # 2. Apply Dynamic Parameters
        strat["stop_loss_atr"] = params["stop_loss_atr"]
        strat["time_stop"] = params["time_stop"]
        strat["limit_ratio"] = params["limit_ratio"]
        
        # Update Rules
        entry_rules = [r.copy() for r in strat["entry_rules"]]
        for r in entry_rules:
            if r["col"] == "adr_pct":   r["val"] = params["adr_pct"]
            if r["col"] == "rs_rating": r["val"] = params["rs_rating"]
            if r["col"] == "bb_width":  r["val"] = params["bb_width"]
        strat["entry_rules"] = entry_rules
        
        # ISOLATE EXITS (Critical Audit Fix)
        if params["exit_mode"].startswith("trail"):
            strat["trail_atr"] = float(params["exit_mode"].replace("trail", ""))
            strat["exit_rules"] = [] # Clear SMA
        else:
            strat["trail_atr"] = 0
            strat["exit_rules"] = [{"col": "close", "op": "<", "ref": params["exit_mode"]}]

        # 3. Execute Backtest
        res = run_backtest(
            GenericStrategy(strat),
            None, None,
            start_cash=100000.0,
            pre_calculated_data=_WORKER_CACHE
        )

        # 4. Composite Scoring (Reward Skew & PF over Smoothness)
        cagr = res.get("cagr", 0) * 100
        pf = res.get("profit_factor", 0)
        wr = res.get("hit_rate", 0)
        dd = abs(res.get("max_drawdown_pct", 0))
        trades = res.get("total_trades", 0)

        # Qullamaggie Formula: (Return * Quality) / Risk Penality
        # Penalty is mild until DD > 30%
        dd_penalty = 1.0 if dd < 30 else (30 / dd)
        score = (cagr * pf * (wr/100)) * dd_penalty
        
        if trades < 40: score = 0 # Statistical significance

        return {
            "params": params,
            "cagr": cagr, "max_dd": dd, "pf": pf, "wr": wr, "trades": trades,
            "score": score
        }
    except Exception as e:
        return {"error": str(e)}

# ============================================================================
# MAIN LOOP
# ============================================================================
def optimize():
    print("\n🚀 APEX KINETIC V2.1 - QULLAMAGGIE OPTIMIZER")
    print(f"Hardware: M3 Max | Workers: {WORKER_COUNT} | RAM Target: 24GB")
    
    # 1. Load Universe
    print("🌍 Loading Russell 3000 Universe...")
    universe = get_index_symbols("R3000")
    if not universe: universe = get_index_symbols("SP1500")
    
    # 2. Build Cache (Once)
    print("🧠 Pre-calculating indicators for the FULL UNIVERSE...")
    raw_data = fetch_data_pack(universe, backtest_mode=True)
    full_cache = prepare_backtest_data(raw_data, None, None, None)
    
    # 3. Serialize for Workers (Safety Fix)
    print("📦 Serializing cache for workers...")
    pickled_cache = pickle.dumps(full_cache)
    
    # 4. Generate Grid
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"🔍 Testing {len(combinations)} Strategy Variants...")

    # 5. Run Parallel Loop
    results = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, initializer=worker_init, initargs=(pickled_cache,)) as executor:
        futures = [executor.submit(worker, c) for c in combinations]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            r = future.result()
            if "error" in r:
                print(f"⚠️ Error: {r['error']}")
                continue
            
            print(f"[{completed}/{len(combinations)}] ADR:{r['params']['adr_pct']} Exit:{r['params']['exit_mode']} -> CAGR:{r['cagr']:.1f}% | PF:{r['pf']:.2f} | Score:{r['score']:.1f}")
            results.append(r)

    # 6. Report
    df = pd.DataFrame(results).sort_values("score", ascending=False)
    print("\n" + "="*60 + "\n🏆 TOP 5 CONFIGURATIONS\n" + "="*60)
    print(df.head(5)[["params", "cagr", "max_dd", "pf", "trades"]].to_string(index=False))
    
    best = df.iloc[0]
    print(f"\n✅ WINNER:\n{json.dumps(best['params'], indent=2)}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    optimize()
