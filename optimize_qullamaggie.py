import os
import json
import itertools
import pandas as pd
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CORRECTED IMPORTS
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest
from strategies.generic import GenericStrategy

# --- CONFIGURATION ---
STRATEGY_TEMPLATE = {
    "name": "Apex Kinetic VCP (Optimizer)",
    "type": "breakout",
    "entry_rules": [
        {"col": "rsi14", "op": ">", "val": 55},
        {"col": "rs_rating", "op": ">", "val": 80},
        {"col": "adr_pct", "op": ">", "val": 3.0},
        {"col": "bb_width", "op": "<", "val": 0.25},
        {"col": "close", "op": ">", "ref": "donchian_20"},
        {"col": "volume", "op": ">", "ref": "vol_ma20", "mult": 1.5}
    ],
    "exit_rules": [
        {"col": "close", "op": "<", "ref": "sma20"}
    ],
    "market_filter_mode": "traffic_light",
    "yellow_rs_floor": 85,
    "red_bypass_rs": 95,
    "stop_loss_atr": 2.0,
    "trail_atr": 0,
    "time_stop": 45,
    "risk_per_trade": 0.02,
    "max_positions": 12,
    "scoring_type": "breakout",
    "scoring_weights": {"rsi_factor": 2.0, "sniper_bonus": 150.0},
    "min_entry_score": 0.0
}

# --- THE SEARCH GRID (V2) ---
PARAM_GRID = {
    "stop_loss_atr": [1.5, 2.0, 2.5],        # Risk Management
    "rs_rating":     [80, 85, 90],           # Leader Quality
    "exit_sma":      ["sma10", "sma20"],     # Trend Duration
    "adr_pct":       [2.5, 3.5, 4.5],        # Volatility Fuel
    "red_bypass_rs": [95, 101]               # 101 = Disabled
}

def get_data():
    """Load data once to share across workers"""
    print("Loading Data Pack...")
    try:
        # FORCE RUSSELL 3000
        print("Requesting Russell 3000 Universe...")
        universe = get_index_symbols("R3000")

        if not universe or len(universe) < 2000:
            print("Warning: 'R3000' key missing or too small. Trying 'IWV' (Russell 3000 ETF)...")
            universe = get_index_symbols("IWV")

        if not universe:
            # Absolute fallback if indices file is empty
            print("Warning: 'IWV' failed. Loading S&P 1500 as fallback.")
            universe = get_index_symbols("SP1500")

        print(f"Fetching data for {len(universe)} symbols...")
        data = fetch_data_pack(universe, backtest_mode=True)
        if not data:
            data = fetch_data_pack(universe)
        return data

    except Exception as e:
        print(f"ERROR loading data: {e}")
        # Fallback
        from data.cache_manager import load_all_data
        return load_all_data()

def worker(params, data_pack):
    """Runs a single backtest for a parameter set"""
    try:
        # Clone Strategy
        strat = STRATEGY_TEMPLATE.copy()
        strat["name"] = f"QM_Stop{params['stop_loss_atr']}_ADR{params['adr_pct']}_{params['exit_sma']}"

        # Apply Scalar Params
        strat["stop_loss_atr"] = params["stop_loss_atr"]
        strat["red_bypass_rs"] = params["red_bypass_rs"]

        # Update Rules
        entry_rules = [r.copy() for r in strat["entry_rules"]]
        for r in entry_rules:
            if r["col"] == "rs_rating":
                r["val"] = params["rs_rating"]
            if r["col"] == "adr_pct":
                r["val"] = params["adr_pct"]
        strat["entry_rules"] = entry_rules

        strat["exit_rules"] = [{"col": "close", "op": "<", "ref": params["exit_sma"]}]

        # Run Backtest
        res = run_backtest(GenericStrategy(strat), data_pack, None, start_cash=100000.0)

        return {
            "params": params,
            "cagr": res.get("cagr", 0),
            "max_dd": res.get("max_drawdown_pct", 0),
            "trades": res.get("total_trades", 0),
            "win_rate": res.get("hit_rate", 0),
            "profit_factor": res.get("profit_factor", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def optimize():
    data = get_data()
    if not data:
        print("ERROR: No data loaded. Check loader.")
        return

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Starting V2 Optimization: {len(combinations)} Strategies")

    def run_pool(executor_cls, label):
        results = []
        errors = []
        with executor_cls(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(worker, combo, data) for combo in combinations]

            for i, f in enumerate(futures):
                res = f.result()
                if "error" in res:
                    errors.append(res["error"])
                    continue
                # Score = CAGR but kill if DD > 30% or Trades < 50
                score = res["cagr"]
                if abs(res["max_dd"]) > 30.0:
                    score *= 0.5
                if res["trades"] < 50:
                    score = 0

                res["score"] = score
                results.append(res)
                print(f"[{i+1}/{len(combinations)}] ADR:{res['params']['adr_pct']} Stop:{res['params']['stop_loss_atr']} -> CAGR: {res['cagr']:.1%} DD: {res['max_dd']:.1%}")
        if errors:
            print(f"{label} errors: {len(errors)}")
            print("Sample error:", errors[0])
        return results

    # Try process pool first, then thread pool if needed
    results = run_pool(ProcessPoolExecutor, "ProcessPoolExecutor")
    if not results:
        print("No results from process pool, retrying with ThreadPoolExecutor...")
        results = run_pool(ThreadPoolExecutor, "ThreadPoolExecutor")
        if not results:
            print("ERROR: No results generated.")
            return

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    print("\nTOP 5 V2 CONFIGURATIONS")
    print(df.head(5)[["params", "cagr", "max_dd", "trades", "profit_factor"]])

    best = df.iloc[0]
    print(f"\nWINNER: {best['params']}")

if __name__ == "__main__":
    optimize()
