import os
import sys
import json
import random
import time
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
from datetime import datetime
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- CORRECTED IMPORTS ---
try:
    from execution.engine import prepare_backtest_data, run_backtest
    from data.loader import fetch_data_pack
    from data.universe import get_universe_symbols
    from strategies.strategy_loader import load_strategies
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RESULTS_FILE = "midnight_results.csv"
BEST_GENOME_FILE = "config/midnight_winner.json"
POPULATION_SIZE = 40  # Reduced slightly for speed
GENERATIONS = 50
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
MAX_WORKERS = 6 

# --- THE "AGGRESSIVE" SEARCH SPACE ---
# Removed "Safe" options to force the AI to take risks
GENE_SPACE = {
    # SELECTION: Looser filters to get more "At Bats"
    "rs_floor": [80, 85, 88],                 # Removed 93, 95 (Too tight)
    "vol_mult": [1.5, 2.0],                   # Removed 2.5, 3.0 (Too picky)
    "adx_min": [15, 20, 25],                  # Removed 30 (Too late)
    "bb_width_max": [0.20, 0.25, 0.30],       # Removed 0.15 (Too restrictive)
    
    # TIMING
    "regime_ma": ["sma200"],                  # Hardcode to 200 for now
    
    # EXIT MECHANICS
    "exit_sma": ["sma20", "sma50"],
    "stop_loss_atr": [3.0, 4.0, 5.0],         # Wider stops to survive volatility
    "partial_profit_day": [3, 5, 8],          # Force fast recycling
    
    # SIZING: Forced Concentration
    "max_positions": [4, 5],                  # Max 5 stocks. Period.
    "risk_per_trade": [0.02, 0.025]           # Bet 2.0% - 2.5% per trade
}

# --- GLOBAL DATA REF ---
_prepared_data_ref = None
_g_data_ref = None

def generate_random_genome():
    return {k: random.choice(v) for k, v in GENE_SPACE.items()}

def mutate_genome(genome):
    new_genome = genome.copy()
    gene = random.choice(list(GENE_SPACE.keys()))
    new_genome[gene] = random.choice(GENE_SPACE[gene])
    return new_genome

def crossover(parent1, parent2):
    child = {}
    for key in GENE_SPACE.keys():
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def compress_data(prepared_obj):
    print("🗜️  Compressing Data (float32)...")
    for sym, s_data in prepared_obj.enriched.items():
        cols = s_data.df.select_dtypes(include=['float64']).columns
        s_data.df[cols] = s_data.df[cols].astype('float32')
        try:
            for attr in ['close', 'high', 'low', 'open', 'volume', 'rs_rating', 'adx', 'sma50', 'sma200']:
                if hasattr(s_data, attr):
                    val = getattr(s_data, attr)
                    if isinstance(val, np.ndarray) and val.dtype == 'float64':
                        setattr(s_data, attr, val.astype('float32'))
        except Exception: pass
    return prepared_obj

def init_worker(prepared_data, g_data):
    global _prepared_data_ref, _g_data_ref
    _prepared_data_ref = prepared_data
    _g_data_ref = g_data

def evaluate_genome(genome_id_and_genome):
    genome_id, genome = genome_id_and_genome
    global _prepared_data_ref, _g_data_ref
    
    if _prepared_data_ref is None: return {"id": genome_id, "score": -999, "error": "Init failed"}

    try:
        # Construct Config
        pos_size = 1.0 / genome["max_positions"]
        
        strategy_config = {
            "name": f"Gen_{genome_id}",
            "parameters": {
                "min_rs": genome["rs_floor"],
                "vol_ma_ratio": genome["vol_mult"],
                "adx_threshold": genome["adx_min"],
                "bb_width_threshold": genome["bb_width_max"],
                "regime_ma": genome["regime_ma"]
            },
            "risk_management": {
                "stop_loss_atr": genome["stop_loss_atr"],
                "max_positions": genome["max_positions"],
                "risk_per_trade": genome["risk_per_trade"],
                "max_pos_size_pct": pos_size
            },
            "execution": {
                "exit_sma": genome["exit_sma"],
                "partial_profit_day": int(genome["partial_profit_day"]),
                "partial_profit_ratio": 0.5
            }
        }
        
        strategies = load_strategies([strategy_config])
        strat = strategies[0]
        # Force attributes
        strat.min_rs = genome["rs_floor"]
        strat.vol_ma_ratio = genome["vol_mult"]
        strat.adx_threshold = genome["adx_min"]
        
        # Run Backtest
        result = run_backtest(
            [strat], 
            _prepared_data_ref, 
            start_cash=100000.0, 
            start_date=START_DATE, 
            end_date=END_DATE,
            global_data=_g_data_ref
        )
        
        if not result: return {"id": genome_id, "score": 0, "error": "No result"}
        metrics = result[0] if isinstance(result, list) else result
        
        # --- FIXING THE METRIC BUG ---
        final_val = metrics.get("final_value", 100000)
        trades = metrics.get("total_trades", 0)
        
        # Hunt for the Drawdown Key
        raw_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0)))
        
        # Normalize: If it looks like a decimal (0.15), make it percent (15.0)
        if raw_dd < 1.0 and raw_dd > 0.0:
            max_dd = raw_dd * 100.0
        else:
            max_dd = raw_dd
            
        # CAGR Logic
        years = 16 
        cagr_pct = ((final_val / 100000.0) ** (1/years) - 1) * 100
        
        # --- NEW FITNESS FUNCTION (Force Growth) ---
        score = cagr_pct
        
        # Punish Low Growth Heavily
        if cagr_pct < 10.0: score -= 50.0 
        
        # Punish Extreme Drawdown
        if max_dd > 35.0: score -= (max_dd * 2)
        
        # Punish Inactivity
        if trades < 100: score = -100.0

        return {
            "id": genome_id,
            "genome": genome,
            "score": score,
            "cagr": cagr_pct,
            "dd": max_dd,
            "trades": trades,
            "raw_metrics_keys": list(metrics.keys()) # DEBUG INFO
        }

    except Exception as e:
        return {"id": genome_id, "score": -999, "error": str(e)}

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError: pass

    print(f"🚀 OPERATION UNSTUCK: Aggressive Optimization")
    print(f"HARDWARE: M3 Max | WORKERS: {MAX_WORKERS}")
    
    print("...Loading Data...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=5800, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=5800, backtest_mode=True)
    
    print("...Preparing & Compressing...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    prepared = compress_data(prepared)
    
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        print(f"\n🧬 GEN {gen+1}/{GENERATIONS}")
        start_time = time.time()
        
        tasks = [(i, g) for i, g in enumerate(population)]
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(prepared, g_data)) as executor:
            futures = [executor.submit(evaluate_genome, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                if "error" not in res:
                    # PRINT KEYS ONCE TO DEBUG
                    if gen == 0 and res['id'] == 0:
                         print(f"🔎 DEBUG: Available Keys: {res['raw_metrics_keys']}")
                    
                    print(f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | DD:{res['dd']:.1f}%")
        
        valid = [r for r in results if "error" not in r]
        if not valid:
            print("CRITICAL: All failed.")
            break
            
        valid.sort(key=lambda x: x["score"], reverse=True)
        winner = valid[0]
        
        print(f"🏆 WINNER: CAGR {winner['cagr']:.2f}% | DD {winner['dd']:.2f}% | Score: {winner['score']:.1f}")
        print(f"🧬 DNA: {winner['genome']}")
        
        with open(BEST_GENOME_FILE, "w") as f:
            json.dump(winner["genome"], f, indent=4)
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{gen+1},{winner['cagr']},{winner['dd']},{winner['trades']},\"{winner['genome']}\"\n")
            
        # Breeding
        next_gen = [r["genome"] for r in valid[:8]]
        while len(next_gen) < POPULATION_SIZE:
            p1 = random.choice(valid[:15])["genome"]
            p2 = random.choice(valid[:15])["genome"]
            child = crossover(p1, p2)
            if random.random() < 0.4: child = mutate_genome(child) # High mutation
            next_gen.append(child)
        population = next_gen
        print(f"⏱️  {time.time()-start_time:.1f}s")
