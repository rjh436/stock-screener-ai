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

# --- IMPORTS (DEBUG MODE - NO ERROR CATCHING) ---
# We removed the try/except block to see the REAL error trace.
from execution.engine import prepare_backtest_data, run_backtest
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from strategies.strategy_loader import load_strategies

# --- CONFIGURATION (M3 MAX OPTIMIZED) ---
RESULTS_FILE = "midnight_results.csv"
BEST_GENOME_FILE = "config/midnight_winner.json"
POPULATION_SIZE = 50
GENERATIONS = 100
START_DATE = "2010-01-01"  # Full Cycle
END_DATE = "2025-12-31"

# HARDWARE TUNING:
# M3 Max has ~14 cores, but 36GB RAM limits us.
# With float32 compression, we can safely run 6-8 workers.
MAX_WORKERS = 6 

# --- THE "ALPHAG" SEARCH SPACE ---
# Audit Findings: Added "Concentration" and "Velocity" genes.
GENE_SPACE = {
    # 1. SELECTION (Quality)
    "rs_floor": [85, 87, 90, 93, 95],         # Tighter = Less Churn
    "vol_mult": [1.5, 2.0],                   # Volume Quality (Looser)
    "adx_min": [15, 20, 25],                  # Trend Strength (Catch trends earlier)
    "bb_width_max": [0.20, 0.25, 0.30],       # VCP Tightness (Looser for velocity)
    
    # 2. MARKET TIMING (Survival)
    "regime_ma": ["sma150", "sma200"],        # Bear Market Filter
    
    # 3. EXIT MECHANICS (Velocity)
    "exit_sma": ["sma10", "sma20", "sma50"],
    "stop_loss_atr": [2.0, 3.0, 4.0, 5.0],
    "partial_profit_day": [3, 5, 10, 999],    # 999 = Disabled. 3-5 = High Velocity.
    
    # 4. SIZING (Concentration - The Key to 30%)
    "max_positions": [4, 5, 6],               # FORCE CONCENTRATION. No 10-stock portfolios.
    "risk_per_trade": [0.015, 0.020, 0.025]   # Bet bigger on fewer stocks.
}

# --- GLOBAL DATA REF (For Workers) ---
_prepared_data_ref = None
_g_data_ref = None

def generate_random_genome():
    return {k: random.choice(v) for k, v in GENE_SPACE.items()}

def mutate_genome(genome):
    new_genome = genome.copy()
    # Mutate 1-2 genes
    for _ in range(random.randint(1, 2)):
        gene = random.choice(list(GENE_SPACE.keys()))
        new_genome[gene] = random.choice(GENE_SPACE[gene])
    return new_genome

def crossover(parent1, parent2):
    child = {}
    for key in GENE_SPACE.keys():
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child

def compress_data(prepared_obj):
    """
    TURBO PATCH: Downcast all float64 to float32.
    Reduces RAM usage by ~45%, preventing M3 Max crashes.
    """
    print("🗜️  Compressing Data for M3 Max (float64 -> float32)...")
    for sym, s_data in prepared_obj.enriched.items():
        # Downcast DataFrame
        cols = s_data.df.select_dtypes(include=['float64']).columns
        s_data.df[cols] = s_data.df[cols].astype('float32')
        
        # Downcast Dataclass Arrays
        # (Assuming _SymbolArrays structure from engine.py)
        try:
            # Common attributes in SymbolArrays
            for attr in ['close', 'high', 'low', 'open', 'volume', 'rs_rating', 'adx', 'sma50', 'sma200']:
                if hasattr(s_data, attr):
                    val = getattr(s_data, attr)
                    if isinstance(val, np.ndarray) and val.dtype == 'float64':
                        setattr(s_data, attr, val.astype('float32'))
        except Exception:
            pass # Safety pass
            
    return prepared_obj

def init_worker(prepared_data, g_data):
    """Initializes global state for each worker process to avoid pickling overhead."""
    global _prepared_data_ref
    global _g_data_ref
    _prepared_data_ref = prepared_data
    _g_data_ref = g_data

def evaluate_genome(genome_id_and_genome):
    """Worker function to test a strategy."""
    genome_id, genome = genome_id_and_genome
    global _prepared_data_ref
    global _g_data_ref
    
    if _prepared_data_ref is None:
        return {"id": genome_id, "score": -999, "error": "Worker init failed"}

    try:
        # 1. DYNAMIC CONFIG GENERATION
        # We construct a full config dictionary based on the genes
        
        # Calculate sizing based on concentration
        # If max_pos=4, size=0.25. If max_pos=5, size=0.20
        pos_size = 1.0 / genome["max_positions"] 
        
        strategy_config = {
            "name": f"Midnight_Gen_{genome_id}",
            "strategy_id": f"gen_{genome_id}",
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
                "partial_profit_ratio": 0.5 # Sell half if taking partials
            }
        }
        
        # HACK: Because engine.py expects specific object structures, 
        # we will load a Default Template and INJECT values.
        strategies = load_strategies([strategy_config])
        strat = strategies[0]
        
        # Manually force attributes that might not map 1:1 in loader
        strat.min_rs = genome["rs_floor"]
        strat.vol_ma_ratio = genome["vol_mult"]
        strat.adx_threshold = genome["adx_min"]
        
        # 2. RUN BACKTEST
        # Use our worker-local data references
        result = run_backtest(
            [strat], 
            _prepared_data_ref, 
            start_cash=100000.0, 
            start_date=START_DATE, 
            end_date=END_DATE,
            global_data=_g_data_ref
        )
        
        if not result:
            return {"id": genome_id, "score": 0, "error": "No result"}

        # Handle list vs dict return
        metrics = result[0] if isinstance(result, list) else result
        
        # 3. CALCULATE FITNESS (The "Minervini Score")
        final_val = metrics.get("final_value", 100000)
        max_dd = metrics.get("max_drawdown", 0)
        trades = metrics.get("total_trades", 0)
        
        # CAGR
        years = 16 # 2010 to 2026
        cagr = (final_val / 100000.0) ** (1/years) - 1
        cagr_pct = cagr * 100
        
        # Validating Drawdown Calculation
        print(f"DEBUG: Gen {genome_id} | Final: {final_val} | DD: {max_dd}")

        # Scoring Logic
        # We want CAGR > 20%, but huge penalty for DD > 30%
        score = cagr_pct
        
        # Penalties
        if cagr_pct < 15.0: score *= 0.1 # Force Growth
        if max_dd > 25.0: score *= 0.5   # Soft ceiling
        if max_dd > 40.0: score = -10.0  # Hard reject
        if trades < 50: score = 0.0      # Inactive
        
        # Bonuses
        if trades > 300: score *= 1.1    # Reward velocity
        if cagr_pct > 25: score *= 1.2   # Reward superperformance

        return {
            "id": genome_id,
            "genome": genome,
            "score": score,
            "cagr": cagr_pct,
            "dd": max_dd,
            "trades": trades,
            "final_value": final_val
        }

    except Exception as e:
        return {"id": genome_id, "score": -999, "error": str(e)}

# --- MAIN LOOP ---
if __name__ == "__main__":
    # 1. OPTIMIZE FOR M3
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"🚀 OPERATION MIDNIGHT: GOLD MASTER EDITION")
    print(f"HARDWARE: M3 Max | WORKERS: {MAX_WORKERS} | DATA: 16 Years (Compressed)")
    
    # 2. LOAD DATA
    print("...Loading Universe...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=5800, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=5800, backtest_mode=True)
    
    print("...Preparing Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 3. APPLY COMPRESSION (The Fix)
    prepared = compress_data(prepared)
    
    # 4. START EVOLUTION
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(GENERATIONS):
        print(f"\n🧬 GENERATION {gen+1} / {GENERATIONS}")
        start_time = time.time()
        
        tasks = [(i, g) for i, g in enumerate(population)]
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(prepared, g_data)) as executor:
            futures = [executor.submit(evaluate_genome, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                # Live stream results to terminal
                if "error" not in res:
                    print(f"   > [G{gen+1}] Trades: {res['trades']} | CAGR: {res['cagr']:.1f}% | DD: {res['dd']:.1f}%")
        
        # Filter failures
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            print("CRITICAL: All strategies failed.")
            break
            
        # Sort and Save
        valid_results.sort(key=lambda x: x["score"], reverse=True)
        winner = valid_results[0]
        
        print(f"🏆 GEN {gen+1} WINNER: CAGR {winner['cagr']:.2f}% | DD {winner['dd']:.2f}% | {winner['genome']}")
        
        # Save persistence
        with open(BEST_GENOME_FILE, "w") as f:
            json.dump(winner["genome"], f, indent=4)
        
        # CSV Log
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{gen+1},{winner['cagr']},{winner['dd']},{winner['trades']},\"{winner['genome']}\"\n")
            
        # Breeding (Elitism)
        next_gen = [r["genome"] for r in valid_results[:10]] # Keep top 10
        while len(next_gen) < POPULATION_SIZE:
            p1 = random.choice(valid_results[:15])["genome"]
            p2 = random.choice(valid_results[:15])["genome"]
            child = crossover(p1, p2)
            if random.random() < 0.3: child = mutate_genome(child)
            next_gen.append(child)
        population = next_gen
        
        print(f"⏱️  Gen Time: {time.time()-start_time:.1f}s")
