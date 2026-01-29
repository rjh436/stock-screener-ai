
import sys
import os
import json
import random
import copy
import multiprocessing
import concurrent.futures
import pandas as pd
import numpy as np
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

# --- MIDNIGHT CONFIG ---
POPULATION_SIZE = 50
GENERATIONS = 100
ELITE_SIZE = 5
MUTATION_RATE = 0.3
NUM_WORKERS = max(1, os.cpu_count() - 2) # Leave some room for OS

# --- SEARCH SPACE ---
GENE_RANGES = {
    "rs_floor": [80, 85, 87, 90, 93, 95],
    "ma_exit": ["sma10", "sma20", "sma50"],
    "stop_loss_atr": (2.0, 5.0),
    "regime_ma": ["sma150", "sma200"],
    "adx_threshold": [15, 20, 25, 30],
    "max_positions": [4, 5, 8, 10]
}

def random_gene(name):
    bounds = GENE_RANGES[name]
    if isinstance(bounds, list):
        return random.choice(bounds)
    if isinstance(bounds, tuple):
        return round(random.uniform(bounds[0], bounds[1]), 2)
    return bounds

def create_random_genome():
    return {k: random_gene(k) for k in GENE_RANGES}

def mutate(genome):
    mutant = genome.copy()
    for gene in genome:
        if random.random() < MUTATION_RATE:
            mutant[gene] = random_gene(gene)
    return mutant

def crossover(parent1, parent2):
    child = {}
    for gene in parent1:
        child[gene] = parent1[gene] if random.random() > 0.5 else parent2[gene]
    return child

def genome_to_config(genome):
    config = {
        "name": "Midnight Candidate",
        "use_fundamentals": False,
        "entry_rules": [
             {"col": "rs_rating", "op": ">", "val": genome["rs_floor"]}, 
             {"col": "bb_width", "op": "<", "val": 0.25},
             # Trend Reinforcement default
             {"col": "close", "op": ">", "ref": "sma50"},
             {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99}
        ],
        "risk_parameters": {
             "risk_per_trade": 1.0, # Placeholder, will be autoscaled by max_pos_size constraints in engine? 
             # Wait, user said "NO LEVERAGE". 
             # We should set risk_per_trade so that we don't exceed 1.0 exposure.
             # If max_positions=4, roughly 0.25 size.
             # Ideally let's set risk appropriately or use max_pos_size_pct.
             "stop_loss_type": "atr",
             "stop_loss_atr": genome["stop_loss_atr"],
             "max_positions": genome["max_positions"],
             # Ensure no leverage:
             "max_pos_size_pct": float(1.0 / genome["max_positions"])
        },
        "execution_parameters": {
             "time_stop": 60, # Reasonable default for robustness
             "partial_profit_day": 999,
             "partial_profit_r": 100.0,
             "vol_mult": 1.5, # Fixed for midnight
             "rs_floor": genome["rs_floor"],
             "adx_min": float(genome["adx_threshold"]),
             "use_trailing_stop": True
        },
        "exit_rules": [{"col": "close", "op": "<", "ref": genome["ma_exit"]}]
    }
    return config

# Global storage for worker processes to access read-only data
# This avoids pickling huge dataframes for every task
_prepared_data_ref = None
_g_data_ref = None

def init_worker(prepared_data, g_data):
    global _prepared_data_ref
    global _g_data_ref
    _prepared_data_ref = prepared_data
    _g_data_ref = g_data

def evaluate_genome(genome):
    # Use global data reference
    global _prepared_data_ref
    global _g_data_ref
    
    if _prepared_data_ref is None or _g_data_ref is None:
        return -999.0, 0.0, 0.0, 0, {}

    try:
        config = genome_to_config(genome)
        
        # REGIME MA SWITCHING HACK
        # engine.py expects 'global_spy_sma200'. 
        # If genome wants SMA150, we point 'sma200' in g_data to valid data?
        # Run_backtest extracts from 'global_data["SPY"]'.
        
        # We need to make a localized shallow copy of g_data for this run
        g_data_run = _g_data_ref.copy()
        
        if genome["regime_ma"] == "sma150":
            # Check if we have spy dataframe
            spy_df = g_data_run.get("SPY")
            if spy_df is not None:
                # We need to trick the engine. 
                # The engine looks for 'sma200'.
                # We can swap the columns in a copy of the dataframe.
                spy_run = spy_df.copy()
                if "sma150" in spy_run.columns:
                    spy_run["sma200"] = spy_run["sma150"] # The Swap
                g_data_run["SPY"] = spy_run

        strat = SEPAChampionStrategy(config)
        res = run_backtest(
            strat,
            data=_prepared_data_ref,
            start_cash=100000.0, 
            start_date="2010-01-01",
            end_date="2025-12-31",
            global_data=g_data_run
        )
        
        if not res: return 0.0, 0.0, 0.0, 0, {}
        res = res if isinstance(res, dict) else res[0]
        
        cagr = res.get("cagr", 0.0) * 100
        dd = res.get("max_drawdown_pct", 1.0) * 100
        trades = res.get("total_trades", 0)
        
        # MINERVINI FITNESS
        # Score = CAGR * (1 - (Max_Drawdown / 100))
        # Penalty for low trades
        
        if trades < 50:
            score = 0.0
        else:
            # Survivability factor
            survivability = 1.0 - (dd / 100.0)
            score = cagr * survivability
            
            # Turnover Bonus: If > 500 trades, small boost (e.g., 5%)
            if trades > 500:
                score *= 1.05

        return score, cagr, dd, trades, res
        
    except Exception as e:
        # print(f"Error in worker: {e}")
        return 0.0, 0.0, 0.0, 0, {}


def main():
    print("🌙 Starting OPERATION MIDNIGHT (20-Year Robustness)...")
    print(f"Workers: {NUM_WORKERS}")
    
    # 1. Load Data
    print("📦 Loading DEEP HISTORY (2010-2025)...")
    symbols = get_universe_symbols("RUSSELL3000")
    
    # Approx 16 years (2010-2026). Loader uses 1.6x multiplier for calendar days.
    # 4000 * 1.6 = 6400 days = 17.5 years. Covers 2009+.
    data = fetch_data_pack(symbols, days=4000, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=4000, backtest_mode=True) or {}
    
    # Ensure SPY has SMA150 computed
    if "SPY" in g_data:
        spy = g_data["SPY"]
        spy.columns = spy.columns.str.lower()
        if "sma150" not in spy.columns:
            spy["sma150"] = spy["close"].rolling(150).mean()
        if "sma200" not in spy.columns:
            spy["sma200"] = spy["close"].rolling(200).mean()
    
    print("⚙️  Pre-calculating Indicators (Cache)...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 2. Init Population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_overall_genome = None
    best_overall_score = -9999.0
    
    results_log = []

    # 3. Process Pool
    # We use 'fork' on Linux, 'spawn' on Mac.
    # To pass data to workers, we can stick it in a global or use initializer.
    # Initializer is cleaner for large read-only data.
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=init_worker, initargs=(prepared, g_data)) as executor:
        
        for gen in range(GENERATIONS):
            print(f"\n🧬 Generation {gen+1}/{GENERATIONS}")
            start_time = time.time()
            
            # Map evaluation
            futures = {executor.submit(evaluate_genome, genome): genome for genome in population}
            
            scored_pop = []
            for future in concurrent.futures.as_completed(futures):
                genome = futures[future]
                try:
                    score, cagr, dd, tr, _ = future.result()
                    scored_pop.append((score, genome, cagr, dd, tr))
                except Exception as e:
                    print(f"Worker Error: {e}")
                    scored_pop.append((-999.0, genome, 0.0, 0.0, 0))

            # Sort
            scored_pop.sort(key=lambda x: x[0], reverse=True)
            
            best_gen_score, best_gen_genome, b_cagr, b_dd, b_tr = scored_pop[0]
            elapsed = time.time() - start_time
            print(f"🏆 Gen {gen+1} Winner: Score={best_gen_score:.2f} | CAGR={b_cagr:.1f}% | DD={b_dd:.1f}% | Tr={b_tr} | {elapsed:.1f}s")
            print(f"   DNA: {best_gen_genome}")
            
            # Log
            results_log.append({
                "Gen": gen+1,
                "Score": best_gen_score,
                "CAGR": b_cagr,
                "Drawdown": b_dd,
                "Trades": b_tr,
                "Genome": json.dumps(best_gen_genome)
            })
            pd.DataFrame(results_log).to_csv("midnight_results.csv", index=False)

            if best_gen_score > best_overall_score:
                best_overall_score = best_gen_score
                best_overall_genome = best_gen_genome
                with open("config/midnight_winner.json", "w") as f:
                    json.dump(best_overall_genome, f, indent=2)

            # Evolution
            elites = [x[1] for x in scored_pop[:ELITE_SIZE]]
            next_gen = elites[:]
            
            while len(next_gen) < POPULATION_SIZE:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites) # Or sample from broader population
                child = crossover(parent1, parent2)
                child = mutate(child)
                next_gen.append(child)
            
            population = next_gen

    print("\n🏁 MIDNIGHT RUN COMPLETE.")
    print(f"👑 Ultimate Winner (Score: {best_overall_score:.2f})")
    print(json.dumps(best_overall_genome, indent=2))

if __name__ == "__main__":
    if sys.platform == "darwin":
        multiprocessing.set_start_method("spawn")
    main()
