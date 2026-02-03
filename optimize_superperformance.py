import os
import sys
import json
import random
import time
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import pickle
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- IMPORTS ---
try:
    from execution.engine import prepare_backtest_data, run_backtest
    from data.loader import fetch_data_pack
    from data.universe import get_universe_symbols
    from strategies.superperformance import SuperperformanceStrategy
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RESULTS_FILE = "superperformance_results.csv"
BEST_GENOME_FILE = "config/superperformance_winner.json"
POPULATION_SIZE = 30
GENERATIONS = 6
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
MAX_WORKERS = 6
CHECKPOINT_FILE = "optimizer_checkpoint_sp.pkl" 

# --- GENOME SPACE (Optimization Variables) ---
GENE_SPACE = {
    # Optional trend filter (use sparingly)
    "require_trend": [False, True],
    "rs_min": [0, 70, 80, 90],
    "mom_rank_min": [0, 80, 90, 95],
    "use_market_filter": [True, False],

    # Qullamaggie Tightness + Breakout
    "natr_max": [2.0, 2.5, 3.0, 4.0, 5.0],
    "natr_days": [10],
    "vol_mult": [1.0, 1.2, 1.5, 2.0],
    "breakout_buffer": [0.0, 0.002, 0.005],

    # Episodic Pivot (High Volume Gap)
    "entry_mode": ["breakout", "ep", "both"],
    "ep_gap_pct": [0.04, 0.06, 0.08, 0.10],
    "ep_vol_mult": [1.3, 1.5, 2.0],

    # Execution / Risk
    "max_stop_pct": [0.05],
    "stop_limit_pct": [0.05, 0.08, 0.10, 0.12],

    # Exit Discipline
    "breakeven_at_pct": [0.08, 0.10, 0.12],
    "trail_ma": ["sma10", "sma20"],

    # Sizing
    "max_positions": [3, 4, 5],
    "risk_per_trade": [0.05, 0.08, 0.10, 0.15],
    "max_pos_size_pct": [0.40, 0.60, 0.80, 1.00],
}

# --- DATA REF ---
_prepared_data_ref = None
_g_data_ref = None

def save_checkpoint(generation, population, best_genome_so_far):
    try:
        checkpoint_data = {
            "generation": generation,
            "population": population,
            "best_genome": best_genome_so_far
        }
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"💾 Checkpoint saved for Generation {generation}")
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                checkpoint_data = pickle.load(f)
            print(f"🚀 RESUMING from Generation {checkpoint_data['generation'] + 1}...")
            return checkpoint_data["generation"], checkpoint_data["population"], checkpoint_data.get("best_genome")
        except Exception as e:
            print(f"⚠️ Checkpoint found but failed to load: {e}")
            return None
    return None

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
        # Construct Strategy Config
        strategy_config = genome.copy()
        strategy_config["name"] = f"Gen_{genome_id}"
        # Align exit SMA with trail SMA for discipline
        if "trail_ma" in strategy_config:
            strategy_config["exit_ma"] = strategy_config["trail_ma"]
        # After-close scan, next-day market entry (MOO) improves fill realism
        strategy_config["signal_mode"] = "open"
        if strategy_config.get("use_market_filter"):
            strategy_config["market_filter_mode"] = "spy_sma200"
        strategy_config["score_mode"] = "momentum"
        
        # Instantiate Specific Strategy
        strat = SuperperformanceStrategy(strategy_config)
        
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
        
        final_val = metrics.get("final_value", 100000)
        trades = metrics.get("total_trades", 0)
        raw_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0)))
        max_dd = raw_dd * 100.0 if raw_dd < 1.0 else raw_dd
            
        # CAGR
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        years = max(years, 1.0)
        cagr_pct = ((final_val / 100000.0) ** (1/years) - 1) * 100
        
        # FITNESS FUNCTION: Minervini/Qullamaggie
        # Prioritize CAGR, Penalize Drawdown > 25%
        if cagr_pct > 0:
            score = (cagr_pct ** 1.5) - (max_dd * 0.5)
        else:
            score = cagr_pct - max_dd
            
        if max_dd > 20.0:
            score -= (max_dd - 20.0) * 12.0  # stricter penalty toward <20% DD

        return {
            "id": genome_id,
            "genome": genome,
            "score": score,
            "cagr": cagr_pct,
            "dd": max_dd,
            "trades": trades
        }
    except Exception as e:
        return {"id": genome_id, "score": -999, "error": str(e)}
    finally:
        gc.collect()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError: pass

    print(f"🚀 PROJECT APEX: Strategic Nuclear Reset")
    print(f"HARDWARE: M3 Max | WORKERS: {MAX_WORKERS}")
    
    print("...Loading Data...")
    symbols = get_universe_symbols("RUSSELL3000")
    total_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
    trading_days = int((total_days / 365.25) * 252) + 400  # warmup buffer
    data = fetch_data_pack(symbols, days=trading_days, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=trading_days, backtest_mode=True)
    
    print("...Preparing & Compressing...")
    # Strict Prep Config
    PREP_CONFIG = {
        "parameters": {
            "min_rs": 60,           
            "adx_threshold": 10,  
            "vol_ma_ratio": 0.75,
            "bb_width_threshold": 0.40 
        }
    }
    prepared = prepare_backtest_data(data, symbols, start_date=START_DATE, global_data=g_data)
    print(f"📊 DATA POOL: {len(prepared.enriched)} tickers prepared.")
    
    prepared = compress_data(prepared)

    checkpoint = None # START FRESH
    start_gen = 0
    
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    
    for gen in range(start_gen, GENERATIONS):
        print(f"\n🧬 GEN {gen+1}/{GENERATIONS} (Superperformance)")
        start_time = time.time()
        
        tasks = [(i, g) for i, g in enumerate(population)]
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=init_worker, initargs=(prepared, g_data)) as executor:
            futures = [executor.submit(evaluate_genome, t) for t in tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                results.append(res)
                if "error" not in res:
                    print(f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | DD:{res['dd']:.1f}%")
                else:
                    print(f"   ⚠️  GENOME {res['id']} FAILED: {res['error']}")
        
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
        next_gen = [r["genome"] for r in valid[:5]] # Keep Elites
        scores = [r["score"] for r in valid]
        mutation_rate = 0.5 # High initial mutation
        
        while len(next_gen) < POPULATION_SIZE:
             p1 = random.choice(valid[:15])["genome"]
             p2 = random.choice(valid[:15])["genome"]
             child = crossover(p1, p2)
             if random.random() < mutation_rate:
                 child = mutate_genome(child)
             next_gen.append(child)
             
        population = next_gen
        save_checkpoint(gen, population, winner)
        gc.collect()
