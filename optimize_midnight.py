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
    "rs_floor": [85, 87, 90, 93, 95],         # High Quality
    "vol_mult": [1.0, 1.25, 1.5, 2.0],        
    "adx_min": [10, 12, 15, 20, 25],          
    "bb_width_max": [0.20, 0.25, 0.30, 0.35], 
    
    # TIMING
    # TIMING
    "regime_ma": ["sma150", "sma200"],        
    "trend_mode": ["sma50", "sma200", "strict"], # NEW: Dynamic Trend Definition        
    
    # EXIT MECHANICS
    "exit_sma": ["sma50"],                    # Loose Hold for Runners
    "stop_loss_atr": [1.5, 1.75, 2.0, 2.25],  # Tight Risk
    
    # PROFIT TAKING (The Control Switch)
    "enable_partial_profit": [True],          # Force Hybrid Model
    "partial_profit_r": [2.0, 2.25, 2.5, 3.0],# Achievable Targets
    "move_stop_to_be": [True, False],         # Let AI decide on Breakeven
    "partial_profit_day": [0, 3, 5],          # Minimal time gating
    
    # SIZING: Forced Concentration
    "max_positions": [4, 5, 6],
    "risk_per_trade": [0.02, 0.025, 0.03]  
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
        # Construct Config - FLATTENED for Engine Compatibility
        pos_size = 1.0 / genome["max_positions"]
        
        # Merge genome directly into config so engine finds keys like 'rs_floor' at top level
        strategy_config = genome.copy()
        strategy_config.update({
            "name": f"Gen_{genome_id}",
            # Inject Rules
            "entry_rules": [
                {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99},
                {"col": "close", "op": ">", "ref": "sma50"}
            ],
            # Helper sub-dicts (Engine flatinizes risk_parameters, not risk_management)
            "risk_parameters": {
                "stop_loss_atr": genome["stop_loss_atr"],
                "max_positions": genome["max_positions"],
                "risk_per_trade": genome["risk_per_trade"],
                "max_pos_size_pct": pos_size
            },
            "execution_parameters": {
                "exit_sma": genome["exit_sma"],
                "enable_partial_profit": genome["enable_partial_profit"],
                "move_stop_to_be": genome["move_stop_to_be"],
                "partial_profit_r": genome["partial_profit_r"],
                "partial_profit_r": genome["partial_profit_r"],
                "partial_profit_day": int(genome["partial_profit_day"]),
                "trend_mode": genome["trend_mode"]
            }
        })
        
        strategies = load_strategies([strategy_config])
        strat = strategies[0]
        # Force attributes (Legacy support)
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
        
        # --- NEW FITNESS FUNCTION (Smart Scoring) ---
        # Formula: (CAGR ^ 1.5) - (DD * 0.5)
        # Soft Penalty: If DD > 25.0, score -= (DD - 25.0) * 50.0
        
        if cagr_pct > 0:
            score = (cagr_pct ** 1.5) - (max_dd * 0.5)
        else:
            score = cagr_pct - max_dd # Linear punishment for losers
            
        if max_dd > 25.0:
            penalty = (max_dd - 25.0) * 50.0
            score -= penalty

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
    # Define Looser Config for Data Prep to ensure we get candidates
    WIDE_NET_CONFIG = {
        "parameters": {
            "min_rs": 60,           # Catch falling stars / recovery plays (Updated per instructions)
            "adx_threshold": 10,    # Allow chopping stocks
            "vol_ma_ratio": 1.0,    # Allow normal volume
            "bb_width_threshold": 0.40  # Allow loose expansions
        }
    }
    prepared = prepare_backtest_data(data, symbols, start_date=START_DATE, global_data=g_data)
    print(f"📊 DATA POOL: {len(prepared.enriched)} tickers passed Wide Net filter.")
    
    # --- DATA PATCHING LOOP ---
    print("🔧 MANUAL PATCH: Calculating 'high_20_prev' and 'atr' for all symbols...")
    # --- DATA PATCHING LOOP ---
    for sym, s_data in prepared.enriched.items():
        df = s_data.df
        # Calculate 20-day high shifted by 1 (required for Breakout Entry)
        if 'high' in df.columns:
            df['high_20_prev'] = df['high'].rolling(window=20).max().shift(1)
        
        # Calculate ATR if missing (required for Stop Loss)
        if 'atr' not in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            # User requested TR calculation using np.maximum
            df['tr'] = np.maximum(
                df['high'] - df['low'], 
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
    
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
        # Breeding with DYNAMIC MUTATION & FRESH BLOOD
        next_gen = [r["genome"] for r in valid[:5]] # Keep top 5 elites
        
        # 1. Calculate Population Variance (using Score Standard Deviation)
        scores = [r["score"] for r in valid]
        score_std = np.std(scores)
        
        # 2. Adaptive Mutation Rate
        # If variance is low (< 5.0), the population is stagnant -> PANIC MODE (80% mutation)
        # If variance is healthy, use normal exploration (20% mutation)
        mutation_rate = 0.8 if score_std < 5.0 else 0.2
        if gen > 0:
            print(f"🧬 DIV: Score StdDev: {score_std:.2f} | Mut Rate: {int(mutation_rate*100)}%")
        
        # 3. Inject Fresh Blood (20% of pop)
        fresh_blood_count = int(POPULATION_SIZE * 0.2)
        for _ in range(fresh_blood_count):
            next_gen.append(generate_random_genome())
            
        # 4. Fill remainder with Crossover children
        while len(next_gen) < POPULATION_SIZE:
            # Tournament selection (randomly pick 15, take best) - simplified here to top 15 list cache
            p1 = random.choice(valid[:15])["genome"]
            p2 = random.choice(valid[:15])["genome"]
            child = crossover(p1, p2)
            
            if random.random() < mutation_rate:
                child = mutate_genome(child)
            
            next_gen.append(child)
        population = next_gen
        print(f"⏱️  {time.time()-start_time:.1f}s")
