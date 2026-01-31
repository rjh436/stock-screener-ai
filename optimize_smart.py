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
RESULTS_FILE = "smart_results.csv"
BEST_GENOME_FILE = "config/smart_winner.json"
POPULATION_SIZE = 40 
GENERATIONS = 50
START_DATE = "2010-01-01"
END_DATE = "2025-12-31"
MAX_WORKERS = 6 

# --- THE "SMART" SEARCH SPACE ---
GENE_SPACE = {
    # SELECTION: Looser filters to get more "At Bats"
    "rs_floor": [85, 87, 90, 93, 95],         # High Quality
    "vol_mult": [1.0, 1.25, 1.5, 2.0],        
    "adx_min": [10, 12, 15, 20, 25],          
    "bb_width_max": [0.20, 0.25, 0.30, 0.35], 
    
    # TIMING
    "regime_ma": ["sma150", "sma200"],        
    "trend_mode": ["strict", "sma200", "sma50"], # NEW: Dynamic Trend Definition        
    
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
        
        # --- EXTRACT METRICS ---
        final_val = metrics.get("final_value", 100000)
        trades = metrics.get("total_trades", 0)
        
        # Drawdown Normalization
        raw_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0)))
        if raw_dd < 1.0 and raw_dd > 0.0:
            max_dd = raw_dd * 100.0
        else:
            max_dd = raw_dd
            
        # CAGR Logic
        years = 16 
        cagr_pct = ((final_val / 100000.0) ** (1/years) - 1) * 100
        
        # --- SMART FITNESS FUNCTION: Expectancy * CAGR ---
        # 1. Calculate Expectancy
        trades_list = metrics.get("trades_list", [])
        avg_win_pct, avg_loss_pct, expectancy = 0.0, 0.0, 0.0
        win_rate = 0.0
        
        if trades_list:
            df_t = pd.DataFrame(trades_list)
            # Find return column (prefer percent)
            col = None
            for c in ["return_pct", "ReturnPct", "PnL_Pct"]:
                if c in df_t.columns:
                    col = c
                    break
            
            # Fallback to calculating from PnL / Basis? 
            # If no % column, we use PnL (which is dollars). 
            # But we prefer % for generalized expectancy.
            # If PnL is present but not %, we might need to rely on PnL.
            if col:
                # Assuming % like 5.0 for 5%
                wins = df_t[df_t[col] > 0][col]
                losses = df_t[df_t[col] <= 0][col]
                
                avg_win_pct = wins.mean() if not wins.empty else 0.0
                avg_loss_pct = abs(losses.mean()) if not losses.empty else 0.0
                
                win_rate = (len(wins) / len(df_t))
                loss_rate = (1.0 - win_rate)
                
                # Expectancy (e.g. 0.5 * 10% - 0.5 * 5% = 2.5% per trade)
                expectancy = (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct)
            else:
                # If we cannot find percent return, use PnL / 1000 approx or just PnL?
                # Let's try PnL and assume it's roughly proportional
                if "PnL" in df_t.columns:
                    wins = df_t[df_t["PnL"] > 0]["PnL"]
                    losses = df_t[df_t["PnL"] <= 0]["PnL"]
                    avg_win = wins.mean() if not wins.empty else 0.0
                    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
                    win_rate = (len(wins) / len(df_t))
                    loss_rate = (1.0 - win_rate)
                    # Dollar Expectancy 
                    expectancy_curr = (win_rate * avg_win) - (loss_rate * avg_loss)
                    # Normalize roughly to a "score" unit (e.g. div by 100?)
                    expectancy = expectancy_curr / 100.0 
        
        # Score Calculation
        if cagr_pct > 0 and expectancy > 0:
            score = cagr_pct * expectancy
        else:
            # Penalize
            score = cagr_pct - max_dd - (abs(expectancy) * 10)
            
        # Drawdown Penalty
        if max_dd > 25.0:
            score -= (max_dd - 25.0) * 10.0

        return {
            "id": genome_id,
            "genome": genome,
            "score": score,
            "cagr": cagr_pct,
            "dd": max_dd,
            "trades": trades,
            "exp": expectancy,
            "win_rate": win_rate * 100,
            "raw_metrics_keys": list(metrics.keys())
        }

    except Exception as e:
        return {"id": genome_id, "score": -999, "error": str(e)}

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError: pass

    print(f"🧠 OPTIMIZE SMART: Trending & Expectancy")
    print(f"HARDWARE: M3 Max | WORKERS: {MAX_WORKERS}")
    
    print("...Loading Data...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=5800, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=5800, backtest_mode=True)
    
    print("...Preparing & Compressing...")
    # Define Looser Config for Data Prep to ensure we get candidates
    WIDE_NET_CONFIG = {
        "parameters": {
            "min_rs": 60,           # Catch falling stars / recovery plays
            "adx_threshold": 10,    # Allow chopping stocks
            "vol_ma_ratio": 1.0,    # Allow normal volume
            "bb_width_threshold": 0.40  # Allow loose expansions
        }
    }
    prepared = prepare_backtest_data(data, symbols, start_date=START_DATE, global_data=g_data)
    print(f"📊 DATA POOL: {len(prepared.enriched)} tickers passed Wide Net filter.")
    
    # --- DATA PATCHING LOOP ---
    print("🔧 MANUAL PATCH: Calculating 'high_20_prev' and 'atr' for all symbols...")
    for sym, s_data in prepared.enriched.items():
        df = s_data.df
        if 'high' in df.columns:
            df['high_20_prev'] = df['high'].rolling(window=20).max().shift(1)
        
        if 'atr' not in df.columns and 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'], 
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)), 
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(window=14).mean()
    
    prepared = compress_data(prepared)
    
    # --- POPULATION SEEDING ---
    population = []
    
    # "Classic Minervini" Seed (The Head Start)
    SEED_TEMPLATE = {
        "rs_floor": 90,
        "vol_mult": 1.25,
        "adx_min": 20,
        "bb_width_max": 0.25,
        "regime_ma": "sma200",
        "trend_mode": "strict", # The Key
        "exit_sma": "sma50",
        "stop_loss_atr": 2.0,
        "enable_partial_profit": True,
        "partial_profit_r": 2.0,
        "move_stop_to_be": True,
        "partial_profit_day": 3,
        "max_positions": 5,
        "risk_per_trade": 0.025
    }
    
    # Create 5 variations of the seed
    for _ in range(5):
        if _ == 0:
            population.append(SEED_TEMPLATE.copy()) # The exact seed
        else:
            # Slight drift
            var = mutate_genome(SEED_TEMPLATE)
            population.append(var)
            
    # Fill the rest with random
    while len(population) < POPULATION_SIZE:
        population.append(generate_random_genome())
        
    print(f"🌱 Population Init: 5/5 Seeds Planted. Total: {len(population)}")
    
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
                    print(f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | EXP:{res['exp']:.2f} | Score:{res['score']:.1f}")
        
        valid = [r for r in results if "error" not in r]
        if not valid:
            print("CRITICAL: All failed.")
            break
            
        valid.sort(key=lambda x: x["score"], reverse=True)
        winner = valid[0]
        
        print(f"🏆 WINNER: CAGR {winner['cagr']:.2f}% | Exp {winner['exp']:.2f} | Score: {winner['score']:.1f}")
        print(f"🧬 DNA: {winner['genome']}")
        
        with open(BEST_GENOME_FILE, "w") as f:
            json.dump(winner["genome"], f, indent=4)
        with open(RESULTS_FILE, "a") as f:
            f.write(f"{gen+1},{winner['cagr']},{winner['dd']},{winner['trades']},{winner['exp']},\"{winner['genome']}\"\n")
            
        # Breeding
        next_gen = [r["genome"] for r in valid[:5]] # Keep top 5 elites
        
        # 1. Calculate Population Variance
        scores = [r["score"] for r in valid]
        score_std = np.std(scores)
        
        # 2. Adaptive Mutation Rate
        mutation_rate = 0.8 if score_std < 5.0 else 0.2
        if gen > 0:
            print(f"🧬 DIV: Score StdDev: {score_std:.2f} | Mut Rate: {int(mutation_rate*100)}%")
        
        # 3. Inject Fresh Blood (20% of pop)
        fresh_blood_count = int(POPULATION_SIZE * 0.2)
        for _ in range(fresh_blood_count):
            next_gen.append(generate_random_genome())
            
        # 4. Fill remainder with Crossover
        while len(next_gen) < POPULATION_SIZE:
            p1 = random.choice(valid[:15])["genome"]
            p2 = random.choice(valid[:15])["genome"]
            child = crossover(p1, p2)
            
            if random.random() < mutation_rate:
                child = mutate_genome(child)
            
            next_gen.append(child)
        population = next_gen
        print(f"⏱️  {time.time()-start_time:.1f}s")
