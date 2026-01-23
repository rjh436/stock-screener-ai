import sys
import os
import random
import json
import copy
import concurrent.futures
import multiprocessing as mp
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# --- MACOS M3 SAFETY BLOCK ---
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Add root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from execution.engine import run_backtest, prepare_backtest_data
    # from data.loader import fetch_data_pack  <--- BYPASSED
except ImportError:
    print("❌ IMPORT ERROR: Could not find 'execution.engine'.")
    sys.exit(1)

# --- CONFIGURATION ---
POPULATION_SIZE = 50
GENERATIONS = 10
WORKERS = 4 

GENE_RANGES = {
    "gap_pct": (0.0, 3.0, 0.1),
    "volume_mult": (1.0, 3.0, 0.1),
    "adr_pct": (2.0, 5.0, 0.25),
    "stop_loss_atr": (1.5, 4.0, 0.25),
    "rs_rating": (80, 95, 1)
}

BASE_STRATEGY = {
    "name": "Apex GA Candidate",
    "type": "breakout",
    "entry_rules": [
        {"col": "clv", "op": ">", "val": 0.60},
        {"col": "close", "op": ">", "ref": "sma20"}
    ],
    "exit_rules": [
        {"col": "close", "op": "<", "ref": "sma20"}
    ],
    "risk_parameters": {
        "stop_loss_type": "atr",
        "risk_per_trade": 0.02,
        "max_positions": 8,
        "max_pos_size_pct": 0.25
    },
    "execution_parameters": {
        "time_stop": 20,
        "partial_profit_day": 3
    }
}

# --- YAHOO DATA FETCHER (THE FIX) ---
def fetch_data_yfinance(tickers, days=7000):
    """
    Fetches data from Yahoo Finance and formats it exactly how the Engine expects.
    Bypasses Schwab auth entirely.
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    print(f"   ⬇️ Downloading {len(tickers)} symbols via Yahoo Finance...")
    
    data_pack = {}
    
    # Download in bulk
    raw_data = yf.download(tickers, start=start_date, group_by='ticker', auto_adjust=True, progress=False, threads=True)
    
    for ticker in tickers:
        try:
            # Handle single ticker vs multi-ticker structure
            if len(tickers) == 1:
                df = raw_data.copy()
            else:
                df = raw_data[ticker].copy()
            
            if df.empty:
                continue
                
            # Rename columns to lowercase for Engine compatibility
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            
            # Drop missing
            df.dropna(inplace=True)
            
            data_pack[ticker] = df
        except Exception as e:
            pass # Skip bad tickers
            
    print(f"   ✅ Successfully loaded {len(data_pack)} symbols.")
    return data_pack

# --- GENETIC ALGORITHM FUNCTIONS ---

def generate_random_genome():
    genome = {}
    for key, (min_val, max_val, step) in GENE_RANGES.items():
        if isinstance(step, int):
            val = random.randrange(min_val, max_val + step, step)
        else:
            steps = int((max_val - min_val) / step)
            val = min_val + (random.randint(0, steps) * step)
            val = round(val, 2)
        genome[key] = val
    return genome

def genome_to_strategy(genome):
    strat = copy.deepcopy(BASE_STRATEGY)
    strat["name"] = f"GA_G{genome['gap_pct']}_V{genome['volume_mult']}_A{genome['adr_pct']}"
    
    if genome['gap_pct'] > 0:
        strat["entry_rules"].insert(0, {"col": "gap_pct", "op": ">", "val": genome['gap_pct']})
    
    strat["entry_rules"].append({"col": "volume", "op": ">", "ref": "vol_ma20", "mult": genome['volume_mult']})
    strat["entry_rules"].append({"col": "adr_pct", "op": ">", "val": genome['adr_pct']})
    strat["entry_rules"].append({"col": "rs_rating", "op": ">", "val": int(genome['rs_rating'])})
    strat["risk_parameters"]["stop_loss_atr"] = genome['stop_loss_atr']
    
    return strat

def evaluate_candidate(genome, prepared_data, global_data):
    strat = genome_to_strategy(genome)
    
    # SLICE A: SURVIVAL (2008)
    res_2008 = run_backtest([strat], prepared_data, start_cash=100000, start_date="2008-01-01", end_date="2008-12-31", global_data=global_data)
    dd_2008 = res_2008[0]['max_drawdown_pct'] * 100
    
    if dd_2008 < -35.0: return -1000.0, {} 
        
    # SLICE B: PATIENCE (2015)
    res_2015 = run_backtest([strat], prepared_data, start_cash=100000, start_date="2015-01-01", end_date="2015-12-31", global_data=global_data)
    trades_2015 = res_2015[0]['total_trades']
    
    if trades_2015 < 5: return -500.0, {} 
        
    # SLICE C: ALPHA (2020)
    res_2020 = run_backtest([strat], prepared_data, start_cash=100000, start_date="2020-01-01", end_date="2020-12-31", global_data=global_data)
    ret_2020 = ((res_2020[0]['final_value'] - 100000) / 100000) * 100
    
    score = ret_2020 - (abs(dd_2008) * 2.5) + (trades_2015 * 0.5)
    metrics = {"2020_Ret": ret_2020, "2008_DD": dd_2008, "2015_Trades": trades_2015}
    return score, metrics

def mutate(genome):
    new_genome = genome.copy()
    gene = random.choice(list(GENE_RANGES.keys()))
    min_v, max_v, step = GENE_RANGES[gene]
    current = new_genome[gene]
    shift = random.choice([-step, step])
    new_val = current + shift
    if new_val < min_v: new_val = min_v
    if new_val > max_v: new_val = max_v
    new_genome[gene] = round(new_val, 2)
    return new_genome

def crossover(parent1, parent2):
    child = {}
    for gene in GENE_RANGES.keys():
        child[gene] = random.choice([parent1[gene], parent2[gene]])
    return child

def main():
    print("🚀 STARTING GENETIC ALGORITHM OPTIMIZATION (YAHOO BYPASS MODE)")
    print(f"Population: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    
    # 1. Fetch Data via Yahoo (Bypass Schwab)
    universe = ["NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "ENPH", 
                "SEDG", "SHOP", "SQ", "ROKU", "TDOC", "ZM", "PTON", "DKNG", "NET", "CRWD"]
    
    g_data = fetch_data_yfinance(["SPY", "^VIX"], days=7000)
    # Rename VIX key if needed
    if "^VIX" in g_data: g_data["VIX"] = g_data.pop("^VIX")
    
    data = fetch_data_yfinance(universe, days=7000)
    
    print("⚙️  Preparing Backtest Data...")
    prepared = prepare_backtest_data(data, symbol_universe=universe, global_data=g_data)
    
    # 2. Genetic Loop
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    best_genome = None
    best_score = -99999
    
    for gen in range(GENERATIONS):
        print(f"\\n🧬 GENERATION {gen + 1}/{GENERATIONS}")
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(evaluate_candidate, genome, prepared, g_data): genome for genome in population}
            for future in concurrent.futures.as_completed(futures):
                try:
                    score, metrics = future.result()
                    results.append((futures[future], score, metrics))
                except Exception as e:
                    pass

        if not results:
            print("❌ Generation failed. Retrying...")
            continue
            
        results.sort(key=lambda x: x[1], reverse=True)
        top_genome, top_score, top_metrics = results[0]
        
        print(f"   🏆 Best Score: {top_score:.2f} | 2020 Ret: {top_metrics['2020_Ret']:.1f}% | 2008 DD: {top_metrics['2008_DD']:.1f}%")
        print(f"      Params: {top_genome}")

        if top_score > best_score:
            best_score = top_score
            best_genome = top_genome
            with open("config/best_genome.json", "w") as f:
                json.dump(best_genome, f, indent=4)
        
        # Breed
        survivors = [r[0] for r in results[:10]]
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            child = crossover(random.choice(survivors), random.choice(survivors))
            if random.random() < 0.2: child = mutate(child)
            new_pop.append(child)
        population = new_pop

    print("\\n🏁 OPTIMIZATION COMPLETE. Best Genome saved to config/best_genome.json")

if __name__ == "__main__":
    main()
