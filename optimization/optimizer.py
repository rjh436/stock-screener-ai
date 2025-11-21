import sys
import os
import json
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.schwab_client import sd
from data.cache_manager import DataCache
from execution.engine import run_backtest
from data.indices import get_index_symbols
from strategies.generic import GenericStrategy
from optimization.evolution import EvolutionEngine

# Config Path
GEN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json'))

def fetch_sp1500_symbols():
    return get_index_symbols("S&P 1500")

def fetch_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    print(f"Fetching data for {len(symbols)} symbols over last {days} days...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50) 
    
    data = {}
    cache_hits = 0
    cache_misses = 0
    
    for sym in symbols:
        try:
            # Check cache first
            df = DataCache.get_cached_data(sym)
            
            if df is not None:
                # Filter to requested date range
                df = df[(df.index >= start) & (df.index <= end)]
                if len(df) >= days * 0.7:
                    data[sym] = df
                    cache_hits += 1
                    continue
            
            # Cache miss - fetch
            cache_misses += 1
            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                df = df.set_index("datetime").sort_index()
                
                # Save to cache
                DataCache.save_to_cache(sym, df)
                data[sym] = df
        except Exception as e:
            print(f"Error fetching {sym}: {e}")
    
    print(f"Cache: {cache_hits} hits, {cache_misses} misses")
    return data

def calculate_fitness(result):
    trades = result.get("trades", 0)
    sharpe = result.get("sharpe", 0) or 0
    cagr = result.get("cagr", 0) or 0
    win_rate = result.get("hit_rate", 0) or 0
    max_dd = abs(result.get("max_drawdown_pct", 0) or 0)
    avg_profit_pct = result.get("avg_profit_pct", 0) or 0

    # Gatekeepers (Relaxed for Swing Mode)
    if trades < 10: return -1000.0 
    if max_dd > 50.0: return -1000.0 
    if avg_profit_pct < 1.0: return -500.0

    # Scoring
    sharpe_score = min(sharpe, 3.0) * 40.0 
    win_score = 0
    if win_rate > 50:
        win_score = (win_rate - 50) * 1.5 
    cagr_score = min(cagr * 100, 50) * 0.6
    
    penalty = 0
    if trades < 50: penalty = 20
    if trades > 500: penalty = (trades - 500) * 0.1

    return sharpe_score + win_score + cagr_score - penalty

def run_evolution(data_map, global_data):
    print("\n--- Starting System Repair Evolutionary Cycle ---")
    engine = EvolutionEngine()
    
    if not engine.population:
         engine.generate_initial_population()

    generations = 3
    
    for gen in range(generations):
        print(f"\nEvaluating Generation {engine.generation_count} ({len(engine.population)} strategies)...")
        
        pop_results = []
        # Use ThreadPoolExecutor for Shared Memory (M3 Max Optimization)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(run_backtest, GenericStrategy(genome), data_map, None, 100000.0, None, global_data): genome for genome in engine.population}
            
            for future in concurrent.futures.as_completed(futures):
                genome = futures[future]
                try:
                    res = future.result()
                    score = calculate_fitness(res)
                    pop_results.append({
                        "genome": genome,
                        "score": score,
                        "stats": res
                    })
                except Exception as e:
                    print(f"Error: {e}")

        ranked = sorted(pop_results, key=lambda x: x["score"], reverse=True)
        
        print(f"Top Gen {engine.generation_count}:")
        for i, r in enumerate(ranked[:3]):
            stats = r['stats']
            print(f"#{i+1} Score: {r['score']:.1f} | Sharpe: {stats.get('sharpe',0):.2f} | CAGR: {stats.get('cagr',0)*100:.1f}% | AvgProfit: {stats.get('avg_profit_pct',0):.2f}% | WR: {stats.get('hit_rate',0):.1f}% | Trades: {stats.get('total_trades',0)}")

        if gen < generations - 1:
            engine.evolve(ranked)
    
    with open(GEN_CONFIG_PATH, "w") as f:
        # Save top 10 only
        top_genomes = [r["genome"] for r in ranked[:10]]
        json.dump(top_genomes, f, indent=4)
    print("\nEvolution complete. Clean population saved.")

def run_optimization():
    print("🚀 Initializing Optimizer (S&P 1500 Universe)...")
    symbols = fetch_sp1500_symbols()
    if not symbols: return
    
    data_map = fetch_data(symbols, days=1260) # 5 Years History
    if not data_map: return

    # Global Data
    print("Fetching Global Data (SPY, VIX)...")
    global_data_raw = fetch_data(["SPY", "$VIX", "VIX"], days=1260)
    
    vix_data = global_data_raw.get("$VIX")
    if vix_data is None: vix_data = global_data_raw.get("VIX")

    global_data = {
        "SPY": global_data_raw.get("SPY"),
        "VIX": vix_data
    }
    
    run_evolution(data_map, global_data)

if __name__ == "__main__":
    run_optimization()