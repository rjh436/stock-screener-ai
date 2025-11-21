import sys
import os
import json
import itertools
import concurrent.futures
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.schwab_client import sd
from execution.engine import run_backtest
from data.indices import get_index_symbols
from strategies.generic import GenericStrategy
from optimization.evolution import EvolutionEngine

# Config Path
GEN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json'))

def fetch_sp100_symbols():
    return get_index_symbols("S&P 100")

def fetch_data(symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
    print(f"Fetching data for {len(symbols)} symbols over last {days} days...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50) 
    
    data = {}
    for sym in symbols:
        try:
            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                df = df.set_index("datetime").sort_index()
                data[sym] = df
        except Exception:
            pass
    return data

def calculate_fitness(result):
    """
    NEW FITNESS FUNCTION: Prioritize Stability (Sharpe) over Gambling (Jackpots).
    """
    trades = result.get("trades", 0)
    sharpe = result.get("sharpe", 0) or 0
    cagr = result.get("cagr", 0) or 0
    win_rate = result.get("hit_rate", 0) or 0
    max_dd = abs(result.get("max_drawdown_pct", 0) or 0)

    # 1. Gatekeepers: Must have trades and not blow up account
    if trades < 20: return -1000.0  # Too few trades = statistical noise
    if max_dd > 25.0: return -1000.0 # Unacceptable risk

    # 2. Sharpe is King (Risk-Adjusted Return)
    # A Sharpe of 2.0 is excellent. We weight this heavily.
    sharpe_score = min(sharpe, 3.0) * 40.0 

    # 3. Consistency Bonus (Win Rate)
    # We want > 50% win rate. 
    win_score = 0
    if win_rate > 50:
        win_score = (win_rate - 50) * 1.5 

    # 4. Growth (CAGR)
    # Capped to prevent "lucky" high returns from dominating
    cagr_score = min(cagr * 100, 50) * 0.6

    # 5. Frequency Penalty (Overtrading vs Undertrading)
    penalty = 0
    if trades < 50: penalty = 20
    if trades > 500: penalty = (trades - 500) * 0.1

    final_score = sharpe_score + win_score + cagr_score - penalty
    return final_score

def run_evolution(data_map, global_data):
    print("\n--- Starting System Repair Evolutionary Cycle ---")
    engine = EvolutionEngine()
    
    # Load if exists, otherwise gen new
    if not engine.population:
         engine.generate_initial_population()

    generations = 10
    
    for gen in range(generations):
        print(f"\nEvaluating Generation {engine.generation_count} ({len(engine.population)} strategies)...")
        
        pop_results = []
        # Use fewer workers to prevent API rate limits
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
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

        # Sort by NEW fitness
        ranked = sorted(pop_results, key=lambda x: x["score"], reverse=True)
        
        print(f"Top Gen {engine.generation_count}:")
        for i, r in enumerate(ranked[:3]):
            stats = r['stats']
            print(f"#{i+1} Score: {r['score']:.1f} | Sharpe: {stats.get('sharpe',0):.2f} | WR: {stats.get('hit_rate',0):.1f}% | Trades: {stats.get('total_trades',0)}")

        if gen < generations - 1:
            engine.evolve(ranked)
    
    with open(GEN_CONFIG_PATH, "w") as f:
        # Save top 10 only
        top_genomes = [r["genome"] for r in ranked[:10]]
        json.dump(top_genomes, f, indent=4)
    print("\nEvolution complete. Clean population saved.")

def run_optimization():
    print("🚀 Initializing Optimizer...")
    symbols = fetch_sp100_symbols()
    if not symbols: return
    
    data_map = fetch_data(symbols, days=400)
    if not data_map: return

    # Mock Global Data for now to ensure it runs
    global_data = {"SPY": None, "VIX": None}
    
    run_evolution(data_map, global_data)

if __name__ == "__main__":
    run_optimization()