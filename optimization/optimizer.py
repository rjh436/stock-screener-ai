import sys
import os
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.schwab_client import sd
from data.cache_manager import DataCache
from execution.engine import run_backtest
from data.indices import get_index_symbols
from strategies.generic import GenericStrategy
from optimization.evolution import EvolutionEngine

GEN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json'))
METRICS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../latest_metrics.json'))

def fetch_sp1500_symbols():
    return get_index_symbols("S&P 1500")

def fetch_data(symbols: List[str], days: int = 1260) -> Dict[str, pd.DataFrame]:
    print(f"Fetching data for {len(symbols)} symbols over last {days} days...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50) 
    data = {}
    
    def load_symbol(sym):
        try:
            df = DataCache.get_cached_data(sym)
            if df is not None:
                if len(df) >= days * 0.9:
                    df = df[(df.index >= start) & (df.index <= end)]
                    return sym, df
            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                df = df.set_index("datetime").sort_index()
                DataCache.save_to_cache(sym, df)
                return sym, df
        except: pass
        return sym, None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(load_symbol, sym): sym for sym in symbols}
        for future in concurrent.futures.as_completed(futures):
            sym, df = future.result()
            if df is not None: data[sym] = df
    return data

def calculate_fitness(result):
    trades = result.get("total_trades", 0)
    sharpe = result.get("sharpe", 0) or 0
    win_rate = result.get("hit_rate", 0) or 0
    avg_profit_pct = result.get("avg_profit_pct", 0) or 0
    profit_factor = result.get("profit_factor", 0) or 0
    payoff = result.get("payoff_ratio", 0) or 0

    if trades < 20: return -1000.0 
    
    # Sniper Score: Heavy on Quality (Payoff/Profit Factor)
    score = (avg_profit_pct * 40.0) + (min(sharpe, 3.0) * 30.0) + (profit_factor * 20.0)
    if payoff > 2.0: score += 50.0 # Bonus for Sniper Payoff
    
    return score

def run_evolution(data_map, global_data):
    print("\n--- Starting Deep Metric Evolution ---")
    engine = EvolutionEngine()
    try:
        with open(GEN_CONFIG_PATH, "r") as f: engine.population = json.load(f)
    except: engine.generate_initial_population()

    generations = 5
    best_stats = {}

    for gen in range(generations):
        print(f"\nEvaluating Gen {engine.generation_count}...")
        pop_results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(run_backtest, GenericStrategy(genome), data_map, None, 100000.0, None, global_data): genome for genome in engine.population}
            for future in concurrent.futures.as_completed(futures):
                genome = futures[future]
                try:
                    res = future.result()
                    score = calculate_fitness(res)
                    pop_results.append({"genome": genome, "score": score, "stats": res})
                except: pass

        ranked = sorted(pop_results, key=lambda x: x["score"], reverse=True)
        best_stats = ranked[0]['stats']
        
        print(f"Top Gen {engine.generation_count}: Profit {best_stats.get('avg_profit_pct',0):.2f}% | PF {best_stats.get('profit_factor',0):.2f} | Payoff {best_stats.get('payoff_ratio',0):.2f} | WR {best_stats.get('hit_rate',0):.1f}% | Hold {best_stats.get('avg_days_held',0):.1f}d")

        if gen < generations - 1: engine.evolve(ranked)
    
    with open(GEN_CONFIG_PATH, "w") as f:
        json.dump([r["genome"] for r in ranked[:10]], f, indent=4)
    
    telemetry = {
        "cagr": best_stats.get("cagr", 0),
        "avg_profit": best_stats.get("avg_profit_pct", 0),
        "win_rate": best_stats.get("hit_rate", 0),
        "hold_days": best_stats.get("avg_days_held", 0),
        "trades": best_stats.get("total_trades", 0),
        "profit_factor": best_stats.get("profit_factor", 0),
        "payoff_ratio": best_stats.get("payoff_ratio", 0)
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(telemetry, f)
    print("\nCycle complete. Telemetry saved.")

def run_optimization():
    symbols = fetch_sp1500_symbols()
    if not symbols: return
    data_map = fetch_data(symbols, days=1260)
    global_data_raw = fetch_data(["SPY", "$VIX", "VIX"], days=1260)
    vix_data = global_data_raw.get("$VIX")
    if vix_data is None:
        vix_data = global_data_raw.get("VIX")
    run_evolution(data_map, {"SPY": global_data_raw.get("SPY"), "VIX": vix_data})

if __name__ == "__main__":
    run_optimization()