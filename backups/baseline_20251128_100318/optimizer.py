import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import fetch_data_pack
from execution.engine import run_backtest
from optimization.evolution import EvolutionEngine
from strategies.generic import GenericStrategy
from data.indices import get_index_symbols

GEN_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json"))


def calculate_fitness(result):
    """
    Objective Function:
    Prioritize Score (CAGR + Sharpe) but ensure minimal trade frequency.
    """
    score = result.get("Score", -1_000_000.0)
    trades = result.get("total_trades", 0)
    
    # FORCE HIGH FREQUENCY: Kill anything with < 40 trades
    if trades < 40:
        return -1_000_000.0
        
    return float(score)


def load_optimization_data(universe="S&P 1500", days=1260):
    print(f"📥 Loading Data for {universe} ({days} days)...")
    symbols = get_index_symbols(universe)
    
    if not symbols:
        print("❌ Error: No symbols found.")
        return {}, {}

    # Fetch Data
    data_map = fetch_data_pack(symbols, days=days)
    
    # Fetch Global Context (SPY, VIX)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
    
    global_context = {"SPY": g_data.get("SPY"), "VIX": vix}
    
    print(f"✅ Loaded {len(data_map)} symbols.")
    return data_map, global_context


def run_evolution_cycle(engine, data_map, global_context):
    """Runs 3 generations of evolution using pre-loaded data."""
    
    # Load existing population if available
    try:
        with open(GEN_CONFIG, "r") as f:
            engine.population = json.load(f)
            # If file exists but is empty/invalid, regenerate
            if not engine.population:
                engine.generate_initial_population()
    except Exception:
        engine.generate_initial_population()

    # Evolution Loop (3 Generations per Cycle)
    for gen in range(3):
        print(f"   🧬 Gen {engine.generation_count} Evaluation...")
        pop_res = []
        
        # Run Backtests in Parallel
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {
                ex.submit(run_backtest, GenericStrategy(g), data_map, None, 100000.0, None, global_context): g
                for g in engine.population
            }
            
            for f in futures:
                try:
                    res = f.result()
                    score = calculate_fitness(res)
                    pop_res.append({"genome": futures[f], "score": score, "stats": res})
                except Exception as e:
                    print(f"Strategy failed: {e}")

        if not pop_res:
            print("⚠️ No valid strategies in this generation.")
            continue

        # Rank and Evolve
        ranked = sorted(pop_res, key=lambda x: x["score"], reverse=True)
        
        # Log Top Performer
        best = ranked[0]["stats"]
        print(f"      🏆 Top: {best.get('strategy', 'Unknown')} | CAGR: {best.get('cagr', 0):.1%} | Trades: {best.get('total_trades', 0)} | Score: {ranked[0]['score']:.1f}")

        # Evolve next generation
        engine.evolve(ranked)

        # Save interim results
        with open(GEN_CONFIG, "w") as f:
            json.dump([r["genome"] for r in ranked[:15]], f, indent=4)


if __name__ == "__main__":
    print("🚀 Starting 'Hyper-Growth' Optimization Engine")
    print("🎯 Target: High CAGR + >2.0% Avg Profit")
    print("-------------------------------------------")
    
    # 1. Load Data ONCE
    # Cached data will make this fast on subsequent runs
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    
    if not data_map:
        sys.exit(1)

    # 2. Initialize Engine
    engine = EvolutionEngine()
    
    # 3. Run 4 Cycles (12 Generations total)
    start_time = time.time()
    
    for i in range(4):
        print(f"\n⚡ [Hyper-Growth Cycle {i+1}/4] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
        
    elapsed = (time.time() - start_time) / 60.0
    print(f"\n✅ Optimization Complete in {elapsed:.1f} minutes.")
