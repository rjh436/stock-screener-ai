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
    # SMART FITNESS FUNCTION (Role-Based Objectives)
    name = result.get("strategy", "")
    cagr = result.get("cagr", 0)
    avg_profit = result.get("avg_profit_pct", 0)
    hit_rate = result.get("hit_rate", 0)
    trades = result.get("total_trades", 0)

    # 1. Global Sanity Check (Dead Strategies)
    if trades < 10: return -1e6

    # 2. MACHINE GUN GOAL: Avg Profit > 5%
    if "MachineGun" in name:
        # Hard Constraint: Must make real money per trade
        if avg_profit < 5.0:
            # Penalty: The further from 5%, the worse the score
            return -5000.0 + (avg_profit * 100) 
        
        # If Constraint Met: Optimize for Total Return (CAGR)
        # Bonus for maintaining high activity (>100 trades)
        activity_bonus = min(trades, 200) / 2.0
        return (cagr * 5000) + activity_bonus

    # 3. SNIPER GOAL: High Hit Rate (Without sacrificing Power)
    if "Sniper" in name or "Gen12" in name:
        # Safety Floor: Don't break the Wealth Building (40% CAGR / 15% Profit)
        if cagr < 0.40 or avg_profit < 15.0:
            return -5000.0
            
        # If Floor Met: Maximize Precision (Hit Rate)
        # We weight Hit Rate heavily to push it up from 57%
        return (hit_rate * 100) + (cagr * 1000)

    # Fallback for generic strategies
    return result.get("Score", 0)

def load_optimization_data(universe="S&P 1500", days=1260):
    print(f"📥 Loading Data for {universe} ({days} days)...")
    symbols = get_index_symbols(universe)
    if not symbols:
        print("❌ Error: No symbols found.")
        return {}, {}
    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX")
    if vix is None: vix = g_data.get("VIX")
    global_context = {"SPY": g_data.get("SPY"), "VIX": vix}
    print(f"✅ Loaded {len(data_map)} symbols.")
    return data_map, global_context

def run_evolution_cycle(engine, data_map, global_context):
    try:
        with open(GEN_CONFIG, "r") as f:
            engine.population = json.load(f)
            if not engine.population: engine.generate_initial_population()
    except: engine.generate_initial_population()

    for gen in range(3):
        print(f"   🧬 Gen {engine.generation_count} Evaluation...")
        pop_res = []
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
                except: pass

        if not pop_res: continue

        ranked = sorted(pop_res, key=lambda x: x["score"], reverse=True)
        best = ranked[0]["stats"]
        # NEW: Display Avg Signals Per Day
        print(f"      🏆 Top: {best.get('strategy', 'Unknown')} | CAGR: {best.get('cagr', 0):.1%} | Signals/Day: {best.get('avg_signals_per_day', 0):.1f} | Score: {ranked[0]['score']:.1f}")

        engine.evolve(ranked)
        with open(GEN_CONFIG, "w") as f:
            json.dump([r["genome"] for r in ranked[:15]], f, indent=4)

if __name__ == "__main__":
    print("🚀 Starting 'Hyper-Growth' Optimization Engine")
    print("🎯 Target: High CAGR + >2.0% Avg Profit + High Signal Density")
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    if not data_map: sys.exit(1)
    engine = EvolutionEngine()
    start_time = time.time()
    for i in range(4):
        print(f"\n⚡ [Hyper-Growth Cycle {i+1}/4] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
    elapsed = (time.time() - start_time) / 60.0
    print(f"\n✅ Optimization Complete in {elapsed:.1f} minutes.")
