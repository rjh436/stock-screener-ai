
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
    name = result.get("strategy", "")
    cagr = result.get("cagr", 0)
    avg_prof = result.get("avg_profit_pct", 0)
    hit = result.get("hit_rate", 0)
    trades = result.get("total_trades", 0)
    
    # 1. Global Activity Floor (Must trade to be valid)
    if trades < 20: return -1e6
    
    # 2. MACHINE GUN MANDATE: "The Heavy Gun"
    # Goal: > 5% Profit per trade.
    if "MachineGun" in name:
        if avg_prof < 5.0: 
            # Extreme Penalty: The AI must fix this first.
            return -10000 + avg_prof 
        
        # If Profit > 5%, maximize CAGR
        return (cagr * 5000)

    # 3. SNIPER MANDATE: "Precision"
    # Goal: > 60% Win Rate (currently ~57%)
    if "Sniper" in name or "Gen12" in name or "Evolved" in name:
        if hit < 60.0:
            # Penalty for low accuracy
            return -5000 + hit
            
        # If Hit Rate > 60%, maximize CAGR
        return (cagr * 5000) + (hit * 100)
        
    return (cagr * 1000)

def load_optimization_data(universe="S&P 1500", days=1260):
    print(f"📥 Loading Data for {universe}...")
    symbols = get_index_symbols(universe)
    if not symbols: return {}, {}
    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    
    # Robust VIX loading
    vix = g_data.get("$VIX")
    if vix is None: vix = g_data.get("VIX")
    
    return data_map, {"SPY": g_data.get("SPY"), "VIX": vix}

def run_evolution_cycle(engine, data_map, global_context):
    try:
        with open(GEN_CONFIG, "r") as f:
            engine.population = json.load(f)
    except: engine.generate_initial_population()

    # Run 3 Generations per Cycle
    for gen in range(3):
        print(f"   🧬 Gen {engine.generation_count} Evaluation...")
        pop_res = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(run_backtest, GenericStrategy(g), data_map, None, 100000.0, None, global_context): g for g in engine.population}
            for f in futures:
                try:
                    res = f.result()
                    score = calculate_fitness(res)
                    pop_res.append({"genome": futures[f], "score": score, "stats": res})
                except: pass
        
        if not pop_res: continue
        ranked = sorted(pop_res, key=lambda x: x["score"], reverse=True)
        best = ranked[0]["stats"]
        print(f"      🏆 Top: {best.get('strategy')[:25]}.. | CAGR: {best.get('cagr',0):.1%} | Profit: {best.get('avg_profit_pct',0):.1f}% | Hit: {best.get('hit_rate',0):.1f}%")
        
        engine.evolve(ranked)
        with open(GEN_CONFIG, "w") as f:
            json.dump([r["genome"] for r in ranked[:15]], f, indent=4)

if __name__ == "__main__":
    print("🚀 Starting 'High Caliber' Optimization")
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    if not data_map: sys.exit(1)
    engine = EvolutionEngine()
    
    # Run 5 Full Cycles
    for i in range(5):
        print(f"\n⚡ [Cycle {i+1}/5] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
