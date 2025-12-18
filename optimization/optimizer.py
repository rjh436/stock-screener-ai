
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
    Soft Fitness:
    - No hard penalties / no string matching.
    - Targets:
      - Win Rate: 60%
      - Avg Profit: 5%
    - If below target, apply a multiplier of (actual/target).
    """
    try:
        cagr = float(result.get("cagr", 0.0) or 0.0)
        avg_profit = float(result.get("avg_profit_pct", 0.0) or 0.0)
        win_rate = float(result.get("hit_rate", 0.0) or 0.0)
        trades = int(result.get("total_trades", 0) or 0)
        max_dd_pct = float(result.get("max_drawdown_pct", 0.0) or 0.0)
    except Exception:
        return 0.0

    # Base (growth + risk-adjusted component)
    dd_pct = abs(max_dd_pct)
    dd_frac = dd_pct / 100.0 if dd_pct > 0 else 0.0
    calmar = (cagr / dd_frac) if dd_frac > 0 else 0.0
    base = max(cagr, 0.0) * 1000.0 + max(calmar, 0.0) * 200.0

    # Soft Activity factor (avoid starvation without hard disqualification)
    activity_factor = min(1.0, trades / 20.0) if trades > 0 else 0.0

    # Soft targets
    win_target = 60.0
    profit_target = 5.0

    win_factor = (win_rate / win_target) if win_rate < win_target else 1.0
    profit_factor = (avg_profit / profit_target) if avg_profit < profit_target else 1.0

    fitness = base * activity_factor * win_factor * profit_factor
    return float(max(fitness, 0.0))

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

    # Run 5 Generations per Cycle
    for gen in range(5):
        print(f"   🧬 Gen {engine.generation_count} Evaluation...")
        pop_res = []
        with ThreadPoolExecutor(max_workers=16) as ex:
            futures = {
                # Fix Wiring: pass genome scoring_weights as final argument to run_backtest.
                ex.submit(
                    run_backtest,
                    GenericStrategy(g),
                    data_map,
                    None,
                    100000.0,
                    None,
                    global_context,
                    g.get("scoring_weights"),
                ): g
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
        print(f"      🏆 Top: {best.get('strategy')[:25]}.. | CAGR: {best.get('cagr',0):.1%} | Profit: {best.get('avg_profit_pct',0):.1f}% | Hit: {best.get('hit_rate',0):.1f}%")
        
        engine.evolve(ranked)
        with open(GEN_CONFIG, "w") as f:
            json.dump([r["genome"] for r in ranked[:15]], f, indent=4)

if __name__ == "__main__":
    print("🚀 Starting 'High Caliber' Optimization")
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    if not data_map: sys.exit(1)
    engine = EvolutionEngine()
    engine.population_size = 100
    
    # Run 5 Full Cycles
    for i in range(5):
        print(f"\n⚡ [Cycle {i+1}/5] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
