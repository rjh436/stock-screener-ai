import os


def configure_high_caliber():
    print("🎯 CONFIGURING 'HIGH CALIBER' OPTIMIZATION PROTOCOLS...")

    # --- 1. THE STRICT JUDGE (Optimizer) ---
    opt_path = "optimization/optimizer.py"
    opt_code = """
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
        print(f"\\n⚡ [Cycle {i+1}/5] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
"""
    with open(opt_path, "w") as f:
        f.write(opt_code)
    print("   ✅ Optimizer Updated: Mandating >5% Profit (MG) and >60% Hit Rate (Sniper).")

    # --- 2. UPGRADE EVOLUTION (The Builder) ---
    evo_path = "optimization/evolution.py"
    evo_code = """
import random
import copy
import json
from typing import List, Dict
from strategies.generic import GenericStrategy

class EvolutionEngine:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.2
        self.generation_count = 0
        self.population = []
        
        self.INDICATORS = {
            "price": ["close", "sma50", "sma200", "bb_lower"],
            "momentum": ["rs_trend", "rs_ratio", "cci"],
            "oscillator": ["rsi2", "rsi14", "adx", "stoch_k"],
            "volatility": ["atr14", "bb_width"],
            "volume": ["volume", "vol_ma20"]
        }
        self.OPERATORS = [">", "<"]

    def _random_rule(self):
        cat = random.choice(list(self.INDICATORS.keys()))
        col = random.choice(self.INDICATORS[cat])
        op = random.choice(self.OPERATORS)
        val = 0
        if "rsi" in col: val = random.randint(5, 95)
        elif "vix" in col: val = random.randint(15, 40)
        elif "adx" in col: val = random.randint(20, 40)
        elif "bb_width" in col: val = round(random.uniform(0.1, 0.5), 2)
        elif "cci" in col: val = random.choice([-100, 0, 100])
        return {"col": col, "op": op, "val": val}

    def generate_initial_population(self):
        pass

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.3:
            # TIME STOP EXPANSION
            # To hit 5% profit, we need longer hold times.
            # Shifting range from [5-20] to [20-60].
            mutant["time_stop"] = random.choice([20, 25, 30, 35, 40, 45, 50, 60])
            
        elif r < 0.5:
            # PROFIT TARGET TUNING
            # Focusing on the 5% - 15% range
            if "exit_rules" in mutant:
                found = False
                for rule in mutant["exit_rules"]:
                    if rule.get("type") == "profit_target":
                        found = True
                        curr = float(rule.get("val", 1.06))
                        # Mutate
                        new_val = curr + random.choice([-0.01, 0.01, 0.02, 0.05])
                        # Keep it between 1.05 and 1.25
                        rule["val"] = round(max(1.05, min(1.25, new_val)), 2)
                
                # If no target, add one
                if not found and random.random() < 0.5:
                    mutant["exit_rules"].append({"type": "profit_target", "val": 1.07})
                    
        elif r < 0.7:
            # ENTRY RULE TWEAKS
            if mutant.get("entry_rules"):
                idx = random.randint(0, len(mutant["entry_rules"])-1)
                mutant["entry_rules"][idx] = self._random_rule()
                
        else:
            mutant["stop_loss_atr"] = round(random.uniform(3.0, 7.0), 1)
            
        return mutant

    def evolve(self, ranked):
        self.generation_count += 1
        survivors = [r["genome"] for r in ranked[:5]]
        next_gen = survivors[:]
        while len(next_gen) < self.population_size:
            # Prefer mutation over crossover to explore new Profit/Time parameters
            next_gen.append(self.mutate(random.choice(survivors)))
        self.population = next_gen
        return next_gen
"""
    with open(evo_path, "w") as f:
        f.write(evo_code)
    print("   ✅ Evolution Updated: AI now has tools to hit 5% targets (Longer Holds).")
    
    print("\n🚀 HIGH CALIBER PROTOCOLS ENGAGED.")
    print("   Run './Run_Optimizer.command' now.")

if __name__ == "__main__":
    configure_high_caliber()
