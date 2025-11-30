import os


def upgrade_optimizer_only():
    print("🚀 UPGRADING OPTIMIZER & EVOLUTION LOGIC (Keeping Engine Intact)...")

    # --- 1. UPGRADE OPTIMIZER (The Judge) ---
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
    # SMART FITNESS FUNCTION (Role-Based Objectives)
    name = result.get("strategy", "")
    cagr = result.get("cagr", 0)
    avg_prof = result.get("avg_profit_pct", 0)
    hit = result.get("hit_rate", 0)
    trades = result.get("total_trades", 0)
    
    if trades < 10: return -1e6
    
    # GOAL 1: Machine Gun must make > 5% Profit per trade
    if "MachineGun" in name:
        if avg_prof < 5.0: 
            # Heavy Penalty to force evolution away from churning
            return -10000 + (avg_prof * 100) 
        # If goal met, maximize CAGR + Activity
        return (cagr * 5000) + (trades / 2.0)

    # GOAL 2: Sniper must maximize Accuracy (Hit Rate)
    if "Sniper" in name or "Gen12" in name:
        # Safety Floor: Don't break the bank
        if cagr < 0.20: return -5000
        # Priority: Hit Rate
        return (hit * 100) + (cagr * 1000)
        
    # Default behavior
    return (cagr * 1000)

def load_optimization_data(universe="S&P 1500", days=1260):
    print(f"📥 Loading Data for {universe} ({days} days)...")
    symbols = get_index_symbols(universe)
    if not symbols: return {}, {}
    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX") or g_data.get("VIX")
    return data_map, {"SPY": g_data.get("SPY"), "VIX": vix}

def run_evolution_cycle(engine, data_map, global_context):
    try:
        with open(GEN_CONFIG, "r") as f:
            engine.population = json.load(f)
    except: engine.generate_initial_population()

    # Reduced to 2 generations per cycle for speed, since we run multiple cycles
    for gen in range(2):
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
        print(f"      🏆 Top: {best.get('strategy')} | CAGR: {best.get('cagr',0):.1%} | Profit: {best.get('avg_profit_pct',0):.1f}% | Hit: {best.get('hit_rate',0):.1f}%")
        
        engine.evolve(ranked)
        with open(GEN_CONFIG, "w") as f:
            json.dump([r["genome"] for r in ranked[:15]], f, indent=4)

if __name__ == "__main__":
    print("🚀 Starting 'Targeted' Optimization Engine")
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    if not data_map: sys.exit(1)
    engine = EvolutionEngine()
    
    for i in range(5):
        print(f"\\n⚡ [Cycle {i+1}/5] Starting...")
        run_evolution_cycle(engine, data_map, global_context)
"""
    with open(opt_path, "w") as f:
        f.write(opt_code)
    print("   ✅ Optimizer Updated: Goals Enforced (MG > 5%, Sniper > Hit Rate).")

    # --- 2. UPGRADE EVOLUTION (The Mutator) ---
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
            "price": ["close", "sma20", "sma50", "sma200", "bb_lower", "highest55", "lowest20"],
            "momentum": ["rs_trend", "rs_ratio"],
            "oscillator": ["rsi2", "rsi14", "adx"],
            "volatility": ["atr14", "bb_width"],
            "volume": ["volume", "vol_ma20"],
            "market_regime": ["vix"]
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
        return {"col": col, "op": op, "val": val}

    def generate_initial_population(self):
        pass

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.3:
            # EXPANDED TIME STOP RANGE (Up to 45 days for 5% targets)
            mutant["time_stop"] = random.choice([10, 15, 20, 25, 30, 35, 40, 45])
        elif r < 0.5:
            # PROFIT TARGET MUTATION (Up to 25%)
            if "exit_rules" in mutant:
                found = False
                for rule in mutant["exit_rules"]:
                    if rule.get("type") == "profit_target":
                        found = True
                        curr = float(rule.get("val", 1.06))
                        # Mutate
                        new_val = curr + random.choice([-0.02, 0.02, 0.05])
                        rule["val"] = round(max(1.04, min(1.25, new_val)), 2)
                
                # If no target, occasionally add one
                if not found and random.random() < 0.4:
                    mutant["exit_rules"].append({"type": "profit_target", "val": 1.08})
        elif r < 0.6: 
            if mutant.get("entry_rules"):
                idx = random.randint(0, len(mutant["entry_rules"])-1)
                mutant["entry_rules"][idx] = self._random_rule()
        elif r < 0.8:
            if len(mutant.get("entry_rules", [])) < 5:
                mutant.setdefault("entry_rules", []).append(self._random_rule())
        else:
            mutant["stop_loss_atr"] = round(random.uniform(2.0, 6.0), 1)
            
        return mutant

    def crossover(self, p1, p2):
        # Simple crossover for robustness
        child = copy.deepcopy(p1)
        child["name"] = f"Cross_{p1.get('name','')}_{p2.get('name','')}"[:30]
        if random.random() < 0.5: child["stop_loss_atr"] = p2.get("stop_loss_atr", 3.0)
        return child

    def evolve(self, ranked):
        self.generation_count += 1
        survivors = [r["genome"] for r in ranked[:5]]
        next_gen = survivors[:]
        while len(next_gen) < self.population_size:
            next_gen.append(self.mutate(random.choice(survivors)))
        self.population = next_gen
        return next_gen
"""
    with open(evo_path, "w") as f:
        f.write(evo_code)
    print("   ✅ Evolution Updated: Mutation Ranges Expanded.")
    
    print("\n✅ READY FOR OPTIMIZATION.")
    print("   - Engine: Golden State (Untouched)")
    print("   - Optimizer: Enforcing 5% Profit Goal")
    print("   - Run './Run_Optimizer.command' now.")


if __name__ == "__main__":
    upgrade_optimizer_only()
