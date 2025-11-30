
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
