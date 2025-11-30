
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
        # ... standard init ...
        pass

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.2:
            # Mutate Time Stop
            mutant["time_stop"] = random.choice([5, 8, 10, 12, 15, 20])
        elif r < 0.4:
            # Mutate Profit Target (CRITICAL UPDATE)
            if "exit_rules" in mutant:
                found_target = False
                for rule in mutant["exit_rules"]:
                    if rule.get("type") == "profit_target":
                        found_target = True
                        current = float(rule.get("val", 1.06))
                        # Small nudge or random reset
                        if random.random() < 0.5:
                            new_val = current + random.choice([-0.01, 0.01])
                        else:
                            new_val = 1.0 + (random.randint(4, 15) / 100.0)
                        rule["val"] = round(max(1.04, min(1.15, new_val)), 2)
                
                # If no target, add one
                if not found_target and random.random() < 0.3:
                    mutant["exit_rules"].append({"type": "profit_target", "val": round(1.0 + (random.randint(4, 12) / 100.0), 2)})

        elif r < 0.6: 
            if mutant.get("entry_rules"):
                idx = random.randint(0, len(mutant["entry_rules"])-1)
                mutant["entry_rules"][idx] = self._random_rule()
        elif r < 0.8:
            if len(mutant.get("entry_rules", [])) < 5:
                mutant.setdefault("entry_rules", []).append(self._random_rule())
        else:
            mutant["stop_loss_atr"] = round(random.uniform(2.0, 5.0), 1)
            
        return mutant

    # ... rest of class ...
    def crossover(self, p1, p2): return p1 # Simplified for restore
    def evolve(self, ranked): return [] # Simplified
