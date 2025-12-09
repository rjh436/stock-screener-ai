
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

    def _random_scoring_weights(self) -> Dict:
        """Generate scoring weights within safe, proven ranges."""
        return {
            "sniper_bonus": round(random.uniform(30.0, 80.0), 2),
            "rsi_factor": round(random.uniform(1.5, 3.5), 2),
            "trend_bonus": round(random.uniform(10.0, 40.0), 2),
            "vol_bonus": round(random.uniform(5.0, 20.0), 2),
            "atr_high_bonus": round(random.uniform(10.0, 25.0), 2),
            # Keep supportive weights near proven baselines
            "atr_med_bonus": round(random.uniform(2.0, 8.0), 2),
            "trend_penalty": round(random.uniform(-25.0, -5.0), 2),
        }

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        """Creates a random but viable strategy with constrained scoring weights."""
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(2, 4))],
            "exit_rules": [{"type": "profit_target", "val": round(random.uniform(1.05, 1.25), 2)}],
            "stop_loss_atr": round(random.uniform(3.0, 7.0), 1),
            "time_stop": random.choice([30, 45, 60, 75, 90]),
            "scoring_weights": self._random_scoring_weights(),
        }
        return genome

    def create_initial_population(self, base_name: str = "Strategy", base_strategies: List[Dict] = None) -> List[Dict]:
        """Seed population with Golden weights, neighborhood perturbations, then exploration (optionally seeding provided strategies)."""
        pop: List[Dict] = []

        golden_weights = {
            "rsi_factor": 2.0,
            "atr_high_bonus": 15.0,
            "atr_med_bonus": 5.0,
            "vol_bonus": 10.0,
            "sniper_bonus": 50.0,
            "trend_bonus": 20.0,
            "trend_penalty": -15.0,
        }

        # Tier 1: Golden Seed
        golden_seed = self._create_random_strategy(base_name)
        golden_seed["name"] = f"{base_name}_GOLDEN"
        golden_seed["scoring_weights"] = copy.deepcopy(golden_weights)
        pop.append(golden_seed)

        # Tier 2: Golden Neighborhood (20% of population)
        neighborhood_count = max(1, int(self.population_size * 0.2))
        for i in range(neighborhood_count):
            neighbor = copy.deepcopy(golden_seed)
            neighbor["name"] = f"{base_name}_NEIGHBOR_{i+1}"
            perturbed = {}
            for k, v in golden_weights.items():
                perturbed[k] = round(v * random.uniform(0.8, 1.2), 2)
            neighbor["scoring_weights"] = perturbed
            pop.append(neighbor)

        # Tier 3a: Include provided base strategies (if any) to preserve known performers
        if base_strategies:
            for idx, strat in enumerate(base_strategies):
                if len(pop) >= self.population_size:
                    break
                clone = copy.deepcopy(strat)
                clone["name"] = f"{base_name}_BASE_{idx+1}"
                pop.append(clone)

        # Tier 3b: Pure Exploration (fill remaining slots)
        while len(pop) < self.population_size:
            pop.append(self._create_random_strategy(f"{base_name}_EXPLORE"))

        self.population = pop
        return pop

    def generate_initial_population(self, base_strategies: List[Dict] = None, base_name: str = "Strategy"):
        # Backward compatibility for callers expecting the old name
        return self.create_initial_population(base_name=base_name, base_strategies=base_strategies)

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
