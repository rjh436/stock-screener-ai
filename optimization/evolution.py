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
        return {
            "sniper_bonus": round(random.uniform(30.0, 80.0), 2),
            "rsi_factor": round(random.uniform(1.5, 3.5), 2),
            "trend_bonus": round(random.uniform(10.0, 40.0), 2),
            "vol_bonus": round(random.uniform(5.0, 20.0), 2),
            "atr_high_bonus": round(random.uniform(10.0, 25.0), 2),
            "atr_med_bonus": round(random.uniform(2.0, 8.0), 2),
            "trend_penalty": round(random.uniform(-25.0, -5.0), 2),
        }

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(2, 4))],
            "exit_rules": [{"type": "profit_target", "val": round(random.uniform(1.05, 1.25), 2)}],
            "stop_loss_atr": round(random.uniform(3.0, 7.0), 1),
            "time_stop": random.choice([30, 45, 60, 75, 90]),
            "scoring_weights": self._random_scoring_weights(),
            # GEN 15: The Limit Runner Params
            "adx_threshold": 25.0,
            "rsi_strong": 40.0,
            "rsi_weak": 15.0,
            "limit_ratio": 0.98,  # 2% Discount Default
            "trail_atr": 3.0      # 3x ATR Trail Default
        }
        return genome

    def create_initial_population(self, base_name: str = "Strategy", base_strategies: List[Dict] = None) -> List[Dict]:
        pop: List[Dict] = []
        
        # 1. Golden Seed
        golden_seed = self._create_random_strategy(base_name)
        golden_seed["name"] = f"{base_name}_GOLDEN"
        pop.append(golden_seed)

        # 2. Neighborhood
        for i in range(10):
            pop.append(self.mutate(golden_seed))

        # 3. Provided Strategies (Seed the Optimizer)
        if base_strategies:
            for strat in base_strategies:
                clone = copy.deepcopy(strat)
                # Ensure Gen 15 params exist
                clone["adx_threshold"] = clone.get("adx_threshold", 25.0)
                clone["rsi_strong"] = clone.get("rsi_strong", 40.0)
                clone["rsi_weak"] = clone.get("rsi_weak", 15.0)
                clone["limit_ratio"] = clone.get("limit_ratio", 0.98)
                clone["trail_atr"] = clone.get("trail_atr", 3.0)
                pop.append(clone)

        # 4. Fill Rest
        while len(pop) < self.population_size:
            pop.append(self._create_random_strategy(f"{base_name}_EXPLORE"))

        self.population = pop
        return pop
    
    def generate_initial_population(self, base_strategies: List[Dict] = None, base_name: str = "Strategy"):
        return self.create_initial_population(base_name=base_name, base_strategies=base_strategies)

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.25:
            # GEN 15 EXECUTION & EXIT MUTATIONS
            trait = random.choice(["limit", "trail", "dual_lane"])
            
            if trait == "limit":
                # Mutate Entry Discount (0.93 to 1.00)
                # 1.00 = Market Order, 0.93 = Deep 7% discount
                mutant["limit_ratio"] = round(random.uniform(0.93, 1.00), 3)
                
            elif trait == "trail":
                # Mutate Trailing Stop (2.0 to 6.0 ATR)
                mutant["trail_atr"] = round(random.uniform(2.0, 6.0), 1)
                
            elif trait == "dual_lane":
                sub_trait = random.choice(["adx", "strong", "weak"])
                if sub_trait == "adx":
                    mutant["adx_threshold"] = float(random.randint(15, 35))
                elif sub_trait == "strong":
                    mutant["rsi_strong"] = float(random.randint(30, 55))
                elif sub_trait == "weak":
                    mutant["rsi_weak"] = float(random.randint(5, 25))

        elif r < 0.45:
            # TIME STOP & PROFIT
            mutant["time_stop"] = random.choice([20, 30, 45, 60, 80])
            # Note: Profit Target is less relevant with Trailing Stop, but kept as failsafe
            for rule in mutant.get("exit_rules", []):
                if rule["type"] == "profit_target":
                    rule["val"] = round(random.uniform(1.05, 1.30), 2)

        elif r < 0.65:
            # BASE STOP LOSS
            mutant["stop_loss_atr"] = round(random.uniform(2.5, 6.0), 1)

        else:
            # SCORING WEIGHTS
            if "scoring_weights" in mutant:
                k = random.choice(list(mutant["scoring_weights"].keys()))
                mutant["scoring_weights"][k] = round(mutant["scoring_weights"][k] * random.uniform(0.8, 1.2), 2)
            
        return mutant

    def evolve(self, ranked):
        self.generation_count += 1
        survivors = [r["genome"] for r in ranked[:10]] # Top 10 survive
        next_gen = survivors[:]
        while len(next_gen) < self.population_size:
            parent = random.choice(survivors)
            next_gen.append(self.mutate(parent))
        self.population = next_gen
        return next_gen
