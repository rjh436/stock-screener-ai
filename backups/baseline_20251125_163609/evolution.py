import random
import copy
import json
from typing import List, Dict
from strategies.generic import GenericStrategy

INDICATORS = ["close", "open", "high", "low", "volume", "sma5", "sma20", "sma50", "sma200", "rsi2", "rsi14", "atr14", "bb_upper", "bb_lower", "vix", "adx"]
OPERATORS = [">", "<"]
TEMPLATES = []

class EvolutionEngine:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.2
        self.generation_count = 0
        self.population = []
        self.INDICATORS = {
            "price": ["close", "sma20", "sma50", "sma200", "bb_lower", "highest55", "lowest20"],
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
        return {"col": col, "op": op, "val": val}

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(1, 3))],
            "exit_rules": [],
            "stop_loss_atr": round(random.uniform(3.0, 6.0), 1),
            "time_stop": random.choice([50, 70, 90, 120])
        }
        return genome

    def generate_initial_population(self):
        pop = []
        for i in range(self.population_size):
            pop.append(self._create_random_strategy(f"Gen0_Strat{i}"))
        self.population = pop
        return pop

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.4:
            mutant["time_stop"] = random.choice([50, 70, 90, 120])
        elif r < 0.7: 
            if mutant["entry_rules"]:
                mutant["entry_rules"][random.randint(0, len(mutant["entry_rules"])-1)] = self._random_rule()
        else:
            mutant["stop_loss_atr"] = round(random.uniform(3.0, 6.0), 1)
            
        mutant["exit_rules"] = [] 
        return mutant

    def crossover(self, p1: Dict, p2: Dict) -> Dict:
        child = {
            "name": f"Cross_{p1['name']}_{p2['name']}",
            "type": "hybrid",
            "entry_rules": copy.deepcopy(p1["entry_rules"] if random.random() < 0.5 else p2["entry_rules"]),
            "exit_rules": [],
            "stop_loss_atr": round((p1["stop_loss_atr"] + p2["stop_loss_atr"]) / 2, 1),
            "time_stop": int((p1["time_stop"] + p2["time_stop"]) / 2)
        }
        return child

    def evolve(self, ranked_results: List[Dict]):
        self.generation_count += 1
        print(f"Creating Generation {self.generation_count}...")
        survivors = [r["genome"] for r in ranked_results[:5]]
        next_gen = survivors[:]
        while len(next_gen) < self.population_size:
            if random.random() < 0.4:
                next_gen.append(self.crossover(random.choice(survivors), random.choice(survivors)))
            elif random.random() < 0.8:
                next_gen.append(self.mutate(random.choice(survivors)))
            else:
                next_gen.append(self._create_random_strategy(f"Gen{self.generation_count}_Fresh"))
        self.population = next_gen
        return next_gen