import random
import copy
import json
from typing import List, Dict
from strategies.generic import GenericStrategy

# ... (Keep existing INDICATORS and OPERATORS lists) ...
# RE-DEFINE EvolutionEngine Class ONLY:

class EvolutionEngine:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.2
        self.generation_count = 0
        self.population = []
        # ... (Keep self.INDICATORS mapping) ...
        self.INDICATORS = {
            "price": ["close", "open", "high", "low", "sma5", "sma10", "sma20", "sma50", "sma100", "sma200", "ema5", "ema10", "ema20", "ema50", "ema100", "ema200", "bb_mid", "bb_upper", "bb_lower", "highest20", "highest55", "lowest5", "lowest20", "kc_mid", "kc_upper"],
            "oscillator": ["rsi2", "rsi3", "rsi5", "rsi14", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist", "adx", "plus_di", "minus_di", "cci", "roc", "mom"],
            "volatility": ["atr10", "atr14", "atr20", "atr14_ma20", "atr_pct10", "atr_pct20", "bb_width"],
            "volume": ["volume", "vol_ma20", "avgvol50"],
            "market_regime": ["vix", "rs_ratio", "rs_sma20", "rs_trend"]
        }
        self.OPERATORS = [">", "<"]

    # ... (Keep helper methods _random_indicator, _random_value, _random_rule as is) ...
    # ONLY showing the Fixed Patience Logic below:

    def _random_indicator(self):
        return random.choice(list(self.INDICATORS.keys())) # Simplified for brevity in prompt

    def _random_value(self, indicator):
         return random.uniform(1, 100) # Simplified

    def _random_rule(self):
        cat = random.choice(list(self.INDICATORS.keys()))
        col = random.choice(self.INDICATORS[cat])
        return {"col": col, "op": random.choice(self.OPERATORS), "val": 0}

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(1, 3))],
            "exit_rules": [self._random_rule() for _ in range(random.randint(1, 2))],
            "stop_loss_atr": round(random.uniform(2.0, 3.5), 1),
            "time_stop": random.choice([30, 45, 60, 80]) # FIXED: Long holds
        }
        return genome

    def generate_initial_population(self):
        pop = []
        for i in range(self.population_size):
            genome = self._create_random_strategy(f"Gen0_Strat{i}")
            genome["time_stop"] = random.choice([30, 45, 60, 80]) # FIXED
            pop.append(genome)
        self.population = pop
        return pop

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        if r < 0.3:
            mutant["stop_loss_atr"] = round(random.uniform(2.0, 3.5), 1)
        elif r < 0.6:
            mutant["time_stop"] = random.choice([30, 45, 60, 80]) # FIXED
        # ... rest of mutation logic ...
        return mutant

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        child = {
            "name": f"Cross_{parent1['name']}_{parent2['name']}",
            "type": "hybrid",
            "entry_rules": copy.deepcopy(parent1["entry_rules"]),
            "exit_rules": copy.deepcopy(parent2["exit_rules"]),
            "stop_loss_atr": round((parent1["stop_loss_atr"] + parent2["stop_loss_atr"]) / 2, 1),
            "time_stop": int((parent1["time_stop"] + parent2["time_stop"]) / 2)
        }
        return child

    def evolve(self, ranked_results: List[Dict]):
        self.generation_count += 1
        print(f"Creating Generation {self.generation_count}...")
        survivors = [r["genome"] for r in ranked_results[:3]]
        next_gen = survivors[:]
        while len(next_gen) < 7:
            next_gen.append(self.crossover(random.choice(survivors), random.choice(survivors)))
        while len(next_gen) < self.population_size:
            if random.random() < 0.3:
                next_gen.append(self.mutate(random.choice(survivors)))
            else:
                next_gen.append(self._create_random_strategy(f"Gen{self.generation_count}_Fresh"))
        self.population = next_gen
        return next_gen