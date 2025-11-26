import random
import copy
import json
from typing import List, Dict
from strategies.generic import GenericStrategy

# Indicators available for the genome
# Added specific shifted indicators (highest20_1) to allow for proper breakout logic
class EvolutionEngine:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.2
        self.generation_count = 0
        self.population = []
        
        self.INDICATORS = {
            "price": ["close", "sma20", "sma50", "sma200", "bb_lower", "highest55", "lowest20"],
            "momentum": ["rs_trend", "rs_ratio"], # <--- NEW CATEGORY
            "oscillator": ["rsi2", "rsi14", "adx"],
            "volatility": ["atr14", "bb_width"],
            "volume": ["volume", "vol_ma20"],
            "market_regime": ["vix"]
        }
        self.OPERATORS = [">", "<"]

    def _random_rule(self):
        """Generate a completely random entry rule."""
        cat = random.choice(list(self.INDICATORS.keys()))
        col = random.choice(self.INDICATORS[cat])
        op = random.choice(self.OPERATORS)
        
        # Determine if we compare to a value or another indicator
        if random.random() < 0.3:
            # Compare to Fixed Value
            val = 0
            if "rsi" in col: val = random.randint(5, 95)
            elif "vix" in col: val = random.randint(15, 40)
            elif "adx" in col: val = random.randint(20, 40)
            elif "bb_width" in col: val = round(random.uniform(0.1, 0.5), 2)
            return {"col": col, "op": op, "val": val}
        else:
            # Compare to Reference Indicator (e.g. Close > SMA200)
            # Try to pick a reference from the same category or compatible
            ref = random.choice(self.INDICATORS[cat])
            return {"col": col, "op": op, "ref": ref}

    def _create_trend_ignition_strategy(self, i: int) -> Dict:
        """
        Seeds the population with a 'Trend Ignition' structure.
        Philosophy: High Volume Breakout in a Long-Term Uptrend.
        """
        genome = {
            "name": f"TrendIgnition_Gen{self.generation_count}_{i}",
            "type": "trend_ignition",
            "entry_rules": [
                # 1. Trend Filter: Close > SMA200
                {"col": "close", "op": ">", "ref": "sma200"},
                # 2. Breakout: Close > Yesterday's 20-Day High
                {"col": "close", "op": ">", "ref": "highest20_1"},
                # 3. Ignition: Volume > 1.5 * AvgVol20 (simulated via rule)
                # Note: Generic strategy logic might need specific handling for multipliers, 
                # but here we force Volume > VolMA20 as a baseline.
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [], # Rely on trailing stop or time stop
            "stop_loss_atr": round(random.uniform(2.0, 4.0), 1), # Tighter stops for breakouts
            "time_stop": random.choice([40, 60, 80]) # Swing trading duration
        }
        return genome

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        """Creates a purely random strategy."""
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(2, 4))],
            "exit_rules": [],
            "stop_loss_atr": round(random.uniform(3.0, 6.0), 1),
            "time_stop": random.choice([50, 70, 90, 120])
        }
        return genome

    def generate_initial_population(self):
        pop = []
        # Seed 30% with Trend Ignition Logic to guide evolution
        ignition_count = int(self.population_size * 0.3)
        
        print(f"🧬 Seeding Population: {ignition_count} Trend Ignition + {self.population_size - ignition_count} Random")
        
        for i in range(ignition_count):
            pop.append(self._create_trend_ignition_strategy(i))
            
        for i in range(self.population_size - ignition_count):
            pop.append(self._create_random_strategy(f"Gen0_Random{i}"))
            
        self.population = pop
        return pop

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        r = random.random()
        
        if r < 0.3:
            # Mutate Time Stop
            mutant["time_stop"] = random.choice([40, 50, 60, 80])
        elif r < 0.6: 
            # Mutate Entry Rules (Replace one)
            if mutant["entry_rules"]:
                idx = random.randint(0, len(mutant["entry_rules"])-1)
                mutant["entry_rules"][idx] = self._random_rule()
        elif r < 0.8:
            # Add a new rule (Constraint tightening)
            if len(mutant["entry_rules"]) < 5:
                mutant["entry_rules"].append(self._random_rule())
        else:
            # Mutate Stop Loss
            mutant["stop_loss_atr"] = round(random.uniform(2.0, 5.0), 1)
            
        return mutant

    def crossover(self, p1: Dict, p2: Dict) -> Dict:
        # Single point crossover of rules
        rules1 = p1["entry_rules"]
        rules2 = p2["entry_rules"]
        
        # Take half from p1, half from p2
        cut1 = len(rules1) // 2
        cut2 = len(rules2) // 2
        
        new_rules = rules1[:cut1] + rules2[cut2:]
        
        # Ensure at least one rule
        if not new_rules:
            new_rules = [self._random_rule()]

        child = {
            "name": f"Cross_{p1['name'].split('_')[0]}_{p2['name'].split('_')[0]}",
            "type": "hybrid",
            "entry_rules": new_rules,
            "exit_rules": [],
            "stop_loss_atr": round((p1["stop_loss_atr"] + p2["stop_loss_atr"]) / 2, 1),
            "time_stop": int((p1["time_stop"] + p2["time_stop"]) / 2)
        }
        return child

    def evolve(self, ranked_results: List[Dict]):
        self.generation_count += 1
        print(f"🧬 Creating Generation {self.generation_count}...")
        
        # Survival of the Fittest (Top 10% Elite)
        elite_count = 5
        survivors = [r["genome"] for r in ranked_results[:elite_count]]
        next_gen = survivors[:] # Elitism: Keep the best unchanged
        
        # Fill the rest
        while len(next_gen) < self.population_size:
            r = random.random()
            if r < 0.4:
                # Crossover (Breeding)
                p1 = random.choice(survivors)
                p2 = random.choice(survivors)
                next_gen.append(self.crossover(p1, p2))
            elif r < 0.8:
                # Mutation
                p = random.choice(survivors)
                next_gen.append(self.mutate(p))
            else:
                # Fresh Blood (Random)
                next_gen.append(self._create_random_strategy(f"Gen{self.generation_count}_Fresh"))
                
        self.population = next_gen
        return next_gen
