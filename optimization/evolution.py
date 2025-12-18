import random
import copy
import math
from typing import List, Dict

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
            # GEN 16.1 PARAMETERS
            "adx_threshold": 25.0,
            "rsi_strong": 40.0,
            "rsi_weak": 15.0,
            "limit_ratio": 0.98,
            "trail_atr": 3.0,
            "trail_activation": 1.03
        }
        return genome

    def calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculates the fitness score of a strategy based on backtest metrics.
        Implements a 'Profit Gate' to kill micro-scalpers.
        """
        cagr = metrics.get('cagr', 0)
        win_rate = metrics.get('hit_rate', 0)
        avg_profit = metrics.get('avg_profit_pct', 0)
        max_dd = metrics.get('max_drawdown_pct', 0) # Negative number
        trades = metrics.get('total_trades', 0)

        # 1. Base Score (Growth driven)
        score = (cagr * 100) * 2.0 

        # 2. Risk Penalty (Punish Drawdown)
        if max_dd < -15.0:
            score -= (abs(max_dd) - 15.0) * 5.0 # Heavy penalty for busting safety limit

        # 3. Win Rate Bonus (But capped impact)
        if win_rate > 60.0:
            score += (win_rate - 60.0) * 0.5

        # 4. THE PROFIT GATE (The Anti-Scalper Logic)
        # If the strategy makes peanuts per trade, we nuke its score.
        if avg_profit < 2.5:
            score *= 0.5  # 50% Penalty
            score -= 20.0 # Flat Penalty

        # 5. Starvation Penalty
        if trades < 30:
            score = 0.0 # Disqualify starvation

        return max(score, 0.0)

    def create_initial_population(self, base_name: str = "Strategy", base_strategies: List[Dict] = None) -> List[Dict]:
        pop: List[Dict] = []
        if base_strategies:
            for strat in base_strategies:
                clone = copy.deepcopy(strat)
                # Ensure Gen 16 params exist
                clone["limit_ratio"] = clone.get("limit_ratio", 0.98)
                clone["trail_atr"] = clone.get("trail_atr", 3.0)
                pop.append(clone)
        while len(pop) < self.population_size:
            pop.append(self._create_random_strategy(f"{base_name}_EXPLORE"))
        self.population = pop
        return pop
    
    def generate_initial_population(self, base_strategies: List[Dict] = None, base_name: str = "Strategy"):
        return self.create_initial_population(base_name=base_name, base_strategies=base_strategies)

    def mutate(self, genome: Dict) -> Dict:
        mutant = copy.deepcopy(genome)
        base_name = mutant.get("name", "Strategy")
        mutant["name"] = f"{base_name}_mut"

        r = random.random()

        if r < 0.3:
            # GEN 16 EXECUTION MUTATIONS
            trait = random.choice(["limit", "trail", "dual_lane"])
            if trait == "limit":
                mutant["limit_ratio"] = round(random.uniform(0.95, 1.00), 3)
            elif trait == "trail":
                mutant["trail_atr"] = round(random.uniform(2.5, 6.0), 1)
            elif trait == "dual_lane":
                sub_trait = random.choice(["adx", "strong", "weak"])
                if sub_trait == "adx":
                    mutant["adx_threshold"] = float(random.randint(15, 35))
                elif sub_trait == "strong":
                    mutant["rsi_strong"] = float(random.randint(30, 55))
                elif sub_trait == "weak":
                    mutant["rsi_weak"] = float(random.randint(5, 25))

        elif r < 0.5:
            # STOP LOSS
            mutant["stop_loss_atr"] = round(random.uniform(2.5, 6.0), 1)

        else:
            # SCORING WEIGHTS (Adaptive Mutation + Breakthrough Mutation)
            scoring = copy.deepcopy(mutant.get("scoring_weights") or {})
            if not scoring:
                scoring = self._random_scoring_weights()

            key = random.choice(list(scoring.keys()))

            # Adaptive mutation: early exploration vs late refinement
            if self.generation_count < 20:
                scale_low, scale_high = 0.50, 1.50
            else:
                scale_low, scale_high = 0.85, 1.15

            # 10% breakthrough mutation: reset weight to a brand-new value.
            if random.random() < 0.10:
                new_val = round(random.uniform(5.0, 80.0), 2)
                if "penalty" in key and float(scoring.get(key, 0) or 0) < 0:
                    new_val = -new_val
                scoring[key] = new_val
            else:
                try:
                    cur = float(scoring.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    cur = 0.0
                scoring[key] = round(cur * random.uniform(scale_low, scale_high), 2)

            mutant["scoring_weights"] = scoring

        return mutant

    def evolve(self, ranked):
        """
        Quota Elitism:
        - The next generation always contains a 50/50 split of:
          - Wealth strategies: type == "wealth"
          - Income strategies: type in {"income", "hybrid"}
        """
        self.generation_count += 1

        target_wealth = self.population_size // 2
        target_income = self.population_size - target_wealth

        elite_total = min(10, self.population_size)
        elite_wealth = min(target_wealth, elite_total // 2)
        elite_income = min(target_income, elite_total - elite_wealth)

        def is_wealth(g: Dict) -> bool:
            return str(g.get("type", "")).lower() == "wealth"

        def is_income(g: Dict) -> bool:
            return str(g.get("type", "")).lower() in {"income", "hybrid"}

        # Select elites per bucket from the ranked list (best -> worst)
        wealth_elites: List[Dict] = []
        income_elites: List[Dict] = []
        for r in ranked or []:
            genome = r.get("genome") if isinstance(r, dict) else None
            if not isinstance(genome, dict):
                continue

            if is_wealth(genome):
                if len(wealth_elites) < elite_wealth:
                    elite = copy.deepcopy(genome)
                    elite["type"] = "wealth"
                    wealth_elites.append(elite)
            else:
                if len(income_elites) < elite_income:
                    elite = copy.deepcopy(genome)
                    if not is_income(elite):
                        elite["type"] = "income"
                    income_elites.append(elite)

            if len(wealth_elites) >= elite_wealth and len(income_elites) >= elite_income:
                break

        # Fallback seeding if a bucket is missing (keeps the system moving forward)
        while len(wealth_elites) < elite_wealth:
            seed = self._create_random_strategy("WEALTH_SEED")
            seed["type"] = "wealth"
            wealth_elites.append(seed)

        while len(income_elites) < elite_income:
            seed = self._create_random_strategy("INCOME_SEED")
            seed["type"] = "income"
            income_elites.append(seed)

        next_gen: List[Dict] = []

        # Start with elites (elitism)
        next_gen.extend(wealth_elites)
        next_gen.extend(income_elites)

        # Fill Wealth quota via mutation
        while sum(1 for g in next_gen if is_wealth(g)) < target_wealth:
            parent = random.choice(wealth_elites)
            child = self.mutate(parent)
            child["type"] = "wealth"
            next_gen.append(child)

        # Fill Income quota via mutation
        while len(next_gen) < self.population_size:
            parent = random.choice(income_elites)
            child = self.mutate(parent)
            if not is_income(child):
                child["type"] = "income"
            next_gen.append(child)

        # Enforce exact population size (safety guard)
        next_gen = next_gen[: self.population_size]

        self.population = next_gen
        return next_gen
