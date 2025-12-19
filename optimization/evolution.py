from __future__ import annotations

import copy
import random
from typing import Dict, Iterable, List, Optional, Sequence


_WEALTH_TYPE = "wealth"
_INCOME_TYPES = {"income", "hybrid"}

# Exit parameter ranges (Sniper mode)
_PROFIT_TARGET_MIN = 1.05
_PROFIT_TARGET_MAX = 1.30
_TIME_STOP_MIN = 20
_TIME_STOP_MAX = 90
_TRAIL_ACTIVATION_MIN = 1.01
_TRAIL_ACTIVATION_MAX = 1.10

# Scoring weight ranges (higher floors to prevent "low standards" loopholes)
_SCORING_RANGES: Dict[str, tuple[float, float]] = {
    "sniper_bonus": (40.0, 100.0),
    "rsi_factor": (2.5, 4.5),
    "trend_bonus": (30.0, 70.0),
    "vol_bonus": (8.0, 30.0),
    "atr_high_bonus": (12.0, 40.0),
    "atr_med_bonus": (4.0, 20.0),
    "trend_penalty": (-40.0, -10.0),
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class EvolutionEngine:
    """
    Quota-aware GA engine.

    Hard constraints:
    - Population is always 50/50 split:
      - Wealth species: `type == "wealth"`
      - Income species: `type in {"income", "hybrid"}`

    Soft constraints:
    - Children inherit their parent's lineage tag (`income` vs `hybrid`).
    - Mutation scaling is adaptive:
      - Gen < 25: exploration (0.50–1.50)
      - Gen >= 25: refinement  (0.85–1.15)
    """

    def __init__(self) -> None:
        self.population_size: int = 100
        self.mutation_rate: float = 0.2
        self.generation_count: int = 0
        self.population: List[Dict] = []

        self.INDICATORS = {
            "price": ["close", "sma50", "sma200", "bb_lower"],
            "momentum": ["rs_trend", "rs_ratio", "cci"],
            "oscillator": ["rsi2", "rsi14", "adx", "stoch_k"],
            "volatility": ["atr14", "bb_width"],
            "volume": ["volume", "vol_ma20"],
        }
        self.OPERATORS = [">", "<"]

    @staticmethod
    def _normalize_type(raw_type: Optional[str], *, default: str = "income") -> str:
        t = str(raw_type or "").strip().lower()
        if t == _WEALTH_TYPE:
            return _WEALTH_TYPE
        if t in _INCOME_TYPES:
            return t
        return default

    @classmethod
    def _is_wealth(cls, genome: Dict) -> bool:
        return cls._normalize_type(genome.get("type"), default="income") == _WEALTH_TYPE

    @classmethod
    def _is_income_species(cls, genome: Dict) -> bool:
        return cls._normalize_type(genome.get("type"), default="income") in _INCOME_TYPES

    def _random_rule(self) -> Dict:
        cat = random.choice(list(self.INDICATORS.keys()))
        col = random.choice(self.INDICATORS[cat])
        op = random.choice(self.OPERATORS)

        val = 0
        if "rsi" in col:
            val = random.randint(5, 95)
        elif "vix" in col:
            val = random.randint(15, 40)
        elif "adx" in col:
            val = random.randint(20, 40)
        elif "bb_width" in col:
            val = round(random.uniform(0.10, 0.50), 2)
        elif "cci" in col:
            val = random.choice([-100, 0, 100])

        return {"col": col, "op": op, "val": val}

    def _random_scoring_weights(self) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        for key, (lo, hi) in _SCORING_RANGES.items():
            weights[key] = round(random.uniform(lo, hi), 2)
        return weights

    @staticmethod
    def _stable_rand_suffix() -> str:
        return f"{random.randint(1000, 9999)}"

    def _ensure_schema(self, genome: Dict, *, default_type: str) -> Dict:
        g = copy.deepcopy(genome) if isinstance(genome, dict) else {}

        g["type"] = self._normalize_type(g.get("type"), default=default_type)
        g["name"] = str(g.get("name") or f"Strategy_{self._stable_rand_suffix()}")

        entry_rules = g.get("entry_rules")
        if not isinstance(entry_rules, list):
            g["entry_rules"] = []

        exit_rules = g.get("exit_rules")
        if not isinstance(exit_rules, list) or not exit_rules:
            exit_rules = [{"type": "profit_target", "val": 1.08}]

        normalized_exit_rules: List[Dict] = []
        has_profit_target = False
        for rule in exit_rules:
            if not isinstance(rule, dict):
                continue
            r = dict(rule)
            if r.get("type") == "profit_target":
                has_profit_target = True
                try:
                    val = float(r.get("val", 1.08) or 1.08)
                except (TypeError, ValueError):
                    val = 1.08
                r["val"] = round(_clamp(val, _PROFIT_TARGET_MIN, _PROFIT_TARGET_MAX), 2)
            normalized_exit_rules.append(r)

        if not has_profit_target:
            normalized_exit_rules.append({"type": "profit_target", "val": 1.08})

        g["exit_rules"] = normalized_exit_rules

        g["stop_loss_atr"] = float(g.get("stop_loss_atr", 5.0) or 5.0)
        try:
            time_stop = int(g.get("time_stop", 45) or 45)
        except Exception:
            time_stop = 45
        g["time_stop"] = int(_clamp(float(time_stop), float(_TIME_STOP_MIN), float(_TIME_STOP_MAX)))

        scoring = g.get("scoring_weights")
        if not isinstance(scoring, dict) or not scoring:
            scoring = self._random_scoring_weights()
        else:
            scoring = copy.deepcopy(scoring)

        # Enforce minimum floors / ranges for core scoring weights
        for key, (lo, hi) in _SCORING_RANGES.items():
            try:
                cur = float(scoring.get(key, random.uniform(lo, hi)))
            except (TypeError, ValueError):
                cur = random.uniform(lo, hi)
            scoring[key] = round(_clamp(cur, lo, hi), 2)

        g["scoring_weights"] = scoring

        # GEN 16.1 PARAMETERS (backward compatible defaults)
        g["adx_threshold"] = float(g.get("adx_threshold", 25.0) or 25.0)
        g["rsi_strong"] = float(g.get("rsi_strong", 40.0) or 40.0)
        g["rsi_weak"] = float(g.get("rsi_weak", 15.0) or 15.0)
        g["limit_ratio"] = float(g.get("limit_ratio", 0.98) or 0.98)
        g["trail_atr"] = float(g.get("trail_atr", 3.0) or 3.0)
        try:
            trail_act = float(g.get("trail_activation", 1.03) or 1.03)
        except (TypeError, ValueError):
            trail_act = 1.03
        g["trail_activation"] = round(_clamp(trail_act, _TRAIL_ACTIVATION_MIN, _TRAIL_ACTIVATION_MAX), 3)

        return g

    def _create_random_strategy(self, *, name_prefix: str, strategy_type: str) -> Dict:
        strategy_type_norm = self._normalize_type(strategy_type, default="income")

        genome = {
            "name": f"{name_prefix}_{self._stable_rand_suffix()}",
            "type": strategy_type_norm,
            "entry_rules": [self._random_rule() for _ in range(random.randint(2, 4))],
            "exit_rules": [{"type": "profit_target", "val": round(random.uniform(_PROFIT_TARGET_MIN, _PROFIT_TARGET_MAX), 2)}],
            "stop_loss_atr": round(random.uniform(3.0, 7.0), 1),
            "time_stop": random.randint(_TIME_STOP_MIN, _TIME_STOP_MAX),
            "scoring_weights": self._random_scoring_weights(),
            # GEN 16.1 PARAMETERS
            "adx_threshold": 25.0,
            "rsi_strong": 40.0,
            "rsi_weak": 15.0,
            "limit_ratio": 0.98,
            "trail_atr": 3.0,
            "trail_activation": round(random.uniform(_TRAIL_ACTIVATION_MIN, _TRAIL_ACTIVATION_MAX), 3),
        }
        return genome

    def create_initial_population(self, base_name: str = "Strategy", base_strategies: Optional[Sequence[Dict]] = None) -> List[Dict]:
        """
        Gen0 seeding:
        - If base strategies exist but are fewer than `population_size`, expand them to
          a full population by mutation (not by shrinking evaluation to a tiny set).
        - Always enforces 50/50 Wealth vs Income species.
        """
        target_wealth = self.population_size // 2
        target_income = self.population_size - target_wealth

        wealth_seeds: List[Dict] = []
        income_seeds: List[Dict] = []

        for s in base_strategies or []:
            if not isinstance(s, dict):
                continue
            t_norm = self._normalize_type(s.get("type"), default="income")
            seeded = self._ensure_schema(s, default_type=t_norm)
            if self._is_wealth(seeded):
                wealth_seeds.append(seeded)
            else:
                income_seeds.append(seeded)

        if not wealth_seeds:
            wealth_seeds.append(self._create_random_strategy(name_prefix=f"{base_name}_WEALTH_SEED", strategy_type=_WEALTH_TYPE))
        if not income_seeds:
            income_seeds.append(self._create_random_strategy(name_prefix=f"{base_name}_INCOME_SEED", strategy_type="income"))

        wealth_pop: List[Dict] = []
        income_pop: List[Dict] = []

        # Keep the provided parents (up to quota) as exact clones in Gen0.
        wealth_pop.extend(copy.deepcopy(wealth_seeds[:target_wealth]))
        income_pop.extend(copy.deepcopy(income_seeds[:target_income]))

        # Fill remaining slots via mutation (seed expansion).
        while len(wealth_pop) < target_wealth:
            parent = random.choice(wealth_seeds)
            child = self.mutate(parent)
            child["type"] = _WEALTH_TYPE
            wealth_pop.append(child)

        while len(income_pop) < target_income:
            parent = random.choice(income_seeds)
            child = self.mutate(parent)
            parent_type = self._normalize_type(parent.get("type"), default="income")
            child["type"] = parent_type if parent_type in _INCOME_TYPES else "income"
            income_pop.append(child)

        pop = wealth_pop[:target_wealth] + income_pop[:target_income]
        random.shuffle(pop)

        self.population = pop
        return pop

    def generate_initial_population(self, base_strategies: Optional[Sequence[Dict]] = None, base_name: str = "Strategy") -> List[Dict]:
        return self.create_initial_population(base_name=base_name, base_strategies=base_strategies)

    def mutate(self, genome: Dict) -> Dict:
        parent_type = self._normalize_type(genome.get("type"), default="income")
        mutant = self._ensure_schema(genome, default_type=parent_type)

        base_name = str(mutant.get("name") or "Strategy")
        mutant["name"] = f"{base_name}_g{self.generation_count}_m{self._stable_rand_suffix()}"

        r = random.random()

        if r < 0.25:
            trait = random.choice(["limit", "trail", "dual_lane"])
            if trait == "limit":
                mutant["limit_ratio"] = round(random.uniform(0.95, 1.00), 3)
            elif trait == "trail":
                mutant["trail_atr"] = round(random.uniform(2.5, 6.0), 1)
            else:
                sub_trait = random.choice(["adx", "strong", "weak"])
                if sub_trait == "adx":
                    mutant["adx_threshold"] = float(random.randint(15, 35))
                elif sub_trait == "strong":
                    mutant["rsi_strong"] = float(random.randint(30, 55))
                else:
                    mutant["rsi_weak"] = float(random.randint(5, 25))

        elif r < 0.45:
            # Exit knobs (sniper selection)
            trait = random.choice(["profit_target", "time_stop", "trail_activation"])

            if trait == "profit_target":
                exit_rules = mutant.get("exit_rules")
                if not isinstance(exit_rules, list):
                    exit_rules = []

                pt_rule = None
                for rule in exit_rules:
                    if isinstance(rule, dict) and rule.get("type") == "profit_target":
                        pt_rule = rule
                        break
                if pt_rule is None:
                    pt_rule = {"type": "profit_target", "val": 1.08}
                    exit_rules.append(pt_rule)

                if self.generation_count < 25:
                    new_val = random.uniform(_PROFIT_TARGET_MIN, _PROFIT_TARGET_MAX)
                else:
                    try:
                        cur = float(pt_rule.get("val", 1.08) or 1.08)
                    except (TypeError, ValueError):
                        cur = 1.08
                    new_val = cur * random.uniform(0.95, 1.05)

                pt_rule["val"] = round(_clamp(float(new_val), _PROFIT_TARGET_MIN, _PROFIT_TARGET_MAX), 2)
                mutant["exit_rules"] = exit_rules

            elif trait == "time_stop":
                if self.generation_count < 25:
                    mutant["time_stop"] = random.randint(_TIME_STOP_MIN, _TIME_STOP_MAX)
                else:
                    try:
                        cur = int(mutant.get("time_stop", 45) or 45)
                    except Exception:
                        cur = 45
                    new_val = int(round(cur * random.uniform(0.85, 1.15)))
                    mutant["time_stop"] = max(_TIME_STOP_MIN, min(_TIME_STOP_MAX, new_val))

            else:
                # trail_activation
                if self.generation_count < 25:
                    new_val = random.uniform(_TRAIL_ACTIVATION_MIN, _TRAIL_ACTIVATION_MAX)
                else:
                    try:
                        cur = float(mutant.get("trail_activation", 1.03) or 1.03)
                    except (TypeError, ValueError):
                        cur = 1.03
                    new_val = cur + random.uniform(-0.01, 0.01)
                mutant["trail_activation"] = round(_clamp(float(new_val), _TRAIL_ACTIVATION_MIN, _TRAIL_ACTIVATION_MAX), 3)

        elif r < 0.60:
            mutant["stop_loss_atr"] = round(random.uniform(2.5, 6.0), 1)

        else:
            scoring = copy.deepcopy(mutant.get("scoring_weights") or {})
            if not scoring:
                scoring = self._random_scoring_weights()

            key = random.choice(list(scoring.keys()))

            if self.generation_count < 25:
                scale_low, scale_high = 0.50, 1.50
            else:
                scale_low, scale_high = 0.85, 1.15

            if random.random() < 0.15:
                lo, hi = _SCORING_RANGES.get(key, (5.0, 80.0))
                scoring[key] = round(random.uniform(lo, hi), 2)
            else:
                try:
                    cur = float(scoring.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    cur = 0.0
                new_val = float(cur) * random.uniform(scale_low, scale_high)
                lo, hi = _SCORING_RANGES.get(key, (0.0, 100.0))
                scoring[key] = round(_clamp(new_val, lo, hi), 2)

            mutant["scoring_weights"] = scoring

        mutant["type"] = parent_type
        return mutant

    def evolve(self, ranked: Optional[Sequence[Dict]]) -> List[Dict]:
        """
        Quota elitism with lineage-correct children.
        """
        self.generation_count += 1

        target_wealth = self.population_size // 2
        target_income = self.population_size - target_wealth

        elite_total = min(10, self.population_size)
        elite_wealth = min(target_wealth, elite_total // 2)
        elite_income = min(target_income, elite_total - elite_wealth)

        wealth_elites: List[Dict] = []
        income_elites: List[Dict] = []

        for r in ranked or []:
            genome = r.get("genome") if isinstance(r, dict) else None
            if not isinstance(genome, dict):
                continue

            t_norm = self._normalize_type(genome.get("type"), default="income")
            g = self._ensure_schema(genome, default_type=t_norm)

            if self._is_wealth(g):
                if len(wealth_elites) < elite_wealth:
                    elite = copy.deepcopy(g)
                    elite["type"] = _WEALTH_TYPE
                    wealth_elites.append(elite)
            else:
                if len(income_elites) < elite_income:
                    elite = copy.deepcopy(g)
                    elite_type = self._normalize_type(elite.get("type"), default="income")
                    elite["type"] = elite_type if elite_type in _INCOME_TYPES else "income"
                    income_elites.append(elite)

            if len(wealth_elites) >= elite_wealth and len(income_elites) >= elite_income:
                break

        while len(wealth_elites) < elite_wealth:
            wealth_elites.append(self._create_random_strategy(name_prefix="WEALTH_SEED", strategy_type=_WEALTH_TYPE))

        while len(income_elites) < elite_income:
            income_elites.append(self._create_random_strategy(name_prefix="INCOME_SEED", strategy_type="income"))

        wealth_next: List[Dict] = [copy.deepcopy(g) for g in wealth_elites]
        income_next: List[Dict] = [copy.deepcopy(g) for g in income_elites]

        while len(wealth_next) < target_wealth:
            parent = random.choice(wealth_elites)
            child = self.mutate(parent)
            child["type"] = _WEALTH_TYPE
            wealth_next.append(child)

        while len(income_next) < target_income:
            parent = random.choice(income_elites)
            child = self.mutate(parent)
            parent_type = self._normalize_type(parent.get("type"), default="income")
            child["type"] = parent_type if parent_type in _INCOME_TYPES else "income"
            income_next.append(child)

        next_gen = wealth_next[:target_wealth] + income_next[:target_income]
        random.shuffle(next_gen)

        self.population = next_gen
        return next_gen
