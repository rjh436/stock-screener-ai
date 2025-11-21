import random
import copy
import json
import os
from typing import List, Dict
from strategies.generic import GenericStrategy

# --- GENETIC BUILDING BLOCKS ---

# Flat list used by _random_indicator (kept for future use)
INDICATORS = [
    "close", "open", "high", "low", "volume",
    "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
    "ema5", "ema10", "ema20", "ema50", "ema100", "ema200",
    "rsi2", "rsi3", "rsi5", "rsi14",
    "atr10", "atr14", "atr20", "atr14_ma20",
    "bb_upper", "bb_lower", "bb_mid", "bb_width",
    "highest20", "highest55", "lowest5", "lowest20",
    "kc_mid", "kc_upper",
    "vol_ma20", "avgvol50",
    "macd", "macd_signal", "macd_hist",
    "adx", "plus_di", "minus_di",
    "stoch_k", "stoch_d",
    "cci", "mom", "roc"
]

OPERATORS = [">", "<"]

# Heuristic Templates to guide generation
TEMPLATES = [
    {
        "type": "trend_pullback",
        "entry_rules": [
            {"col": "close", "op": ">", "ref": "sma200"}, # Trend
            {"col": "rsi2", "op": "<", "val": 10}         # Pullback
        ],
        "exit_rules": [
            {"col": "close", "op": ">", "ref": "sma5"}    # Quick exit
        ]
    },
    {
        "type": "mean_reversion",
        "entry_rules": [
            {"col": "close", "op": "<", "ref": "bb_lower"}, # Oversold
            {"col": "rsi2", "op": "<", "val": 5}
        ],
        "exit_rules": [
            {"col": "close", "op": ">", "ref": "bb_mid"}    # Revert to mean
        ]
    },
    {
        "type": "volatility_breakout",
        "entry_rules": [
            {"col": "close", "op": ">", "ref": "highest20"},   # Breakout
            {"col": "volume", "op": ">", "ref": "vol_ma20"} # Volume confirm
        ],
        "exit_rules": [
            {"col": "close", "op": "<", "ref": "ema20"}     # Trend trailing
        ]
    },
    {
        "type": "momentum_macd",
        "entry_rules": [
            {"col": "macd_hist", "op": ">", "val": 0},      # Momentum turning up
            {"col": "adx", "op": ">", "val": 25}            # Strong trend
        ],
        "exit_rules": [
            {"col": "macd_hist", "op": "<", "val": 0}       # Momentum fading
        ]
    },
    {
        "type": "stoch_rsi_combo",
        "entry_rules": [
            {"col": "stoch_k", "op": "<", "val": 20},       # Oversold Stoch
            {"col": "rsi14", "op": "<", "val": 40}          # Weak RSI
        ],
        "exit_rules": [
            {"col": "stoch_k", "op": ">", "val": 80}        # Overbought Stoch
        ]
    }
]

class EvolutionEngine:
    def __init__(self):
        self.population_size = 50
        self.mutation_rate = 0.2
        self.generation_count = 0
        self.population = [] # List of genomes (dicts)
        # Indicator groups align with columns produced in execution/engine._compute_indicators
        self.INDICATORS = {
            "price": [
                "close", "open", "high", "low",
                "sma5", "sma10", "sma20", "sma50", "sma100", "sma200",
                "ema5", "ema10", "ema20", "ema50", "ema100", "ema200",
                "bb_mid", "bb_upper", "bb_lower",
                "highest20", "highest55", "lowest5", "lowest20",
                "kc_mid", "kc_upper",
            ],
            "oscillator": [
                "rsi2", "rsi3", "rsi5", "rsi14",
                "stoch_k", "stoch_d",
                "macd", "macd_signal", "macd_hist",
                "adx", "plus_di", "minus_di",
                "cci", "roc", "mom",
            ],
            "volatility": [
                "atr10", "atr14", "atr20", "atr14_ma20",
                "atr_pct10", "atr_pct20", "bb_width",
            ],
            "volume": [
                "volume", "vol_ma20", "avgvol50",
            ],
            "market_regime": [
                # VIX indicators (already calculated in engine.py)
                "vix",  # Current VIX level
                # Relative Strength indicators (already calculated in engine.py)
                "rs_ratio",  # Stock price / SPY price
                "rs_sma20",  # 20-day SMA of RS ratio
                "rs_trend",  # 1 if outperforming SPY, 0 otherwise
            ],
        }
        self.OPERATORS = OPERATORS

    def _random_indicator(self):
        return random.choice(INDICATORS)

    def _random_value(self, indicator):
        # Smart values based on indicator type
        if "rsi" in indicator or "stoch" in indicator:
            return random.randint(5, 95)
        if "atr" in indicator:
            return random.uniform(0.5, 3.0)
        if "adx" in indicator:
            return random.randint(15, 40)
        if "cci" in indicator:
            return random.randint(-200, 200)
        if "macd" in indicator:
            return random.uniform(-2.0, 2.0)
        if "roc" in indicator or "mom" in indicator:
            return random.uniform(-5.0, 5.0)
        # Market regime indicators
        if indicator == "vix":
            return random.randint(10, 50)  # VIX typically 10-40, occasionally higher
        if indicator == "rs_trend":
            return random.choice([0, 1])  # Binary: 0 = lagging, 1 = leading
        if indicator == "rs_ratio" or indicator == "rs_sma20":
            return round(random.uniform(0.8, 1.5), 2)  # Stock/SPY ratio
        return 0 # Fallback

    def _random_rule(self):
        """Generate a random rule with STRICT type safety."""
        # Pick a category first
        category = random.choice(list(self.INDICATORS.keys()))
        
        col = random.choice(self.INDICATORS[category])
        op = random.choice(self.OPERATORS)
        
        # STRICT MATCHING LOGIC
        if category == "oscillator":
            # Oscillators (RSI, Stoch) -> Compare to Oscillator OR Constant (0-100)
            if random.random() < 0.5:
                # Compare to another oscillator
                ref = random.choice(self.INDICATORS["oscillator"])
                while ref == col:
                    ref = random.choice(self.INDICATORS["oscillator"])
                return {"col": col, "op": op, "ref": ref}
            else:
                # Compare to constant
                val = random.randint(10, 90)
                if "cci" in col: val = random.randint(-150, 150)
                if "mom" in col or "roc" in col: val = random.uniform(-2, 2)
                return {"col": col, "op": op, "val": val}

        elif category == "price":
            # Price (Close, Open) -> ONLY compare to Price Averages (SMA, EMA, BB)
            # Never compare Price to a fixed number (e.g. Close > 100 is meaningless generally)
            ref = random.choice(self.INDICATORS["price"])
            while ref == col:
                ref = random.choice(self.INDICATORS["price"])
            return {"col": col, "op": op, "ref": ref}

        elif category == "volume":
            # Volume -> ONLY compare to Volume Averages
            ref = random.choice(self.INDICATORS["volume"])
            while ref == col:
                ref = random.choice(self.INDICATORS["volume"])
            return {"col": col, "op": op, "ref": ref}

        elif category == "volatility":
            # Volatility -> Compare to other volatility metrics or small constants
            if random.random() < 0.5:
                ref = random.choice(self.INDICATORS["volatility"])
                return {"col": col, "op": op, "ref": ref}
            else:
                val = random.uniform(0.5, 3.0)
                return {"col": col, "op": op, "val": val}

        elif category == "market_regime":
             # Regime -> Compare to constants usually
            if "vix" in col:
                val = random.randint(15, 35)
                return {"col": col, "op": op, "val": val}
            elif "rs_trend" in col:
                val = 0 # 0 or 1
                return {"col": col, "op": op, "val": val}
            else:
                val = 1.0
                return {"col": col, "op": op, "val": val}

        return {"col": col, "op": op, "val": 0}

    def _create_random_strategy(self, name_prefix="Random") -> Dict:
        """Create a completely new random strategy."""
        genome = {
            "name": f"{name_prefix}_{random.randint(1000,9999)}",
            "type": "random",
            "entry_rules": [self._random_rule() for _ in range(random.randint(1, 3))],
            "exit_rules": [self._random_rule() for _ in range(random.randint(1, 2))],
            "stop_loss_atr": round(random.uniform(1.0, 4.0), 1),
            "time_stop": random.choice([5, 10, 15, 20, 30])
        }
        return genome

    def generate_initial_population(self):
        """Create initial strategies based on templates with random variations."""
        pop = []
        for i in range(self.population_size):
            template = random.choice(TEMPLATES)
            genome = copy.deepcopy(template)
            genome["name"] = f"Gen0_Strat{i}"
            
            # Randomize parameters slightly
            genome["stop_loss_atr"] = round(random.uniform(1.0, 4.0), 1)
            genome["time_stop"] = random.choice([5, 10, 15, 20, 30])
            
            # Tweak values in rules
            for rule in genome["entry_rules"]:
                if "val" in rule:
                    # Jitter the value by +/- 20%
                    val = rule["val"]
                    jitter = val * 0.2
                    rule["val"] = round(val + random.uniform(-jitter, jitter), 1)
            
            pop.append(genome)
        self.population = pop
        return pop

    def mutate(self, genome: Dict) -> Dict:
        """Randomly alter a strategy."""
        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant["name"] + "_mut"
        
        r = random.random()
        
        # Mutation Type 1: Change Stop Loss (30% chance)
        if r < 0.3:
            mutant["stop_loss_atr"] = round(random.uniform(1.0, 4.0), 1)
            
        # Mutation Type 2: Change Time Stop (30% chance)
        elif r < 0.6:
            mutant["time_stop"] = random.choice([5, 10, 15, 20, 30])
            
        # Mutation Type 3: Modify a Rule Value (40% chance)
        else:
            if mutant["entry_rules"]:
                rule = random.choice(mutant["entry_rules"])
                if "val" in rule:
                    val = rule["val"]
                    # +/- 10% change
                    rule["val"] = round(val * random.uniform(0.9, 1.1), 1)
                elif "ref" in rule:
                    # Swap reference
                    # Use price indicators as safe alternatives for moving averages
                    alternatives = self.INDICATORS["price"]
                    if alternatives:
                        rule["ref"] = random.choice(alternatives)

            # Mutation Type 4: Add/Replace with New Random Rule (20% chance)
            if random.random() < 0.2:
                if random.random() < 0.5 and mutant["entry_rules"]:
                    mutant["entry_rules"][random.randint(0, len(mutant["entry_rules"])-1)] = self._random_rule()
                else:
                    mutant["entry_rules"].append(self._random_rule())

        return mutant

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Combine two strategies."""
        child = {
            "name": f"Cross_{parent1['name']}_{parent2['name']}",
            "type": "hybrid",
            # Inherit entry from P1, exit from P2
            "entry_rules": copy.deepcopy(parent1["entry_rules"]),
            "exit_rules": copy.deepcopy(parent2["exit_rules"]),
            # Average the risk params
            "stop_loss_atr": round((parent1["stop_loss_atr"] + parent2["stop_loss_atr"]) / 2, 1),
            "time_stop": int((parent1["time_stop"] + parent2["time_stop"]) / 2)
        }
        return child

    def evolve(self, ranked_results: List[Dict]):
        """
        Create the next generation based on performance results.
        ranked_results: List of dicts with 'genome' and 'pnl_pct' keys, sorted by performance.
        """
        self.generation_count += 1
        print(f"Creating Generation {self.generation_count}...")
        
        # 1. Elitism: Keep top 3
        survivors = [r["genome"] for r in ranked_results[:3]]
        for s in survivors:
            s["name"] = s["name"].split("_gen")[0] + f"_gen{self.generation_count}" # Update gen tag

        next_gen = survivors[:]
        
        # 2. Crossover: Breed top performers to fill 4 spots
        while len(next_gen) < 7:
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            if p1 != p2:
                child = self.crossover(p1, p2)
                next_gen.append(child)
                
        # 3. Mutation / New Blood: Fill remaining spots
        while len(next_gen) < self.population_size:
            # 50% chance of mutation of a survivor, 50% fresh random
            if random.random() < 0.5:
                parent = random.choice(survivors)
                child = self.mutate(parent)
                next_gen.append(child)
            else:
                # Fresh blood - Random Strategy
                child = self._create_random_strategy(f"Gen{self.generation_count}_Fresh")
                next_gen.append(child)
                
        self.population = next_gen
        return next_gen
