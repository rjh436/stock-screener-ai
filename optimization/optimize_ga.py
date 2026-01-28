from __future__ import annotations
import os
import sys
import random
import json
import copy
import traceback
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. Setup Project Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import fetch_data_pack
from execution.engine import DEFAULT_SCORING_WEIGHTS, prepare_backtest_data, run_backtest
from strategies.generic import GenericStrategy

# --- CONFIGURATION (M3 MAX OPTIMIZED) ---
POPULATION_SIZE = 50
GENERATIONS = 50
WORKERS = 10  # Increased for M3 Max 12-core

# AUDIT-ALIGNED GENOME (SEPA V18)
# We are optimizing the 'Risk Kernel' while the logic is hard-coded in the engine.
GENE_RANGES = {
    "rs_rating": (85, 99, 1),            # Minervini Proxy (High RS)
    "bb_width_max": (0.10, 0.20, 0.01),  # VCP Tightness (0.15 standard)
    "risk_per_trade": (0.005, 0.015, 0.001), # Conservative (0.5% - 1.5%)
    "max_pos_size_pct": (0.20, 0.30, 0.05),  # Hard Cap (20% - 30%)
    "stop_loss_type": ["low_of_day", "atr"], # LOD vs ATR
    "stop_loss_atr": (1.5, 3.5, 0.25),   # Buffer if ATR used
    "partial_profit_day": (3, 5, 1)      # Fast profit taking
}

# The 3 Test Regimes (Regime-Weighted Sortino)
SLICES = {
    "A": {"start": "2008-01-01", "end": "2008-12-31"},  # Survival (GFC)
    "B": {"start": "2015-01-01", "end": "2015-12-31"},  # Patience (Chop)
    "C": {"start": "2020-04-01", "end": "2021-02-01"}   # Alpha (Post-Covid Run)
}

# SEPA V18 Hybrid Template
BASE_STRATEGY = {
    "name": "Apex SEPA V18",
    "type": "breakout",
    "entry_rules": [
        # Hard filters are now in engine.py (Trend Template).
        # We just reinforce the genome here.
    ],
    "exit_rules": [{"col": "close", "op": "<", "ref": "sma10"}], # Qullamaggie Trail
    "risk_parameters": {
        "max_positions": 10
    },
    "execution_parameters": {"time_stop": 30}
}


def generate_random_genome():
    genome = {}
    for key, range_def in GENE_RANGES.items():
        if isinstance(range_def, list):
            genome[key] = random.choice(range_def)
        else:
            min_val, max_val, step = range_def
            if isinstance(step, int):
                val = random.randrange(min_val, max_val + step, step)
            else:
                steps = int((max_val - min_val) / step)
                val = min_val + (random.randint(0, steps) * step)
                val = round(val, 3)
            genome[key] = val
    return genome


def genome_to_strategy(genome):
    strat = copy.deepcopy(BASE_STRATEGY)
    strat["name"] = f"SEPA_RS{int(genome['rs_rating'])}_Risk{genome['risk_per_trade']}"

    # Inject Genes into Strategy Config
    strat["entry_rules"] = [
        {"col": "rs_rating", "op": ">", "val": int(genome['rs_rating'])},
        {"col": "bb_width", "op": "<", "val": genome['bb_width_max']},
        {"col": "close", "op": ">", "ref": "sma50"} # Reinforce Trend
    ]

    strat["risk_parameters"]["risk_per_trade"] = genome['risk_per_trade']
    strat["risk_parameters"]["max_pos_size_pct"] = genome['max_pos_size_pct']
    strat["risk_parameters"]["stop_loss_type"] = genome['stop_loss_type']
    strat["risk_parameters"]["stop_loss_atr"] = genome['stop_loss_atr']
    strat["execution_parameters"]["partial_profit_day"] = int(genome['partial_profit_day'])

    return strat


def evaluate_genome(genome, prepared_data, global_data):
    try:
        strat = GenericStrategy(genome_to_strategy(genome))

        def run_slice(start, end):
            res = run_backtest(
                strategy=strat,
                data=prepared_data,
                start_cash=100000.0,
                start_date=start,
                end_date=end,
                global_data=global_data,
                scoring_weights=DEFAULT_SCORING_WEIGHTS
            )
            # --- BUG FIX: HANDLE DICT RETURN ---
            # Engine returns a dict for single-strategy runs, but GA expected a list.
            if isinstance(res, dict):
                return res
            # -----------------------------------

            if isinstance(res, list) and res:
                return res[0]

            # AUDIT FIX: Don't fail on None, return empty result
            return {"total_trades": 0, "final_value": 100000.0, "max_drawdown_pct": 0.0}

        # Run 3 Slices
        res_a = run_slice(SLICES["A"]["start"], SLICES["A"]["end"])
        res_b = run_slice(SLICES["B"]["start"], SLICES["B"]["end"])
        res_c = run_slice(SLICES["C"]["start"], SLICES["C"]["end"])

        # Metrics Extraction
        dd_2008 = res_a['max_drawdown_pct'] * 100
        trades_2015 = res_b['total_trades'] # AUDIT FIX: Corrected from res_a to res_b
        ret_2020 = ((res_c['final_value'] - 100000) / 100000) * 100

        metrics = {
            "2020_Ret": ret_2020,
            "2008_DD": dd_2008,
            "2015_Trades": trades_2015
        }

        total_trades = res_a.get("total_trades", 0) + res_b.get("total_trades", 0) + res_c.get("total_trades", 0)
        if total_trades == 0:
            return 0.0, metrics

        # Scoring Logic:
        # 1. Survival: 2008 DD must be < 20%
        # 2. Patience: 2015 should have few trades OR break-even
        # 3. Alpha: 2020 should rip (> 50%)
        score = ret_2020

        # Penalties
        if dd_2008 < -20.0:
            score -= (abs(dd_2008) * 10) # Heavy penalty for drawdown

        # 2. PATIENCE PENALTY (Gradient) [AUDIT FIX APPLIED]
        # Previous hard cliff (-200) replaced with progressive penalty.
        # This rewards the GA for reducing trades from 100 -> 80 -> 60 -> 50.
        if trades_2015 > 50 and res_b['final_value'] < 100000:
            excess_trades = trades_2015 - 50
            # Formula: Base Penalty (50) + (4 points per excess trade)
            # Example: 51 trades = -54 penalty
            # Example: 100 trades = -250 penalty (High deterrent)
            penalty = 50.0 + (excess_trades * 4.0)
            score -= penalty

        # AUDIT FIX: Zero trades in 2008 is GOOD (Defensive).
        # Zero trades in 2020 is BAD.
        if res_c['total_trades'] < 5:
            score -= 500

        return score, metrics

    except Exception:
        return -9999, {}


def evaluate_population(population, prepared_data, global_data):
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        future_to_genome = {
            executor.submit(evaluate_genome, genome, prepared_data, global_data): genome
            for genome in population
        }
        for future in as_completed(future_to_genome):
            genome = future_to_genome[future]
            try:
                score, metrics = future.result()
                if score <= -9000:
                    continue
                results.append(
                    (
                        genome,
                        score,
                        metrics.get("2020_Ret"),
                        metrics.get("2008_DD"),
                    )
                )
            except Exception:
                continue
    return results


def mutate(genome):
    new = genome.copy()
    key = random.choice(list(GENE_RANGES.keys()))
    range_def = GENE_RANGES[key]

    if isinstance(range_def, list):
        new[key] = random.choice(range_def)
    else:
        min_v, max_v, step = range_def
        current = new[key]
        shift = random.choice([-step, step])
        new_val = max(min_v, min(max_v, current + shift))
        new[key] = round(new_val, 3)

    return new


def crossover(p1, p2):
    child = {}
    for k in GENE_RANGES:
        child[k] = random.choice([p1[k], p2[k]])
    return child


def main():
    print("🚀 Starting Apex SEPA V18 Optimization (M3 Max Mode)")

    # 1. Load Russell 3000
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cache_indices", "russell3000_iwv.json"))

    universe = []
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            universe = json.load(f)
        print(f"✅ Loaded {len(universe)} symbols from cache.")
    else:
        # Fallback Universe
        universe = ["NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "ENPH"]

    # 2. Fetch Data
    print(f"📉 Fetching data for {len(universe)} symbols...")
    data_map = fetch_data_pack(universe, days=7300, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=7300, backtest_mode=True)

    if not data_map:
        print("❌ Error: No data returned.")
        return

    print("⚙️  Preparing Backtest Data (Injecting SEPA Metrics)...")
    prepared = prepare_backtest_data(data_map, None, None, g_data)

    # 3. Evolution Loop
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    best_score = -9999

    for gen in range(GENERATIONS):
        print(f"\n🧬 Generation {gen + 1}/{GENERATIONS}")
        results = evaluate_population(population, prepared, g_data)
        if not results:
            continue

        results.sort(key=lambda x: x[1], reverse=True)
        top_genome, top_score, top_ret, top_dd = results[0]

        print(f"   🏆 Best: Score {top_score:.2f} | 2020 Ret: {top_ret:.1f}% | 2008 DD: {top_dd:.1f}%")
        print(f"      RS: {top_genome['rs_rating']} | Risk: {top_genome['risk_per_trade']} | Stop: {top_genome['stop_loss_type']}")

        if top_score > best_score:
            best_score = top_score
            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "best_sepa_genome.json"))
            with open(output_path, "w") as f:
                json.dump(top_genome, f, indent=4)

        # Breed
        survivors = [r[0] for r in results[:10]]
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            child = crossover(random.choice(survivors), random.choice(survivors))
            if random.random() < 0.2:
                child = mutate(child)
            new_pop.append(child)
        population = new_pop

    print("\n🏁 SEPA GA COMPLETE.")


if __name__ == "__main__":
    main()
