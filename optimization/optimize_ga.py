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

# --- CONFIGURATION ---
POPULATION_SIZE = 50
GENERATIONS = 10
WORKERS = 8  # M3 Max Optimized

# The Genome Ranges
GENE_RANGES = {
    "gap_pct": (0.0, 3.0, 0.1),
    "volume_mult": (1.0, 3.0, 0.1),
    "adr_pct": (2.0, 5.0, 0.25),
    "stop_loss_atr": (1.5, 4.0, 0.25),
    "rs_rating": (80, 95, 1)
}

# The 3 Test Regimes (Regime-Weighted Sortino)
SLICES = {
    "A": {"start": "2008-01-01", "end": "2008-12-31"},  # Survival
    "B": {"start": "2015-01-01", "end": "2015-12-31"},  # Patience
    "C": {"start": "2020-01-01", "end": "2020-12-31"}   # Alpha
}

# Base Strategy Template
BASE_STRATEGY = {
    "name": "Apex GA Candidate",
    "type": "breakout",
    "entry_rules": [
        {"col": "clv", "op": ">", "val": 0.60},
        {"col": "close", "op": ">", "ref": "sma20"}
    ],
    "exit_rules": [{"col": "close", "op": "<", "ref": "sma20"}],
    "risk_parameters": {
        "stop_loss_type": "atr",
        "risk_per_trade": 0.02,
        "max_positions": 8,
        "max_pos_size_pct": 0.25
    },
    "execution_parameters": {"time_stop": 20, "partial_profit_day": 3}
}


def generate_random_genome():
    genome = {}
    for key, (min_val, max_val, step) in GENE_RANGES.items():
        if isinstance(step, int):
            val = random.randrange(min_val, max_val + step, step)
        else:
            steps = int((max_val - min_val) / step)
            val = min_val + (random.randint(0, steps) * step)
            val = round(val, 2)
        genome[key] = val
    return genome


def genome_to_strategy(genome):
    strat = copy.deepcopy(BASE_STRATEGY)
    strat["name"] = f"GA_G{genome['gap_pct']}_V{genome['volume_mult']}_A{genome['adr_pct']}"

    # Inject Genes
    if genome['gap_pct'] > 0:
        strat["entry_rules"].insert(0, {"col": "gap_pct", "op": ">", "val": genome['gap_pct']})

    strat["entry_rules"].append({"col": "volume", "op": ">", "ref": "vol_ma20", "mult": genome['volume_mult']})
    strat["entry_rules"].append({"col": "adr_pct", "op": ">", "val": genome['adr_pct']})
    strat["entry_rules"].append({"col": "rs_rating", "op": ">", "val": int(genome['rs_rating'])})
    strat["risk_parameters"]["stop_loss_atr"] = genome['stop_loss_atr']
    return strat


def evaluate_genome(genome, prepared_data, global_data):
    try:
        strat = GenericStrategy(genome_to_strategy(genome))

        # Helper to run a slice safely
        def run_slice(start, end):
            # KEY FIX: USE KEYWORD ARGUMENTS
            res = run_backtest(
                strategy=strat,
                data=prepared_data,
                start_cash=100000.0,
                start_date=start,
                end_date=end,
                global_data=global_data,
                scoring_weights=DEFAULT_SCORING_WEIGHTS
            )
            if isinstance(res, list) and res:
                return res[0]
            return None

        # Run 3 Slices
        res_a = run_slice(SLICES["A"]["start"], SLICES["A"]["end"])
        res_b = run_slice(SLICES["B"]["start"], SLICES["B"]["end"])
        res_c = run_slice(SLICES["C"]["start"], SLICES["C"]["end"])

        if not res_a or not res_b or not res_c:
            return -9999, {}  # Fail if any slice fails

        # Metrics
        dd_2008 = res_a['max_drawdown_pct'] * 100
        trades_2015 = res_a['total_trades']
        ret_2020 = ((res_c['final_value'] - 100000) / 100000) * 100

        # Scoring: High Alpha + Low 2008 Drawdown
        score = ret_2020 - (abs(dd_2008) * 2.5)

        # Penalties
        if dd_2008 < -35.0:
            score -= 1000  # Death Penalty
        if trades_2015 < 5:
            score -= 500  # Starvation Penalty

        metrics = {
            "2020_Ret": ret_2020,
            "2008_DD": dd_2008,
            "2015_Trades": trades_2015
        }
        return score, metrics

    except Exception:
        # print(f"Err: {e}") # Silent error to keep console clean
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
                if score <= -9000 or not metrics:
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
    min_v, max_v, step = GENE_RANGES[key]

    current = new[key]
    shift = random.choice([-step, step])
    new_val = max(min_v, min(max_v, current + shift))

    new[key] = round(new_val, 2)
    return new


def crossover(p1, p2):
    return {k: random.choice([p1[k], p2[k]]) for k in GENE_RANGES}


def main():
    print("🚀 Starting Apex Genetic Algorithm (FULL UNIVERSE MODE)")
    
    # 1. Load Russell 3000 from existing project cache
    cache_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cache_indices", "russell3000_iwv.json"))
    
    universe = []
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                universe = json.load(f)
            print(f"✅ Loaded {len(universe)} symbols from Russell 3000 cache.")
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}")
    
    # Fallback if cache is missing
    if not universe:
        print("⚠️ Cache not found. Using high-beta fallback...")
        universe = ["NVDA", "TSLA", "AAPL", "AMD", "META", "AMZN", "GOOGL", "MSFT", "NFLX", "ENPH", "SEDG", "SHOP", "SQ", "ROKU", "TDOC", "ZM", "PTON", "DKNG", "NET", "CRWD"]

    # 2. Fetch Data (Uses Existing Project Loader)
    # Loading 20 years of data for the full universe
    print(f"📉 Fetching data for {len(universe)} symbols (this may take a few minutes)...")
    
    data_map = fetch_data_pack(universe, days=7300, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=7300, backtest_mode=True)
    
    if not data_map:
        print("❌ Error: No data returned. Check Schwab Auth and Token.")
        return

    print("⚙️  Preparing Backtest Data (Injecting RS Ratings)...")
    prepared = prepare_backtest_data(data_map, None, None, g_data)
    
    # 3. Evolution Loop
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    best_score = -9999
    
    for gen in range(GENERATIONS):
        print(f"\n🧬 Generation {gen + 1}/{GENERATIONS}")
        results = evaluate_population(population, prepared, g_data)
        if not results: continue
        
        results.sort(key=lambda x: x[1], reverse=True)
        top_genome, top_score, top_ret, top_dd = results[0]
        
        print(f"   🏆 Best: Score {top_score:.2f} | 2020 Ret: {top_ret:.1f}% | DD: {top_dd:.1f}%")
        print(f"      {top_genome}")
        
        if top_score > best_score:
            best_score = top_score
            output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "best_genome.json"))
            with open(output_path, "w") as f:
                json.dump(top_genome, f, indent=4)
        
        # Breed
        survivors = [r[0] for r in results[:10]]
        new_pop = survivors[:]
        while len(new_pop) < POPULATION_SIZE:
            child = crossover(random.choice(survivors), random.choice(survivors))
            if random.random() < 0.2: child = mutate(child)
            new_pop.append(child)
        population = new_pop

    print("\n🏁 GA COMPLETE. Best parameters saved to config/best_genome.json")


if __name__ == "__main__":
    main()
