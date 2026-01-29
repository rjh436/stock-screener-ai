
import sys
import os
import json
import random
import copy
import multiprocessing
import pandas as pd
import numpy as np
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

# --- GENETIC CONFIG ---
POPULATION_SIZE = 20
GENERATIONS = 10
ELITE_SIZE = 5
MUTATION_RATE = 0.3
NUM_PROCESSES = min(multiprocessing.cpu_count(), 8)

# --- GENOME BOUNDS ---
GENE_RANGES = {
    "rs_floor": (85.0, 99.0),          # Float
    "vol_multiplier": (1.2, 4.0),     # Float
    "adx_threshold": (15, 40),        # Int
    "max_positions": [4, 5, 6, 8, 10],# Choice
    "stop_loss_atr": (2.0, 5.0),      # Float
    "take_profit_r_multiple": (2.0, 6.0), # Float
    "use_trailing_stop": [True, False]# Choice
}

def random_gene(name):
    bounds = GENE_RANGES[name]
    if isinstance(bounds, list):
        return random.choice(bounds)
    if name == "adx_threshold":
        return random.randint(bounds[0], bounds[1])
    return round(random.uniform(bounds[0], bounds[1]), 2)

def create_random_genome():
    return {k: random_gene(k) for k in GENE_RANGES}

def mutate(genome):
    mutant = genome.copy()
    for gene in genome:
        if random.random() < MUTATION_RATE:
            mutant[gene] = random_gene(gene)
    return mutant

def crossover(parent1, parent2):
    child = {}
    for gene in parent1:
        child[gene] = parent1[gene] if random.random() > 0.5 else parent2[gene]
    return child

def genome_to_config(genome):
    # Base Config
    config = {
        "name": "Apex GA Candidate",
        "use_fundamentals": False,
        "entry_rules": [
             {"col": "rs_rating", "op": ">", "val": genome["rs_floor"]}, 
             {"col": "bb_width", "op": "<", "val": 0.25}, # Keep VCP fixed
             {"col": "close", "op": ">", "ref": "sma50"},
             {"col": "close", "op": ">", "ref": "high_20_prev", "val": 0.99}
        ],
        "risk_parameters": {
             "stop_loss_type": "atr",
             "stop_loss_atr": genome["stop_loss_atr"],
             "max_positions": genome["max_positions"],
             "max_pos_size_pct": min(1.0, (1.0 / genome["max_positions"]) * 1.1)
        },
        "execution_parameters": {
             "time_stop": 30,
             "partial_profit_day": 5,
             "partial_profit_r": genome["take_profit_r_multiple"],
             "vol_mult": genome["vol_multiplier"],
             "rs_floor": genome["rs_floor"],
             "adx_min": float(genome["adx_threshold"]),
             "use_trailing_stop": genome["use_trailing_stop"]
        },
        "exit_rules": [{"col": "close", "op": "<", "ref": "sma10"}]
    }
    return config

def evaluate_genome(genome, prepared_data, g_data):
    try:
        config = genome_to_config(genome)
        strat = SEPAChampionStrategy(config)
        res = run_backtest(
            strat,
            data=prepared_data,
            start_cash=100000.0,
            start_date="2015-01-01",
            end_date="2021-01-01",
            global_data=g_data
        )
        if not res: return 0.0, 0.0, 0, {}
        res = res if isinstance(res, dict) else res[0]
        
        cagr = res.get("cagr", 0.0) * 100
        dd = res.get("max_drawdown_pct", 1.0) * 100
        trades = res.get("total_trades", 0)
        
        # Fitness Function
        score = (cagr * 2.0) - dd
        
        # Constraints
        if trades < 50: score = 0.0
        
        return score, cagr, dd, trades, res
    except Exception as e:
        return 0.0, 0.0, 0, 0, {}

def main():
    print("🧬 Starting Genetic Optimization (Phase 6)...")
    
    # Load Data (Memoized by engine logic, but loaded here for passing)
    print("📦 Loading Data Pack...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=252*7, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=252*7, backtest_mode=True) or {}
    print("⚙️  Preparing Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)

    # Init Population
    population = [create_random_genome() for _ in range(POPULATION_SIZE)]
    best_overall_genome = None
    best_overall_score = -9999.0

    for gen in range(GENERATIONS):
        print(f"\n🧬 Generation {gen+1}/{GENERATIONS}")
        
        scored_pop = []
        for i, genome in enumerate(population):
            # Sequential for simplicity unless we want to deal with pickling huge data
            score, cagr, dd, tr, _ = evaluate_genome(genome, prepared, g_data)
            scored_pop.append((score, genome, cagr, dd, tr))
            if gen % 1 == 0 and i % 5 == 0: 
                print(f"   Candidate {i+1}: Score={score:.1f} (CAGR={cagr:.1f}%, DD={dd:.1f}%, Tr={tr})")

        # Sort
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        best_gen_score, best_gen_genome, b_cagr, b_dd, b_tr = scored_pop[0]
        print(f"🏆 Best of Gen {gen+1}: Score={best_gen_score:.1f} | CAGR={b_cagr:.1f}% | DD={b_dd:.1f}% | Tr={b_tr}")
        print(f"   DNA: {best_gen_genome}")

        if best_gen_score > best_overall_score:
            best_overall_score = best_gen_score
            best_overall_genome = best_gen_genome

        # Elitism
        elites = [x[1] for x in scored_pop[:ELITE_SIZE]]
        
        # Selection & Crossover
        next_gen = elites[:]
        while len(next_gen) < POPULATION_SIZE:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)
            
        population = next_gen
        
    print("\n🏁 EVOLUTION COMPLETE.")
    print(f"👑 Ultimate Winner (Score: {best_overall_score:.1f})")
    print(json.dumps(best_overall_genome, indent=2))
    
    # Save winner to file
    with open("alpha_dna.json", "w") as f:
        json.dump(best_overall_genome, f, indent=2)

    # AUTO-UPDATE generated_strategies (Dangerous but requested)
    config_path = os.path.join("config", "generated_strategies.json")
    with open(config_path, "r") as f:
        strategies = json.load(f)
        
    # Find SEPA
    for s in strategies:
        if s["name"] == "Apex SEPA Champion (2026)":
            winner_cfg = genome_to_config(best_overall_genome)
            # Update specific fields
            s["risk_parameters"] = winner_cfg["risk_parameters"]
            s["execution_parameters"] = winner_cfg["execution_parameters"]
            # Update entry rules specifically too
            for rule in s["entry_rules"]:
                if rule.get("col") == "rs_rating":
                    rule["val"] = winner_cfg["execution_parameters"]["rs_floor"]
            break
            
    with open(config_path, "w") as f:
        json.dump(strategies, f, indent=2)
    print("✅ Strategy Updated.")

if __name__ == "__main__":
    main()
