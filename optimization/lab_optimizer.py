import os
import sys
import json
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest
from strategies.generic import GenericStrategy
from optimization.evolution import EvolutionEngine


GEN_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json"))
LAB_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/lab_candidates.json"))


def load_data(universe="S&P 1500", days=1260):
    symbols = get_index_symbols(universe)
    if not symbols:
        return {}, {}
    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX") or g_data.get("VIX")
    return data_map, {"SPY": g_data.get("SPY"), "VIX": vix}


def calculate_lab_fitness(result: dict) -> float:
    cagr = result.get("cagr", 0.0)
    max_dd = result.get("max_drawdown_pct", 0.0)
    return (cagr * 5000) - (max_dd * 200)


class LabEvolution(EvolutionEngine):
    def mutate(self, genome: dict) -> dict:
        targetable = any(tag in genome.get("name", "") for tag in ["Super Signal", "Gen 12", "Gen12"])
        if not targetable:
            return super().mutate(genome)

        mutant = copy.deepcopy(genome)
        mutant["name"] = mutant.get("name", "Strategy") + "_lab"
        exit_rules = mutant.get("exit_rules") or []
        mutant["exit_rules"] = exit_rules

        scoring_weights = copy.deepcopy(mutant.get("scoring_weights") or {})
        scoring_weights["sniper_bonus"] = round(random.uniform(40, 80), 2)
        scoring_weights["rsi_factor"] = round(random.uniform(1.5, 3.5), 2)
        scoring_weights["trend_bonus"] = round(random.uniform(15, 35), 2)
        mutant["scoring_weights"] = scoring_weights

        roll = random.random()
        if roll < 0.4:
            # Profit target between 15%-50%
            pt = round(random.uniform(1.15, 1.50), 2)
            replaced = False
            for rule in exit_rules:
                if rule.get("type") == "profit_target":
                    rule["val"] = pt
                    replaced = True
                    break
            if not replaced:
                exit_rules.append({"type": "profit_target", "val": pt})
        elif roll < 0.7:
            # RSI exit for overheated moves
            rsi_cut = random.choice([80, 85, 90])
            exit_rules.append({"col": "rsi14", "op": ">", "val": rsi_cut})
        else:
            # Tighten stop loss
            mutant["stop_loss_atr"] = round(random.uniform(3.0, 4.4), 1)

        return mutant


def run_generation(engine: LabEvolution, data_map, global_ctx, population):
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(
                run_backtest,
                GenericStrategy(g),
                data_map,
                None,
                100000.0,
                None,
                global_ctx,
                g.get("scoring_weights"),
            ): g
            for g in population
        }
        for fut in as_completed(futures):
            genome = futures[fut]
            try:
                res = fut.result()
                score = calculate_lab_fitness(res)
                results.append({"genome": genome, "score": score, "stats": res})
            except Exception as e:
                print(f"⚠️ Backtest failed for {genome.get('name', 'unknown')}: {e}")
    return results


def run_lab_optimization(generations=3):
    print("🚀 Launching Strategy Lab (Safe-Mode)")
    data_map, global_ctx = load_data("S&P 1500", days=1260)
    if not data_map:
        print("❌ Data load failed; aborting.")
        return

    try:
        with open(GEN_CONFIG, "r") as f:
            base_population = json.load(f)
    except Exception as e:
        print(f"❌ Unable to read base strategies: {e}")
        return

    # Baseline (first Super Signal found)
    baseline = next((g for g in base_population if "Super Signal" in g.get("name", "")), None)
    baseline_stats = None
    if baseline:
        print("📊 Running baseline Super Signal...")
        baseline_stats = run_backtest(GenericStrategy(baseline), data_map, None, 100000.0, None, global_ctx)
        print(f"   Baseline CAGR: {baseline_stats.get('cagr',0):.2%} | DD: {baseline_stats.get('max_drawdown_pct',0):.1f}%")

    engine = LabEvolution()
    engine.population = base_population

    ranked = []
    for g in range(generations):
        print(f"\n🧬 Generation {g+1}/{generations}")
        gen_results = run_generation(engine, data_map, global_ctx, engine.population)
        if not gen_results:
            print("   No results; stopping.")
            break
        ranked = sorted(gen_results, key=lambda x: x["score"], reverse=True)
        top = ranked[0]["stats"]
        print(f"   🏆 Top: {top.get('strategy','?')} | CAGR: {top.get('cagr',0):.2%} | DD: {top.get('max_drawdown_pct',0):.1f}%")
        engine.evolve(ranked)

    if ranked:
        candidates = [r["genome"] for r in ranked[:10]]
        with open(LAB_OUTPUT, "w") as f:
            json.dump(candidates, f, indent=4)
        print(f"\n✅ Wrote lab candidates to {LAB_OUTPUT}")

        if baseline_stats:
            best_super = next((r for r in ranked if "Super Signal" in r["genome"].get("name", "")), ranked[0])
            best_stats = best_super["stats"]
            print("\n📈 Baseline vs Lab Candidate (Super Signal):")
            print(f"   Baseline CAGR: {baseline_stats.get('cagr',0):.2%} | DD: {baseline_stats.get('max_drawdown_pct',0):.1f}%")
            print(f"   Candidate CAGR: {best_stats.get('cagr',0):.2%} | DD: {best_stats.get('max_drawdown_pct',0):.1f}%")
    else:
        print("❌ No ranked candidates produced.")


if __name__ == "__main__":
    run_lab_optimization(generations=5)
