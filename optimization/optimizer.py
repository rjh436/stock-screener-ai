import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import fetch_data_pack
from execution.engine import run_backtest
from optimization.evolution import EvolutionEngine
from strategies.generic import GenericStrategy

GEN_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json"))


def calculate_fitness(result):
    score = result.get("Score", -1_000_000.0)
    trades = result.get("total_trades", 0)
    if trades <= 0:
        return -1_000_000.0
    return float(score)


def run_evolution():
    print("🚀 Initializing Optimizer...")
    from data.indices import get_index_symbols

    symbols = get_index_symbols("S&P 1500")
    data_map = fetch_data_pack(symbols, days=1260)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=1260)

    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
    global_context = {"SPY": g_data.get("SPY"), "VIX": vix}

    engine = EvolutionEngine()
    try:
        with open(GEN_CONFIG, "r") as f:
            engine.population = json.load(f)
    except Exception:
        engine.generate_initial_population()

    for gen in range(3):
        print(f"\nEvaluating Gen {gen}...")
        pop_res = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {
                ex.submit(run_backtest, GenericStrategy(g), data_map, None, 100000.0, None, global_context): g
                for g in engine.population
            }
            for f in futures:
                try:
                    res = f.result()
                    score = calculate_fitness(res)
                    pop_res.append({"genome": futures[f], "score": score, "stats": res})
                except Exception as e:
                    print(f"Strategy failed: {e}")

        if not pop_res:
            print("⚠️ No valid strategies. Check data.")
            continue

        ranked = sorted(pop_res, key=lambda x: x["score"], reverse=True)
        best = ranked[0]["stats"]
        print(
            f"🏆 Top: {best.get('avg_profit_pct', 0):.2f}% Profit | {best.get('total_trades', 0)} Trades | Score {ranked[0]['score']:.1f}"
        )

        engine.evolve(ranked)

    with open(GEN_CONFIG, "w") as f:
        json.dump([r["genome"] for r in ranked[:10]], f, indent=4)


if __name__ == "__main__":
    run_evolution()
