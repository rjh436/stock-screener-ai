import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.cache_manager import DataCache
from data.indices import get_index_symbols
from data.schwab_client import sd
from execution.engine import run_backtest
from optimization.evolution import EvolutionEngine
from strategies.generic import GenericStrategy

GEN_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json"))
HISTORY_DAYS = 1260  # 5 years


def fetch_sp1500_symbols() -> List[str]:
    return get_index_symbols("S&P 1500")


def fetch_data(symbols: List[str], days: int = HISTORY_DAYS) -> Dict[str, pd.DataFrame]:
    print(f"Fetching data for {len(symbols)} symbols over last {days} days...")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50)
    data: Dict[str, pd.DataFrame] = {}

    def load_symbol(sym: str):
        try:
            df = DataCache.get_cached_data(sym)
            if df is not None and len(df) >= days * 0.9:
                return sym, df[(df.index >= start) & (df.index <= end)]
            candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
            if candles:
                df = pd.DataFrame(candles)
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                df = df.set_index("datetime").sort_index()
                DataCache.save_to_cache(sym, df)
                return sym, df
        except Exception:
            pass
        return sym, None

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(load_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, df = future.result()
            if df is not None:
                data[sym] = df
    return data


def calculate_fitness(result: Dict) -> float:
    trades = result.get("total_trades", 0) or 0
    sharpe = result.get("sharpe", 0) or 0
    win_rate = result.get("hit_rate", 0) or 0
    avg_profit_pct = result.get("avg_profit_pct", 0) or 0
    profit_factor = result.get("profit_factor", 0) or 0
    payoff = result.get("payoff_ratio", 0) or 0

    if trades < 20:
        return -1000.0

    score = (avg_profit_pct * 40.0) + (min(sharpe, 3.0) * 30.0) + (profit_factor * 20.0)
    if payoff > 2.0:
        score += 50.0
    return score


def run_evolution(data_map: Dict, global_data: Dict):
    print("\n--- Starting Deep Metric Evolution ---")
    engine = EvolutionEngine()
    try:
        with open(GEN_CONFIG_PATH, "r") as f:
            engine.population = json.load(f)
    except Exception:
        engine.generate_initial_population()

    generations = 5
    for gen_i in range(generations):
        print(f"\nEvaluating Gen {engine.generation_count}...")
        pop_results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(run_backtest, GenericStrategy(genome), data_map, None, 100000.0, None, global_data): genome
                for genome in engine.population
            }
            for future in as_completed(futures):
                genome = futures[future]
                try:
                    res = future.result()
                    score = calculate_fitness(res)
                    pop_results.append({"genome": genome, "score": score, "stats": res})
                except Exception:
                    continue

        if not pop_results:
            break

        ranked = sorted(pop_results, key=lambda x: x["score"], reverse=True)

        print(f"Top Gen {engine.generation_count}:")
        for i, r in enumerate(ranked[:3]):
            stats = r["stats"]
            print(
                f"#{i+1} Score: {r['score']:.1f} | Sharpe: {stats.get('sharpe',0):.2f} | Profit: {stats.get('avg_profit_pct',0):.2f}% "
                f"| WR: {stats.get('hit_rate',0):.1f}% | Hold: {stats.get('avg_days_held',0):.1f}d"
            )

        if gen_i < generations - 1:
            engine.evolve(ranked)

    if pop_results:
        with open(GEN_CONFIG_PATH, "w") as f:
            json.dump([r["genome"] for r in ranked[:10]], f, indent=4)


def run_optimization():
    symbols = fetch_sp1500_symbols()
    if not symbols:
        return
    data_map = fetch_data(symbols, days=HISTORY_DAYS)
    global_data_raw = fetch_data(["SPY", "$VIX", "VIX"], days=HISTORY_DAYS)
    vix_data = global_data_raw.get("$VIX") or global_data_raw.get("VIX")
    run_evolution(data_map, {"SPY": global_data_raw.get("SPY"), "VIX": vix_data})


if __name__ == "__main__":
    run_optimization()
