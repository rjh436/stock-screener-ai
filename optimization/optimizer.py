
from __future__ import annotations

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from optimization.evolution import EvolutionEngine
from strategies.generic import GenericStrategy


GEN_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "generated_strategies.json"))

MAX_WORKERS = 16
POPULATION_SIZE = 100

# Persistence quotas (exact)
EXPORT_TOP_WEALTH = 7
EXPORT_TOP_INCOME = 8  # includes "income" and "hybrid"

# Strategic mandates (fractions)
MANDATE_CAGR_TARGET = 0.40
MANDATE_PROFIT_TARGET = 0.05
MANDATE_WIN_TARGET = 0.60


logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def _to_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


def _pct_to_frac(x: float) -> float:
    """
    Accept either percent units (e.g., 65.0) or fraction units (e.g., 0.65).
    """
    x = _to_float(x, 0.0)
    return (x / 100.0) if x > 1.0 else x


def _threshold_multiplier(value: float, target: float, *, power_below: float) -> float:
    """
    Returns a smooth multiplier that:
    - Strongly penalizes falling short of target (power scaling)
    - Continues to reward exceeding target (linear)
    """
    target = _to_float(target, 0.0)
    value = _to_float(value, 0.0)

    if target <= 0.0 or value <= 0.0:
        return 0.0

    ratio = value / target
    if ratio < 1.0:
        return float(ratio**power_below)
    return float(1.0 + (ratio - 1.0))


def calculate_fitness(result: Dict) -> float:
    """
    Fitness uses fractional units (strategic mandates):
    - CAGR target: 0.40
    - Avg profit target: 0.05
    - Win rate target: 0.60

    Fitness is a "sniper" objective:
    - Avg profit per trade has a cubic penalty below mandate and a linear reward above it.
    - Win rate has a quadratic penalty below mandate and a linear reward above it.
    - CAGR has a linear penalty below mandate and a linear reward above it.
    - Calmar is capped so ultra-low drawdowns can't dominate selection.
    """
    cagr = _to_float(result.get("cagr", 0.0) or 0.0)
    win_rate = _pct_to_frac(result.get("hit_rate", 0.0) or 0.0)
    avg_profit = _pct_to_frac(result.get("avg_profit_pct", 0.0) or 0.0)
    max_dd_pct = _to_float(result.get("max_drawdown_pct", 0.0) or 0.0)

    dd_abs = abs(max_dd_pct)
    dd_frac = (dd_abs / 100.0) if dd_abs > 1.0 else dd_abs
    calmar_raw = (cagr / dd_frac) if dd_frac > 0.0 else 0.0
    calmar_capped = min(max(calmar_raw, 0.0), 20.0)

    # Strategic multipliers (always a gradient, even above mandates)
    profit_multiplier = _threshold_multiplier(avg_profit, MANDATE_PROFIT_TARGET, power_below=3.0)
    win_multiplier = _threshold_multiplier(win_rate, MANDATE_WIN_TARGET, power_below=2.0)
    cagr_multiplier = _threshold_multiplier(cagr, MANDATE_CAGR_TARGET, power_below=1.0)

    # Calmar bonus is deliberately modest; can't dominate profit.
    calmar_bonus = 1.0 + (calmar_capped / 40.0)  # max 1.5x

    fitness = 1000.0 * calmar_bonus * profit_multiplier * win_multiplier * cagr_multiplier
    return float(max(fitness, 0.0))


def load_optimization_data(universe: str = "S&P 1500", days: int = 1260):
    print(f"📥 Loading Data for {universe}...")
    symbols = get_index_symbols(universe)
    if not symbols:
        return {}, {}

    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)

    vix = g_data.get("$VIX") or g_data.get("VIX")
    return data_map, {"SPY": g_data.get("SPY"), "VIX": vix}


def _read_parent_strategies(path: str) -> Optional[List[Dict]]:
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [p for p in payload if isinstance(p, dict)]
        return None
    except Exception:
        return None


def _format_top_line(stats: Dict, genome: Dict, score: float) -> str:
    name = str(stats.get("strategy") or genome.get("name") or "Unknown")
    name_short = (name[:25] + "..") if len(name) > 27 else name

    cagr = _to_float(stats.get("cagr", 0.0) or 0.0)
    hit_pct = _pct_to_frac(stats.get("hit_rate", 0.0) or 0.0) * 100.0
    profit_pct = _pct_to_frac(stats.get("avg_profit_pct", 0.0) or 0.0) * 100.0

    return (
        f"      🏆 Top: {name_short} | Type: {str(genome.get('type','?')).lower():6s}"
        f" | Fitness: {score:9.2f} | CAGR: {cagr:6.1%} | Profit: {profit_pct:6.2f}% | Hit: {hit_pct:6.2f}%"
    )


def _evaluate_population(
    population: Sequence[Dict],
    prepared_data,
    global_context: Dict,
) -> Tuple[List[Dict], List[Dict]]:
    results: List[Dict] = []
    failures: List[Dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_genome = {
            ex.submit(
                run_backtest,
                GenericStrategy(genome),
                prepared_data,
                None,
                100000.0,
                None,
                global_context,
                genome.get("scoring_weights"),
            ): genome
            for genome in population
        }

        for fut in as_completed(future_to_genome):
            genome = future_to_genome[fut]
            g_name = str(genome.get("name") or "unknown")
            g_type = str(genome.get("type") or "income").lower()
            try:
                stats = fut.result()
                score = calculate_fitness(stats)
                results.append({"genome": genome, "score": score, "stats": stats})
            except Exception as e:
                failures.append({"name": g_name, "type": g_type, "error": str(e)})
                logger.exception("Backtest failed for %s [%s]", g_name, g_type)
                results.append({"genome": genome, "score": 0.0, "stats": {"strategy": g_name, "cagr": 0.0, "avg_profit_pct": 0.0, "hit_rate": 0.0}})

    return results, failures


def _export_quota_strategies(ranked: Sequence[Dict]) -> List[Dict]:
    wealth: List[Dict] = []
    income: List[Dict] = []

    def is_wealth(g: Dict) -> bool:
        return str(g.get("type") or "").lower() == "wealth"

    def is_income(g: Dict) -> bool:
        return str(g.get("type") or "").lower() in {"income", "hybrid"}

    for r in ranked:
        genome = r.get("genome") if isinstance(r, dict) else None
        if not isinstance(genome, dict):
            continue

        if is_wealth(genome):
            if len(wealth) < EXPORT_TOP_WEALTH:
                wealth.append(genome)
        else:
            if len(income) < EXPORT_TOP_INCOME:
                if not is_income(genome):
                    genome = dict(genome)
                    genome["type"] = "income"
                income.append(genome)

        if len(wealth) >= EXPORT_TOP_WEALTH and len(income) >= EXPORT_TOP_INCOME:
            break

    # With quota-aware populations, these should always be met.
    return wealth[:EXPORT_TOP_WEALTH] + income[:EXPORT_TOP_INCOME]


def run_evolution_cycle(engine: EvolutionEngine, data_map: Dict, global_context: Dict, generations: int = 5) -> None:
    parents = _read_parent_strategies(GEN_CONFIG)
    engine.population_size = POPULATION_SIZE
    engine.population = engine.generate_initial_population(base_strategies=parents, base_name="HighCaliber")

    prepared_data = prepare_backtest_data(data_map, symbol_universe=None, start_date=None, global_data=global_context)

    for _ in range(generations):
        print(f"   🧬 Gen {engine.generation_count} Evaluation... (pop={len(engine.population)})")

        pop_res, failures = _evaluate_population(engine.population, prepared_data, global_context)
        ranked = sorted(pop_res, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        if failures:
            by_type = {}
            for f in failures:
                by_type[f["type"]] = by_type.get(f["type"], 0) + 1
            logger.warning("Backtest failures: %d (by type=%s)", len(failures), by_type)

        top = ranked[0] if ranked else None
        if top is not None:
            print(_format_top_line(top.get("stats") or {}, top.get("genome") or {}, float(top.get("score") or 0.0)))

        engine.evolve(ranked)

        export = _export_quota_strategies(ranked)
        with open(GEN_CONFIG, "w") as f:
            json.dump(export, f, indent=4)


if __name__ == "__main__":
    print("🚀 Starting 'High Caliber' Optimization")
    data_map, global_context = load_optimization_data("S&P 1500", days=1260)
    if not data_map:
        sys.exit(1)

    engine = EvolutionEngine()
    engine.population_size = POPULATION_SIZE

    for i in range(5):
        print(f"\n⚡ [Cycle {i+1}/5] Starting...")
        run_evolution_cycle(engine, data_map, global_context, generations=5)
