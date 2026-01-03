from __future__ import annotations

import copy
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import DEFAULT_SCORING_WEIGHTS, PreparedBacktestData, prepare_backtest_data, run_backtest
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


def _to_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


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
    Fitness objective:
    - Primary: CAGR
    - Constraint: max drawdown must be < 25%
    - Tie-breaker: win rate
    """
    cagr = _to_float(result.get("cagr", 0.0) or 0.0)
    win_rate = _to_float(result.get("hit_rate", 0.0) or 0.0) / 100.0
    max_dd_pct = _to_float(result.get("max_drawdown_pct", 0.0) or 0.0)

    if abs(max_dd_pct) >= 25.0:
        return 0.0

    if cagr <= 0:
        return 0.0

    return float((cagr * 1000.0) + win_rate)


class Optimizer:
    def __init__(
        self,
        strategy_name: str,
        population_size: int = 30,
        generations: int = 10,
        n_jobs: int = 10,
        universe: str = "S&P 1500",
        days: int = 1260,
    ) -> None:
        self.strategy_name = str(strategy_name)
        self.population_size = max(2, int(population_size or 0))
        self.generations = max(1, int(generations or 0))
        self.n_jobs = max(1, min(int(n_jobs or 1), MAX_WORKERS))
        self.universe = universe
        self.days = int(days or 0)
        self.gene_ranges: Dict[str, Tuple[float, float, Any]] = {}
        self.elite_fraction = 0.20

    def set_gene_range(self, gene: str, min_val: float, max_val: float, value_type: Any) -> None:
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        self.gene_ranges[str(gene)] = (float(min_val), float(max_val), value_type)

    def _load_base_genome(self) -> Dict:
        parents = _read_parent_strategies(GEN_CONFIG)
        if not parents:
            raise RuntimeError(f"Unable to read strategies from {GEN_CONFIG}")

        for g in parents:
            if str(g.get("name")) == self.strategy_name:
                return dict(g)
        raise RuntimeError(f"Strategy '{self.strategy_name}' not found in {GEN_CONFIG}")

    def _sample_value(self, gene: str, lo: float, hi: float, value_type: Any) -> Any:
        if value_type is int:
            return int(random.randint(int(lo), int(hi)))
        val = float(random.uniform(float(lo), float(hi)))
        if gene == "breakeven_pct":
            return round(val, 3)
        if gene in {"profit_target", "stop_loss_atr"}:
            return round(val, 2)
        return val

    def _clamp_value(self, gene: str, value: Any, lo: float, hi: float, value_type: Any) -> Any:
        if value_type is int:
            try:
                val = int(round(float(value)))
            except Exception:
                val = int(lo)
            return max(int(lo), min(int(hi), val))

        try:
            val = float(value)
        except Exception:
            val = float(lo)
        val = max(float(lo), min(float(hi), val))
        if gene == "breakeven_pct":
            return round(val, 3)
        if gene in {"profit_target", "stop_loss_atr"}:
            return round(val, 2)
        return val

    @staticmethod
    def _get_profit_target(genome: Dict) -> Optional[float]:
        exit_rules = genome.get("exit_rules")
        if not isinstance(exit_rules, list):
            return None
        for rule in exit_rules:
            if isinstance(rule, dict) and rule.get("type") == "profit_target":
                try:
                    return float(rule.get("val"))
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _set_profit_target(genome: Dict, value: float) -> None:
        exit_rules = genome.get("exit_rules")
        if not isinstance(exit_rules, list):
            exit_rules = []

        rule = None
        for item in exit_rules:
            if isinstance(item, dict) and item.get("type") == "profit_target":
                rule = item
                break

        if rule is None:
            rule = {"type": "profit_target", "val": value}
            exit_rules.append(rule)
        else:
            rule["val"] = value

        genome["exit_rules"] = exit_rules

    def _apply_gene_ranges(self, genome: Dict, *, randomize: bool) -> Dict:
        for gene, (lo, hi, value_type) in self.gene_ranges.items():
            if gene == "profit_target":
                if randomize:
                    val = self._sample_value(gene, lo, hi, value_type)
                else:
                    cur = self._get_profit_target(genome)
                    val = self._clamp_value(gene, cur, lo, hi, value_type)
                self._set_profit_target(genome, val)
                continue

            if randomize:
                val = self._sample_value(gene, lo, hi, value_type)
            else:
                val = self._clamp_value(gene, genome.get(gene), lo, hi, value_type)
            genome[gene] = val

        genome["sizing_mode"] = "risk"
        return genome

    def _mutate(self, genome: Dict) -> Dict:
        child = copy.deepcopy(genome)
        if not self.gene_ranges:
            return child
        gene = random.choice(list(self.gene_ranges.keys()))
        lo, hi, value_type = self.gene_ranges[gene]
        val = self._sample_value(gene, lo, hi, value_type)
        if gene == "profit_target":
            self._set_profit_target(child, val)
        else:
            child[gene] = val
        return self._apply_gene_ranges(child, randomize=False)

    def _evaluate_population(
        self,
        population: Sequence[Dict],
        prepared_data: PreparedBacktestData,
        global_context: Dict,
    ) -> Tuple[List[Dict], List[Dict]]:
        results: List[Dict] = []
        failures: List[Dict] = []

        with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
            future_to_genome = {
                ex.submit(
                    run_backtest,
                    GenericStrategy(genome),
                    prepared_data,
                    None,
                    100000.0,
                    None,
                    global_context,
                    scoring_weights=genome.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS,
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

    def run(self) -> Dict:
        base_genome = self._load_base_genome()
        base_genome = self._apply_gene_ranges(base_genome, randomize=False)

        data_map, global_context = load_optimization_data(self.universe, days=self.days)
        if not data_map:
            raise RuntimeError("Data load failed; aborting optimization.")

        prepared_data = prepare_backtest_data(
            data_map,
            symbol_universe=None,
            start_date=None,
            global_data=global_context,
        )

        population: List[Dict] = []
        for i in range(self.population_size):
            genome = copy.deepcopy(base_genome)
            population.append(self._apply_gene_ranges(genome, randomize=(i != 0)))

        best_genome = base_genome
        for gen in range(self.generations):
            print(f"\n🧬 Generation {gen + 1}/{self.generations} (pop={len(population)})")
            pop_res, failures = self._evaluate_population(population, prepared_data, global_context)
            ranked = sorted(pop_res, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

            if failures:
                by_type = {}
                for f in failures:
                    by_type[f["type"]] = by_type.get(f["type"], 0) + 1
                logger.warning("Backtest failures: %d (by type=%s)", len(failures), by_type)

            top = ranked[0] if ranked else None
            if top is not None:
                best_genome = dict(top.get("genome") or best_genome)
                print(_format_top_line(top.get("stats") or {}, top.get("genome") or {}, float(top.get("score") or 0.0)))

            elite_count = max(2, int(self.population_size * self.elite_fraction))
            elites = [r.get("genome") for r in ranked[:elite_count] if isinstance(r.get("genome"), dict)]
            if not elites:
                elites = [base_genome]

            next_population = [copy.deepcopy(g) for g in elites]
            while len(next_population) < self.population_size:
                parent = random.choice(elites)
                next_population.append(self._mutate(parent))

            population = next_population

        return best_genome


def load_optimization_data(universe: str = "S&P 1500", days: int = 1260):
    print(f"📥 Loading Data for {universe}...")
    symbols = get_index_symbols(universe)
    if not symbols:
        return {}, {}

    data_map = fetch_data_pack(symbols, days=days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)

    spy_df = g_data.get("SPY")

    # Avoid DataFrame truthiness (ValueError: ambiguous truth value)
    vix_df = g_data.get("$VIX")
    if vix_df is None:
        vix_df = g_data.get("VIX")

    return data_map, {"SPY": spy_df, "VIX": vix_df}


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
    hit_pct = _to_float(stats.get("hit_rate", 0.0) or 0.0)
    avg_profit = _to_float(stats.get("avg_profit_pct", 0.0) or 0.0) / 100.0
    profit_pct = avg_profit * 100.0

    return (
        f"      🏆 Top: {name_short} | Type: {str(genome.get('type','?')).lower():6s}"
        f" | Fitness: {score:9.2f} | CAGR: {cagr:6.1%} | Profit: {profit_pct:5.1f}% | Hit: {hit_pct:6.2f}%"
    )


def _evaluate_population(
    population: Sequence[Dict],
    prepared_data: PreparedBacktestData,
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
                scoring_weights=genome.get("scoring_weights") or DEFAULT_SCORING_WEIGHTS,
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
