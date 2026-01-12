from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List, Optional

from data import universe
from data.loader import fetch_data_pack
from execution import engine
from simulation.paper_trader import PaperTrader
from strategies.strategy_loader import load_strategies

DEFAULT_CONFIG_PATH = os.path.join("config", "generated_strategies.json")


def run_apex(mode: str, universe_name: str, strategies: Optional[Iterable] = None):
    symbols = universe.get_universe_symbols(universe_name)

    configs = _load_strategy_configs(DEFAULT_CONFIG_PATH)
    selected_configs = _select_strategy_configs(configs, strategies)

    mode_key = (mode or "").strip().upper()
    if mode_key == "BACKTEST":
        data = fetch_data_pack(symbols, days=5000)
        strategy_objs = load_strategies(selected_configs)
        result = engine.run_backtest(strategy_objs, data, symbol_universe=symbols)
        _print_performance_summary(result)
        return result

    if mode_key == "LIVE":
        data = fetch_data_pack(symbols, days=300)
        trader = PaperTrader(configs=selected_configs)
        return trader.run_daily_scan(data_dict=data)

    raise ValueError(f"Unknown mode: {mode}. Use BACKTEST or LIVE.")


def _load_strategy_configs(path: str) -> List[dict]:
    if not os.path.exists(path):
        print(f"WARNING: Strategy config not found: {path}")
        return []
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except Exception as exc:
        print(f"WARNING: Failed to read strategy config: {exc}")
        return []
    if isinstance(data, list):
        return data
    return []


def _select_strategy_configs(configs: List[dict], strategies: Optional[Iterable]) -> List[dict]:
    if not strategies:
        return configs

    if isinstance(strategies, str):
        raw_names = [name.strip() for name in strategies.split(",") if name.strip()]
    else:
        raw_names = list(strategies)

    if raw_names and all(isinstance(item, dict) for item in raw_names):
        return list(raw_names)

    names = {_normalize_name(name) for name in raw_names if isinstance(name, str)}
    if not names:
        return configs
    if "ALL" in names:
        return configs

    selected = [cfg for cfg in configs if _normalize_name(cfg.get("name", "")) in names]
    if not selected:
        print("WARNING: No strategies matched selection. Using all strategies.")
        return configs
    return selected


def _normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _print_performance_summary(result) -> None:
    if not isinstance(result, dict):
        print("Performance Summary: No results returned.")
        return

    final_value = result.get("final_value")
    total_trades = result.get("total_trades")
    hit_rate = result.get("hit_rate")
    cagr = result.get("cagr")
    max_dd = result.get("max_drawdown_pct")
    strategy = result.get("strategy")

    print("\nPerformance Summary")
    if strategy:
        print(f"Strategy: {strategy}")
    if final_value is not None:
        print(f"Final Value: {final_value:,.2f}")
    if cagr is not None:
        print(f"CAGR: {float(cagr) * 100.0:.2f}%")
    if max_dd is not None:
        print(f"Max Drawdown: {float(max_dd):.2f}%")
    if hit_rate is not None:
        print(f"Win Rate: {float(hit_rate):.2f}%")
    if total_trades is not None:
        print(f"Total Trades: {int(total_trades)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apex controller for backtest/live runs.")
    parser.add_argument("--mode", required=True, help="BACKTEST or LIVE")
    parser.add_argument("--universe", required=True, help="Universe name (e.g., SP1500)")
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategy names from generated_strategies.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_apex(args.mode, args.universe, args.strategies)
