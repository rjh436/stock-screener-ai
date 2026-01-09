#!/usr/bin/env python3
import json
import os
import sys

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from strategies.strategy_loader import load_strategies


CONFIG_PATH = os.path.join("config", "generated_strategies.json")
STRATEGY_NAME = "Apex Sniper Wealth v2.1 (Dynamic)"
UNIVERSE = "S&P 1500"
DAYS = 10000
START_CASH = 100000.0

configs = []
targets = [1.08, 1.10, 1.12, 1.15]
time_stops = [5, 7, 10]
adx_vals = [20, 25, 30]

for t in targets:
    for d in time_stops:
        for a in adx_vals:
            configs.append(
                {
                    "profit_target": t,
                    "time_stop": d,
                    "min_adx": float(a),
                }
            )


def _profit_factor(trades_list):
    gains = 0.0
    losses = 0.0
    for trade in trades_list or []:
        try:
            pnl = float(trade.get("PnL", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl = 0.0
        if pnl > 0:
            gains += pnl
        elif pnl < 0:
            losses += -pnl
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _load_strategy_config(path, name):
    with open(path, "r") as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError("generated_strategies.json must be a list")
    for cfg in configs:
        if cfg.get("name") == name:
            return cfg
    raise ValueError(f"Strategy not found: {name}")


def main():
    base_config = _load_strategy_config(CONFIG_PATH, STRATEGY_NAME)

    symbols = get_index_symbols(UNIVERSE) or []
    if not symbols:
        raise RuntimeError(f"No symbols found for universe: {UNIVERSE}")

    data = fetch_data_pack(symbols, days=DAYS + 200, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=DAYS + 200, backtest_mode=True) or {}
    spy_df = g_data.get("SPY")
    vix_df = g_data.get("$VIX")
    if vix_df is None:
        vix_df = g_data.get("VIX")
    global_data = {"SPY": spy_df, "VIX": vix_df}

    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=None,
        global_data=global_data,
    )

    rows = []
    for config in configs:
        variant = f"pt{config['profit_target']}_ts{config['time_stop']}_adx{config['min_adx']}"
        params = dict(base_config)
        params["exit_rules"] = [{"type": "profit_target", "val": config["profit_target"]}]
        params["time_stop"] = config["time_stop"]
        params["min_adx"] = config["min_adx"]

        strategies = load_strategies([params])
        result = run_backtest(
            strategies,
            prepared,
            start_cash=START_CASH,
            start_date=None,
        )

        trades_list = result.get("trades_list") or []
        rows.append(
            {
                "Variant": variant,
                "ProfitTarget": config["profit_target"],
                "TimeStop": config["time_stop"],
                "MinADX": config["min_adx"],
                "CAGR": float(result.get("cagr", 0.0) or 0.0) * 100.0,
                "MaxDD": float(result.get("max_drawdown_pct", 0.0) or 0.0),
                "ProfitFactor": _profit_factor(trades_list),
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("ProfitFactor", ascending=False, ignore_index=True)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
