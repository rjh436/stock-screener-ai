#!/usr/bin/env python3
import copy
import json
import os

import pandas as pd

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution import engine
from strategies import strategy_loader


CONFIG_PATH = os.path.join("config", "generated_strategies.json")
OUTPUT_PATH = os.path.join("data", "apex_experiments_results.csv")
UNIVERSE = "S&P 1500"
DAYS = 10000
START_CASH = 100000.0


def _load_baseline_configs(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("generated_strategies.json must be a list of strategy configs")
    return data


def _build_variant(configs: list[dict], updates: dict) -> list[dict]:
    cloned = [copy.deepcopy(cfg) for cfg in configs]
    for cfg in cloned:
        cfg.update(updates)
    return cloned


def _profit_factor(trades_list: list[dict]) -> float:
    if not trades_list:
        return 0.0
    gains = 0.0
    losses = 0.0
    for trade in trades_list:
        pnl = float(trade.get("PnL", 0.0) or 0.0)
        if pnl > 0:
            gains += pnl
        elif pnl < 0:
            losses += -pnl
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def main() -> None:
    baseline_configs = _load_baseline_configs(CONFIG_PATH)
    if not baseline_configs:
        raise RuntimeError("No baseline strategies found in generated_strategies.json")

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

    prepared = engine.prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=None,
        global_data=global_data,
    )

    experiments = [
        ("Baseline", baseline_configs),
        ("Variant A (Regime)", _build_variant(baseline_configs, {"regime_filter": True})),
        ("Variant B (Dynamic)", _build_variant(
            baseline_configs,
            {"vix_limit_scaling": True, "vix_position_sizing": True},
        )),
        ("Variant C (Quality)", _build_variant(baseline_configs, {"min_adx": 20})),
        ("Variant D (Combined)", _build_variant(
            baseline_configs,
            {
                "regime_filter": True,
                "vix_limit_scaling": True,
                "vix_position_sizing": True,
                "min_adx": 20,
            },
        )),
    ]

    detailed_rows = []
    summary_rows = []

    for label, configs in experiments:
        strategies = strategy_loader.load_strategies(configs)
        result = engine.run_backtest(
            strategies,
            prepared,
            start_cash=START_CASH,
            start_date=None,
        )

        trades_list = result.get("trades_list") or []
        profit_factor = _profit_factor(trades_list)
        cagr = float(result.get("cagr", 0.0) or 0.0)
        max_dd = float(result.get("max_drawdown_pct", 0.0) or 0.0)
        win_rate = float(result.get("hit_rate", 0.0) or 0.0)
        total_trades = int(result.get("total_trades", 0) or 0)

        detailed = {k: v for k, v in result.items() if k not in ("equity_curve", "trades_list")}
        detailed.update(
            {
                "variant": label,
                "cagr_pct": cagr * 100.0,
                "profit_factor": profit_factor,
                "win_rate": win_rate,
                "total_trades": total_trades,
            }
        )
        detailed_rows.append(detailed)

        summary_rows.append(
            {
                "Variant": label,
                "CAGR (%)": round(cagr * 100.0, 2),
                "Max Drawdown (%)": round(max_dd, 2),
                "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else float("inf"),
                "Win Rate (%)": round(win_rate, 2),
                "Total Trades": total_trades,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nApex 2.0 Experiment Results")
    print(summary_df.to_string(index=False))

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(OUTPUT_PATH, index=False)

    print("\u2705 Experiments Complete. No config files were modified.")


if __name__ == "__main__":
    main()
