#!/usr/bin/env python3
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure root package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from execution.engine import run_backtest
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols_pit, get_universe_symbols
from data.indices import get_index_symbols
from execution.engine import prepare_backtest_data
from strategies.superperformance import SuperperformanceStrategy

UI_STRATEGIES = [
    {"name": "Superperformance", "file": "config/superperformance_winner.json"},
    {"name": "Superperformance (Practical EOD No Leverage)", "file": "config/superperformance_practical_no_leverage_optimized.json"},
    {"name": "Superperformance (Practical EOD Cash V2)", "file": "config/superperformance_practical_eod_cash_v2.json"},
    {"name": "Superperformance Practical Selective V3", "file": "config/superperformance_practical_selective_v3.json"},
    {"name": "Superperformance Practical Selective V4 Candidate", "file": "config/superperformance_practical_selective_v4_candidate.json"},
    {"name": "Superperformance Practical Risk-Off Only", "file": "config/superperformance_practical_riskoff_only.json"},
    {"name": "Superperformance Alpha B4", "file": "config/superperformance_alpha_b4.json"},
]

WINDOWS = {
    "5Y": ("2021-02-22", "2026-02-27"),
    "10Y": ("2016-02-23", "2026-02-27")
}

def load_ui_strategies():
    strats = []
    for s_info in UI_STRATEGIES:
        conf_path = s_info["file"]
        if not os.path.exists(conf_path):
            logging.warning(f"Strategy config not found: {conf_path}")
            continue
        try:
            with open(conf_path, "r") as f:
                cfg = json.load(f)
            # Enforce constraints directly in config before loading
            cfg["allow_margin"] = False
            cfg["same_day_open_entries"] = 0
            strat = SuperperformanceStrategy(cfg)
            # Enforce name match to UI representation if missing
            if not strat.name or strat.name == "Superperformance Strategy":
                strat.name = s_info["name"]
            strats.append(strat)
        except Exception as e:
            logging.error(f"Failed to load {conf_path}: {e}")
    return strats

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    strategies = load_ui_strategies()
    if not strategies:
        logging.error("No UI strategies loaded!")
        sys.exit(1)

    logging.info("Loading market data...")
    # Load SP1500 for the backtest window to ensure we cover all potential symbols + PIT
    symbols = get_index_symbols("S&P 1500") or []
    try:
        # For rank harness we try to use a wide 1260 days (5y) lookback for 10Y backtest? 
        # Actually our window is 10Y, so we need fetch days=3000
        pit = get_universe_symbols_pit("RUSSELL3000", "2026-02-27")
        if pit: symbols = list(set(symbols + pit))
    except RuntimeError as re:
        logging.warning(f"Caught PIT enforcement error: {re}. Using SP1500 baseline.")
    
    data_dict = fetch_data_pack(symbols, days=3500)

    full_results = {}
    
    for window_name, (start_date, end_date) in WINDOWS.items():
        logging.info(f"--- Running {window_name} window ({start_date} to {end_date}) ---")
        prepared = prepare_backtest_data(data_dict, start_date=start_date)
        
        # Enforce PIT failure for fundamentals.
        # This will be tested or we skip if data missing.
        res = run_backtest(strategies, prepared, start_cash=100000.0, start_date=start_date, end_date=end_date)
        
        for i, strat in enumerate(strategies):
            summary = res[i]
            s_name = strat.name
            if s_name not in full_results:
                full_results[s_name] = {}
            
            full_results[s_name][window_name] = {
                "cagr": summary.get("cagr", 0.0),
                "max_drawdown": summary.get("max_drawdown", 0.0),
                "hit_rate": summary.get("hit_rate", 0.0),
                "total_trades": summary.get("total_trades", 0),
                "avg_trade_pct": summary.get("avg_trade_pct", 0.0),
                "expectancy": summary.get("expectancy", 0.0),
                "max_gross_exposure_pct": sum(summary.get("audit_report", {}).get("exposure_profile", [0]))/len(summary.get("audit_report", {}).get("exposure_profile", [1])) if summary.get("audit_report", {}).get("exposure_profile") else 0.0
            }
            logging.info(f"[{window_name}] {s_name} | CAGR: {summary.get('cagr', 0.0):.2%} | DD: {summary.get('max_drawdown', 0.0):.2%}")

    # Output machine readable JSON
    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "ui_strategy_rank_latest.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    logging.info(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
