import os
import sys
import json
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from execution.engine import run_backtest, prepare_backtest_data
from strategies.strategy_loader import load_strategies


def verify_qullamaggie():
    print("🚀 VERIFY_QULLAMAGGIE: Starting Breakout + EP Verification...")

    # 1. Load Strategy Configs
    config_path = "config/generated_strategies.json"
    try:
        with open(config_path, "r") as f:
            strategies_config = json.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load {config_path}: {e}")
        return

    targets = {"Qullamaggie Breakout (Daily)", "Qullamaggie EP (Daily)"}
    strat_configs = [s for s in strategies_config if s.get("name") in targets]
    if not strat_configs:
        print("ERROR: Qullamaggie strategies not found.")
        print("Available:", [s.get("name") for s in strategies_config])
        return

    # 2. Load Data
    print("...Loading Data (Russell 3000, ~20 years)...")
    symbols = get_universe_symbols("RUSSELL3000")
    years = 20
    start_date = (datetime.utcnow().date() - timedelta(days=365 * years)).isoformat()
    days = (252 * years) + 250
    data = fetch_data_pack(symbols, days=days, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True)

    # 3. Prepare Data
    print("...Preparing Data...")
    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=g_data)

    # 4. Run Backtest
    print("...Running Backtest...")
    strategies = load_strategies(strat_configs)
    results = run_backtest(
        strategies,
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=None,
        global_data=g_data,
    )

    if not isinstance(results, list):
        results = [results]

    # 5. Report
    for res in results:
        print("\n" + "=" * 40)
        print(f"RESULTS: {res.get('strategy', 'Unknown')}")
        print("=" * 40)
        print(f"Total Trades:   {res.get('total_trades', 0)}")
        print(f"Final Equity:   ${res.get('final_value', 0):,.2f}")
        print(f"CAGR:           {res.get('cagr', 0) * 100:.2f}%")
        print(f"Max Drawdown:   {res.get('max_drawdown_pct', 0) * 100:.2f}%")
        gate_audit = res.get("gate_audit")
        if isinstance(gate_audit, dict):
            print(f"Gate Audit: {gate_audit}")
        print("=" * 40)


if __name__ == "__main__":
    verify_qullamaggie()
