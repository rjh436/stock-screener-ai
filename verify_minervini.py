import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from execution.engine import prepare_backtest_data, run_backtest
    from data.loader import fetch_data_pack
    from data.universe import get_universe_symbols
    from strategies.strategy_loader import load_strategies
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def verify_minervini():
    print("🚀 VERIFY_MINERVINI: Starting SEPA Verification...")
    
    # 1. Load Data
    print("...Loading Data (Russell 3000, ~20 years)...")
    symbols = get_universe_symbols("RUSSELL3000")
    years = 20
    start_date = (datetime.utcnow().date() - timedelta(days=365 * years)).isoformat()
    days = (252 * years) + 250  # trading days + buffer
    data = fetch_data_pack(symbols, days=days, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True)
    
    # 2. Prepare Data
    print("...Preparing Data...")
    prepared = prepare_backtest_data(data, symbols, start_date=start_date, global_data=g_data)

    # 3. Load Strategy from Config
    print("...Loading Strategy Config...")
    config_path = "config/generated_strategies.json"
    try:
        with open(config_path, "r") as f:
            strategies_config = json.load(f)
    except Exception as e:
        print(f"CRITICAL: Failed to load {config_path}: {e}")
        return

    target_name = "Minervini SEPA (Daily)"
    strat_configs = [s for s in strategies_config if s.get("name") == target_name]
    if not strat_configs:
        print(f"ERROR: Strategy '{target_name}' not found.")
        print("Available:", [s.get("name") for s in strategies_config])
        return

    print(f"🛡️  Strategy: {target_name}")

    # 4. Run Backtest
    print("...Running Backtest...")
    strat = load_strategies(strat_configs)[0]
    
    results = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=None,
        global_data=g_data
    )
    
    if not results:
        print("❌ CRITICAL: No results returned.")
        return

    res = results[0] if isinstance(results, list) else results
    final_val = res.get("final_value", 100000)
    cagr = ((final_val / 100000.0) ** (1/6) - 1) * 100 # Approx 6 years
    dd = res.get("max_drawdown", 0) * 100
    trades = res.get("total_trades", 0)
    
    print("\n" + "="*40)
    print(f"🏁 RESULT: CAGR: {cagr:.2f}% | DD: {dd:.2f}% | Trades: {trades}")
    gate_audit = res.get("gate_audit")
    if isinstance(gate_audit, dict):
        print(f"Gate Audit: {gate_audit}")
    print("="*40)
    
    if cagr > 15.0:
        print("✅ SUCCESS: Strategy > 15% CAGR. Zombie Runner is ALIVE!")
    else:
        print("⚠️  WARNING: Strategy underperformed (< 15%). Logic check needed.")

if __name__ == "__main__":
    verify_minervini()
