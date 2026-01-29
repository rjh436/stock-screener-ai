
import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from strategies.sepa_champion import SEPAChampionStrategy
from execution.engine import run_backtest, prepare_backtest_data
from data.loader import fetch_data_pack
from data.universe import get_universe_symbols

def main():
    print("🚀 Initiating Operation Unshackle (Wealth Mode)...")
    print("🎯 Target: CAGR > 20% | Drawdown < 30%")

    # 1. Load Data
    print("📦 Loading Data Pack (Full Cycle 2015-2021)...")
    symbols = get_universe_symbols("RUSSELL3000")
    data = fetch_data_pack(symbols, days=252*7, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "VIX"], days=252*7, backtest_mode=True) or {}
    
    print("⚙️  Preparing Enriched Data...")
    prepared = prepare_backtest_data(data, symbols, None, g_data)
    
    # 2. Load "Wide Net" Config
    config_path = os.path.join("config", "generated_strategies.json")
    with open(config_path, "r") as f:
        strategies = json.load(f)
        
    strat_config = next((s for s in strategies if s["name"] == "Apex SEPA Champion (2026)"), None)
    if not strat_config:
        print("❌ Error: Strategy not found.")
        return

    # 3. Instantiate Strategy (With Overrides for Trend Following)
    # Widen the net: RS 80, ADX 15
    if "execution_parameters" not in strat_config: strat_config["execution_parameters"] = {}
    strat_config["execution_parameters"]["rs_floor"] = 80.0
    strat_config["execution_parameters"]["adx_min"] = 15.0
    
    # Also update the rules list to match
    for rule in strat_config.get("entry_rules", []):
        if rule.get("col") == "rs_rating":
            rule["val"] = 80.0

    strat = SEPAChampionStrategy(strat_config)
    
    # 4. Run Backtest
    print("⚔️  Executing Backtest...")
    res = run_backtest(
        strat,
        data=prepared,
        start_cash=100000.0,
        start_date="2015-01-01",
        end_date="2021-01-01",
        global_data=g_data
    )
    
    if not res:
        print("❌ No results returned.")
        return
        
    res = res if isinstance(res, dict) else res[0]
    
    # 5. Report
    final_val = res.get("final_value", 0)
    cagr = res.get("cagr", 0.0) * 100
    max_dd = res.get("max_drawdown_pct", 0.0) * 100
    trades = res.get("total_trades", 0)
    win_rate = res.get("hit_rate", 0.0)
    
    print("\n📊 WEALTH MODE RESULTS:")
    print(f"   Final Value: ${final_val:,.2f}")
    print(f"   CAGR:        {cagr:.2f}%")
    print(f"   Max DD:      {max_dd:.2f}%")
    print(f"   Trades:      {trades}")
    print(f"   Win Rate:    {win_rate:.1f}%")
    
    # Logic check
    if cagr > 20.0:
        print("\n✅ MISSION SUCCESS: Wealth Mode Unlocked (>20% CAGR).")
    elif cagr > 15.0:
         print("\n⚠️ MISSION PARTIAL SUCCESS: Significant Growth (>15% CAGR).")
    else:
        print("\n❌ MISSION FAILURE: Strategy Needs Tuning.")

if __name__ == "__main__":
    main()
