import sys
import os
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest
from strategies.strategy_loader import load_strategies

def calibrate():
    print("🚀 Starting Stop-Loss Calibration (Phase 4)...")
    
    # 1. Load Data (FULL S&P 1500 for Accuracy)
    universe = "S&P 1500" 
    print(f"📊 Loading {universe} Data (500 days)...")
    symbols = get_index_symbols(universe)
    # 500 days captures the recent regime (High Rate/Vol)
    data = fetch_data_pack(symbols, days=500)
    
    # 2. Load Config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "generated_strategies.json")
    try:
        with open(config_path, "r") as f:
            default_configs = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # 3. Define Test Range
    atr_settings = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    # 4. Select Strategies
    target_names = ["Apex Wealth (Gen 12)", "Super Signal (Gen 12 Wealth)", "Apex Income (Gen 9)"]
    configs = [s for s in default_configs if any(t in s['name'] for t in target_names)]
    
    if not configs:
        print("❌ Target strategies not found in config!")
        return

    # 5. Run Calibration Loop
    results = []  # Collect metrics for optional downstream use
    for config in configs:
        base_name = config['name']
        print(f"\n🔬 Calibrating: {base_name}")
        print(f"{'ATR':<6} | {'CAGR':<8} | {'Win Rate':<8} | {'Avg PnL':<8} | {'Trades':<6} | {'Max DD':<8}")
        print("-" * 65)
        
        best_cagr = -999.0
        best_atr = 0.0
        
        for atr_val in atr_settings:
            # Create a temporary config override
            # We must Modify the genome directly so the Strategy Class sees it
            test_config = config.copy()
            test_config['stop_loss_atr'] = atr_val
            
            # Load strategy with new config
            test_strat_obj = load_strategies([test_config])[0]
            
            # Run Backtest
            stats = run_backtest(test_strat_obj, data, start_cash=100000.0)
            
            # Log Result
            cagr = stats.get('cagr', 0.0)
            print(f"{atr_val:<6} | {cagr:<8.1%} | {stats.get('hit_rate',0):<8.1f}% | {stats.get('avg_profit_pct',0):<8.2f}% | {stats.get('total_trades',0):<6} | {stats.get('max_drawdown_pct',0):<8.1f}%")
            
            results.append({
                "Strategy": base_name,
                "ATR": atr_val,
                "CAGR": cagr,
                "WinRate": stats.get('hit_rate', 0)
            })
            
            if cagr > best_cagr:
                best_cagr = cagr
                best_atr = atr_val

        print(f"👉 Recommended Stop Loss: {best_atr} ATR (CAGR: {best_cagr:.1%})")

    print("\n✅ Calibration Complete.")

if __name__ == "__main__":
    calibrate()
