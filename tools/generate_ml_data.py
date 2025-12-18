import sys
import os
import json
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution import engine
from strategies.strategy_loader import load_strategies

def generate_data():
    print("🚀 Starting ML Data Generation...")
    
    # 1. Load Data
    print("📊 Loading S&P 1500 Data...")
    symbols = get_index_symbols("S&P 1500")
    data = fetch_data_pack(symbols, days=500)
    
    # 2. Load Best Strategy
    config_path = "config/generated_strategies.json"
    if not os.path.exists(config_path):
        print("❌ No strategy config found!")
        return

    with open(config_path, "r") as f:
        configs = json.load(f)
        
    best_config = configs[0] # Take the top ranked strategy
    print(f"🧬 Using Strategy: {best_config['name']}")
    
    # Load Strategy Object
    strategies = load_strategies([best_config])
    if not strategies:
        print("❌ Failed to load strategy object.")
        return
    
    strategy = strategies[0]
    
    # 3. Run Backtest with Export Enabled
    print("⚙️  Running Simulation (Exporting Data)...")
    engine.run_backtest(strategy, data, export_ml_data=True)
    
    # 4. Verify
    if os.path.exists("ml_training_data.csv"):
        df = pd.read_csv("ml_training_data.csv")
        print(f"✅ SUCCESS: Generated {len(df)} training samples.")
        print(f"   Saved to: ml_training_data.csv")
    else:
        print("❌ FAILURE: No data file was created. (Did the strategy make any trades?)")

if __name__ == "__main__":
    generate_data()
