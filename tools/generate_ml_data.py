import sys
import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution import engine
from strategies.strategy_loader import load_strategies

LOOKBACK_DAYS = 1260
BATCH_SIZE = 100


def fetch_data_batch(symbols):
    """
    Fetches a batch of symbols using the project's standard loader.
    Wrapped so joblib can parallelize it.
    """
    try:
        return fetch_data_pack(symbols, days=LOOKBACK_DAYS)
    except Exception as e:
        print(f"⚠️ Batch fetch failed for {len(symbols)} symbols: {e}")
        return {}


def generate_data():
    print("🚀 Starting ML Data Generation...")
    
    # 1. Load Data
    print("📊 Loading S&P 1500 Data...")
    all_symbols = get_index_symbols("S&P 1500")
    all_symbols = [s for s in all_symbols if s]

    if not all_symbols:
        print("❌ No symbols returned for S&P 1500.")
        return

    chunks = [all_symbols[i : i + BATCH_SIZE] for i in range(0, len(all_symbols), BATCH_SIZE)]
    num_batches = int(np.ceil(len(all_symbols) / float(BATCH_SIZE)))
    print(f"   - Symbols:   {len(all_symbols)}")
    print(f"   - Lookback:  {LOOKBACK_DAYS} days (~5y)")
    print(f"   - Batches:   {num_batches} x {BATCH_SIZE}")

    print("⚡ Fetching batches in parallel (joblib)...")
    batch_results = Parallel(n_jobs=-1, backend="threading")(
        delayed(fetch_data_batch)(batch) for batch in chunks
    )

    full_data = {}
    for batch_dict in batch_results:
        if batch_dict:
            full_data.update(batch_dict)

    print(f"✅ Data loaded: {len(full_data)} symbols with price history.")
    
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
    engine.run_backtest(strategy, full_data, export_ml_data=True)
    
    # 4. Verify
    if os.path.exists("ml_training_data.csv"):
        df = pd.read_csv("ml_training_data.csv")
        print(f"✅ SUCCESS: Generated {len(df)} training samples.")
        print(f"   Saved to: ml_training_data.csv")
    else:
        print("❌ FAILURE: No data file was created. (Did the strategy make any trades?)")

if __name__ == "__main__":
    generate_data()
