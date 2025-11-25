import sys
import os
import time
import json
import pandas as pd
from optimization.optimizer import run_evolution

# Ensure we are running in the right environment
print("🚀 Initializing 'Hyper-Growth' Optimization Loop...")
print("🎯 Target: CAGR > 30% | Avg Profit > 2.0%")
print("-----------------------------------------------")

# Verify directories
os.makedirs("config", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Run the Optimizer
# The optimizer.py script now imports the updated evolution.py
# and runs for 3 generations internally per call. 
# We will loop it 4 times = 12 Generations total.

start_time = time.time()

for i in range(4):
    print(f"\n⚡ [Hyper-Growth Loop {i+1}/4] Starting Optimization Cycle...")
    
    # We execute the existing optimizer script which handles the threading and backtest logic
    try:
        # Re-import or run the function directly
        # We can just call run_evolution() if we imported it
        run_evolution()
        
        # Check metrics
        if os.path.exists("config/generated_strategies.json"):
            with open("config/generated_strategies.json", "r") as f:
                best = json.load(f)[0]
                print(f"   🏆 Current Champion: {best.get('name', 'Unknown')}")
                
    except Exception as e:
        print(f"   ❌ Cycle {i+1} Failed: {e}")
        import traceback
        traceback.print_exc()

elapsed = (time.time() - start_time) / 60.0
print(f"\n✅ Optimization Complete in {elapsed:.1f} minutes.")
print("📂 Results saved to: config/generated_strategies.json")
