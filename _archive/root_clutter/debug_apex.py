import sys
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
sys.path.append(os.getcwd())

from data.schwab_client import sd
from execution.engine import run_backtest
from strategies.generic import GenericStrategy
import json

# Load the strategy
with open("config/generated_strategies.json", "r") as f:
    gens = json.load(f)
    apex_genome = gens[0] # Strategy_Apex_Gen9_Alpha

print(f"Debugging {apex_genome['name']}...")

# Fetch Data for a few split-heavy stocks
symbols = ["NVDA", "AAPL", "AMZN", "TSLA"]
data_map = {}
end = datetime.now(timezone.utc)
start = end - timedelta(days=365*10) # 10 years

print("Fetching data...")
for sym in symbols:
    try:
        candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
        if candles:
            df = pd.DataFrame(candles)
            df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
            df = df.set_index("datetime").sort_index()
            data_map[sym] = df
    except Exception as e:
        print(f"Error {sym}: {e}")

# Run Backtest
print("Running backtest...")
strat = GenericStrategy(apex_genome)
results = run_backtest(strat, data_map)

print(f"Final Value: {results['final_value']}")
print(f"Total Trades: {results['total_trades']}")
print(f"Avg Profit %: {results['avg_profit_pct']}")

# Inspect Trades (we need to modify engine to return trade list, or just trust the avg)
# Since I can't modify engine easily to return the list without breaking things, 
# I'll rely on the fact that if this script shows crazy numbers, it's the strategy/data.
