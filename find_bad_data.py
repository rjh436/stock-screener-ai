import sys
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
sys.path.append(os.getcwd())

from data.schwab_client import sd
from data.indices import get_index_symbols

print("Fetching S&P 100 symbols...")
symbols = get_index_symbols("S&P 100")

print(f"Scanning {len(symbols)} symbols for data anomalies...")
end = datetime.now(timezone.utc)
start = end - timedelta(days=365*10) # 10 years

bad_symbols = []

for sym in symbols:
    try:
        candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
        if not candles:
            continue
            
        df = pd.DataFrame(candles)
        
        # Check for massive daily moves (> 300%)
        df["pct_change"] = (df["close"] - df["open"]) / df["open"]
        max_change = df["pct_change"].max()
        min_change = df["pct_change"].min()
        
        # Check for price < 1.0 (Penny stock territory in S&P 100 is suspicious)
        min_price = df["low"].min()
        
        if max_change > 3.0: # > 300% gain in one day
            print(f"🚨 {sym}: Max Daily Gain {max_change*100:.0f}% on {df.iloc[df['pct_change'].argmax()]['datetime']}")
            bad_symbols.append(sym)
        elif min_price < 1.0:
             print(f"⚠️ {sym}: Low Price ${min_price:.2f}")
             # Don't necessarily blacklist, but note it.
             
    except Exception as e:
        print(f"Error {sym}: {e}")

print("\nPotential Bad Symbols:", bad_symbols)
