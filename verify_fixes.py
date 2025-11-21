import pandas as pd
import numpy as np
from execution.engine import run_backtest
from strategies.base import BaseStrategy
from typing import Dict, Optional

class MockStrategy(BaseStrategy):
    def __init__(self):
        super().__init__({})

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        # Force entry at index 200
        if i == 200:
            row = df.iloc[i]
            # Entry at 100, Stop at 95
            return {"entry_price": 100.0, "stop_price": 95.0}
        return None

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        # Hold until index 202
        if i >= 202:
            return True
        return False

def verify_gap_down():
    print("--- VERIFYING GAP DOWN FIX WITH MOCK STRATEGY ---")
    
    # Create Data
    # Index 200: Entry Signal (Close 100)
    # Index 201: Entry Execution (Open 100)
    # Index 202: GAP DOWN (Open 90) -> Should trigger Stop at 90 (Open), not 95 (Stop Price)
    
    dates = pd.date_range(start="2022-01-01", periods=300)
    df = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 300,
        "high": [105.0] * 300,
        "low": [95.0] * 300,
        "close": [100.0] * 300,
        "volume": [1000000] * 300,
    })
    
    # Index 202: Gap Down
    # Stop is 95.0.
    # Open is 90.0.
    # Low is 80.0.
    # The Engine should see Low (80) < Stop (95).
    # Then check Open (90) < Stop (95).
    # Exit Price should be 90.0.
    
    idx = 202
    df.loc[idx, "open"] = 90.0
    df.loc[idx, "high"] = 92.0
    df.loc[idx, "low"] = 80.0
    df.loc[idx, "close"] = 85.0
    
    strat = MockStrategy()
    res = run_backtest(strat, {"MOCK": df})
    
    print("\n--- RESULTS ---")
    print(f"Total Trades: {res['total_trades']}")
    print(f"PnL: {res['pnl']}")
    
    if res['total_trades'] > 0:
        # 100k start. 10% pos size = 10k.
        # Shares = 10000 / 100 = 100 shares.
        # Entry 100. Exit 90. Loss = 10 per share.
        # Total Loss = 100 * 10 = 1000.
        # If it used Stop Price (95): Loss = 100 * 5 = 500.
        
        expected_loss = -1000.0
        tolerance = 50.0 # Allow for slight rounding
        
        if abs(res['pnl'] - expected_loss) < tolerance:
            print("✅ PASS: PnL reflects execution at Open Price (90.0). Gap Down handled correctly.")
        else:
            print(f"❌ FAIL: PnL {res['pnl']} does not match expected {expected_loss}.")
            print("Likely executed at Stop Price (95.0) -> Old Bug behavior.")
            
    else:
        print("No trades generated.")

if __name__ == "__main__":
    verify_gap_down()
