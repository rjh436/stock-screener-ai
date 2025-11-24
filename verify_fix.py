import pandas as pd
from strategies.generic import GenericStrategy

def test_profit_target():
    print("Testing Profit Target Logic...")
    
    # 1. Setup Dummy Data
    data = {
        "open": [100, 105, 110],
        "high": [105, 112, 115],
        "low": [95, 100, 105],
        "close": [102, 110, 112],
        "volume": [1000, 1000, 1000]
    }
    df = pd.DataFrame(data)
    
    # 2. Define Strategy with Profit Target (10% gain)
    genome = {
        "name": "Test_Strat",
        "entry_rules": [],
        "exit_rules": [
            {"type": "profit_target", "val": 1.10} 
        ],
        "stop_loss_atr": 2.0,
        "time_stop": 10
    }
    
    strategy = GenericStrategy(genome)
    
    # 3. Test Case: Target NOT Hit
    # Entry at 100. Target is 110.
    # Row 0: High is 105. Should be False.
    entry_price = 100.0
    stop_price = 90.0
    
    res1 = strategy.exit(df, 0, 0, entry_price, stop_price)
    print(f"Row 0 (High 105, Target 110): Exit? {res1}")
    if res1:
        print("❌ FAILED: Exited too early.")
        return

    # 4. Test Case: Target HIT
    # Row 1: High is 112. Target is 110. Should be True.
    res2 = strategy.exit(df, 1, 0, entry_price, stop_price)
    print(f"Row 1 (High 112, Target 110): Exit? {res2}")
    if not res2:
        print("❌ FAILED: Did not exit on profit target.")
        return

    print("✅ SUCCESS: Profit Target logic verified.")

if __name__ == "__main__":
    test_profit_target()
