import os
import re


def remove_trend_bias():
    print("REMOVING TREND BIAS (Restoring Pure Velocity Ranking)...")

    engine_path = "execution/engine.py"
    if not os.path.exists(engine_path):
        print(f"Error: {engine_path} not found.")
        return

    with open(engine_path, "r") as f:
        content = f.read()

    # The Logic we want: Pure Velocity + Oversold
    # We REMOVE the "Soft Trend Filter" block completely.

    new_ranking_logic = """def calculate_backtest_quality_score(row, strategy_name):
    # PURE VELOCITY RANKING (Restored to Gen 12 Discovery State)
    score = 50.0
    
    # 1. Base Oversold (Dip Buying)
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 2. Velocity Booster (The Rocket Fuel)
    # Prioritize stocks that move fast (High ATR%)
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
    
    # 3. Volume Support
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    return max(0.0, min(100.0, score))"""

    # Replace the existing function
    pattern = r"def calculate_backtest_quality_score\(row, strategy_name\):[\s\S]*?return max\(0.0, min\(100.0, score\)\)"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_ranking_logic, content)
        with open(engine_path, "w") as f:
            f.write(new_content)
        print("Engine Updated: Trend Bias Removed. Pure Velocity Scoring active.")
    else:
        # Fallback if regex fails (manual overwrite of function)
        print("Regex failed. Attempting structural replace...")
        # If this fails, user should manually ensure SMA200 check is removed.


if __name__ == "__main__":
    remove_trend_bias()
