import os
import re


def restore_trend_bonus():
    print("📈 RESTORING TREND BONUS (The final piece of the Golden State)...")
    
    engine_path = "execution/engine.py"
    if not os.path.exists(engine_path):
        print(f"❌ Error: {engine_path} not found.")
        return

    with open(engine_path, "r") as f:
        content = f.read()

    # The Logic: Velocity + Oversold + TREND + Tie-Breaker
    
    new_ranking_logic = """def calculate_backtest_quality_score(row, strategy_name):
    # GOLDEN STATE RANKING (Velocity + Trend + Deep Dip)
    score = 50.0
    
    # 1. Base Oversold (Dip Buying)
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 2. Velocity Booster (Prioritize High Volatility)
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
        
    # 3. Smart Trend Filter (The "Quality" Bias)
    # This was the missing link. Gen 12 prefers Uptrends.
    if row.get("close", 0) > row.get("sma200", 999999):
        score += 20
    
    # 4. Volume Support
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    return max(0.0, min(100.0, score))"""

    # Regex to replace the function
    pattern = r"def calculate_backtest_quality_score\(row, strategy_name\):[\s\S]*?return max\(0.0, min\(100.0, score\)\)"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_ranking_logic, content)
        with open(engine_path, "w") as f:
            f.write(new_content)
        print("   ✅ Engine Updated: Soft Trend Bonus (+20) Restored.")
    else:
        print("   ⚠️ Regex failed. Please check engine.py manually.")


if __name__ == "__main__":
    restore_trend_bonus()
