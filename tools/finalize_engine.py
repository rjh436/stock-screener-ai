import os
import re


def finalize_engine():
    print("🚀 REMOVING TREND BIAS (Restoring Explosive Ranking)...")
    
    engine_path = "execution/engine.py"
    if not os.path.exists(engine_path):
        print(f"❌ Error: {engine_path} not found.")
        return

    with open(engine_path, "r") as f:
        content = f.read()

    # The Logic: Pure Velocity + Oversold + Tie-Breaker
    # We deliberately REMOVE the 'Smart Trend Filter' block.
    
    new_ranking_logic = """def calculate_backtest_quality_score(row, strategy_name):
    # GOLDEN STATE RANKING (Pure Velocity + Deep Dip)
    score = 50.0
    
    # 1. Base Oversold (Dip Buying)
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 2. Velocity Booster (Prioritize High Volatility)
    # This pushes the most explosive stocks to the top.
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15  # Rocket fuel
        elif atr_pct > 2.0: score += 5
        
    # NOTE: Trend Filter removed to allow Deep Value crashes (Gen 12 logic)
    
    # 3. Volume Support
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    return max(0.0, min(100.0, score))"""

    # We also need to ensure the sorting logic (Tie-Breaker) is correct in run_backtest
    # Check for the sort line
    sort_target = 'daily_candidates.sort(key=lambda x: (x["score"], -x["rsi2"]), reverse=True)'
    
    # Regex to replace the ranking function
    pattern = r"def calculate_backtest_quality_score\(row, strategy_name\):[\s\S]*?return max\(0.0, min\(100.0, score\)\)"
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, new_ranking_logic, content)
        
        # Verify Tie-Breaker Exists
        if sort_target not in new_content:
            print("   ⚠️ Tie-Breaker missing! Injecting...")
            # Attempt to fix the sort line if it reverted
            old_sort = 'daily_candidates.sort(key=lambda x: x["score"], reverse=True)'
            new_content = new_content.replace(old_sort, sort_target)
            
        with open(engine_path, "w") as f:
            f.write(new_content)
        print("   ✅ Engine Updated: Trend Bias Removed. Pure Velocity Scoring + Tie-Breaker Active.")
    else:
        print("   ⚠️ Regex failed. Please check engine.py manually.")


if __name__ == "__main__":
    finalize_engine()
