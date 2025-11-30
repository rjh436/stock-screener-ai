import json
import os
import re

def hotfix_system():
    print("🚑 Starting Emergency Hotfix...")

    # --- 1. PATCH CONFIGURATION (Restore Precision) ---
    config_path = "config/generated_strategies.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            strategies = json.load(f)
        
        for strat in strategies:
            # FIX MACHINE GUN
            if "MachineGun" in strat.get("name", ""):
                print("   🔧 Patching Machine Gun...")
                # Force Profit Target
                strat["exit_rules"] = [r for r in strat.get("exit_rules", []) if r.get("type") != "profit_target"]
                strat["exit_rules"].append({"type": "profit_target", "val": 1.06})
                strat["time_stop"] = 10
                print("      ✅ Restored: 6% Profit Target + 10d Time Stop.")

            # AUDIT GEN 12 SNIPER
            if "Gen12" in strat.get("name", ""):
                print(f"   🧐 Auditing Sniper ({strat['name']})...")
                print(f"      - Rules: {len(strat['entry_rules'])}")
                print(f"      - Stop: {strat.get('stop_loss_atr')} ATR")
                print(f"      - Time: {strat.get('time_stop')} Days")

        with open(config_path, "w") as f:
            json.dump(strategies, f, indent=4)
    
    # --- 2. RESTORE ENGINE LOGIC (Precision Ranking) ---
    engine_path = "execution/engine.py"
    if os.path.exists(engine_path):
        # We enforce the 'Soft Trend' + 'Velocity' ranking block
        ranking_logic = """    else: 
        # MACHINE GUN MODE: Deep Dips + High Volatility (Velocity)
        rsi2 = row.get(\"rsi2\", 50)
        score += (100 - rsi2) * 2.0  # Base: How deep is the dip?
        
        # NEW: Velocity Booster (Prioritize High Volatility Stocks)
        close_px = row.get(\"close\", 1.0)
        if close_px > 0:
            atr_pct = (row.get(\"atr14\", 0) / close_px) * 100
            if atr_pct > 3.0: score += 15  # High Velocity
            elif atr_pct > 2.0: score += 5
            
        # NEW: Smart Trend Filter (Soft Filter)
        if row.get(\"close\", 0) > row.get(\"sma200\", 999999):
            score += 20
        
        vol_rel = row.get(\"volume\", 0) / (row.get(\"vol_ma20\", 1) + 1)
        if vol_rel > 1.5: score += 10"""

        with open(engine_path, "r") as f:
            content = f.read()
        
        # Regex to replace the 'else' block of the scoring function
        pattern = r"(else:\s+rsi2 = row\.get\(\"rsi2\", 50\).*?if vol_rel > 1\.5: score \+= 10)"
        
        if "Smart Trend Filter" not in content:
            match = re.search(r"else:\s+# MACHINE GUN MODE.*?if vol_rel > 1.5: score \+= 10", content, re.DOTALL)
            if match:
                new_content = content.replace(match.group(0), ranking_logic)
                with open(engine_path, "w") as f:
                    f.write(new_content)
                print("   ✅ Engine Logic Patched (Soft Trend + Velocity Restored).")
            else:
                print("   ⚠️ Could not auto-patch engine.py. Please verify ranking logic manually.")
        else:
            print("   ✅ Engine Logic already correct.")

    print("\n✅ Hotfix Complete. Run Backtest immediately.")

if __name__ == "__main__":
    hotfix_system()
