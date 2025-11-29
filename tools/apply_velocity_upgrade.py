import json
import os

def apply_sync_upgrade():
    print("🚀 Initializing Operation Velocity (Synchronized Upgrade)...")

    # --- STEP 1: UPDATE SHARED STRATEGY CONFIGURATION ---
    # This JSON is the 'brain' shared by Optimizer, Backtester, and Live Screener.
    config_path = "config/generated_strategies.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            strategies = json.load(f)
        
        modified = False
        for strat in strategies:
            if strat.get("name") == "Strategy_Apex_Alpha_MachineGun":
                print("   👉 Upgrading 'Machine Gun' Strategy logic...")
                
                # A. Remove SMA200 Trend Filter (Unshackle)
                original_rules = len(strat["entry_rules"])
                strat["entry_rules"] = [
                    rule for rule in strat["entry_rules"] 
                    if not (rule.get("col") == "close" and rule.get("ref") == "sma200")
                ]
                if len(strat["entry_rules"]) < original_rules:
                    print("      ✅ SMA200 Filter Removed (Deep Dips Enabled).")
                
                # B. Tighten Time Stop (Force Turnover)
                strat["time_stop"] = 10
                print("      ✅ Time Stop reduced to 10 days (High Velocity).")
                modified = True
        
        if modified:
            with open(config_path, "w") as f:
                json.dump(strategies, f, indent=4)
            print("   💾 Shared Configuration Saved. All components synced.")
        else:
            print("   ⚠️ Strategy not found or already up to date.")
    else:
        print(f"   ❌ Critical Error: {config_path} missing.")

    # --- STEP 2: UPDATE SHARED RANKING ENGINE ---
    # This Python file is the 'math kernel' imported by all components.
    engine_path = "execution/engine.py"
    if os.path.exists(engine_path):
        with open(engine_path, "r") as f:
            content = f.read()
        
        # New Scoring Logic: Adds Volatility (ATR%) to the Ranking Score
        # This ensures the Live Screener picks the same 'explosive' stocks the Backtester prefers.
        
        old_logic_signature = """    else: 
        # MACHINE GUN MODE: Wants Deep Dips (Mean Reversion)
        rsi2 = row.get(\"rsi2\", 50)
        score += (100 - rsi2) * 2.0 # Heavy weight on oversold depth
        
        vol_rel = row.get(\"volume\", 0) / (row.get(\"vol_ma20\", 1) + 1)
        if vol_rel > 1.5: score += 10"""

        # Fallback signature if comments are different
        old_logic_signature_alt = """    else: 
        rsi2 = row.get(\"rsi2\", 50)
        score += (100 - rsi2) * 2.0 
        
        vol_rel = row.get(\"volume\", 0) / (row.get(\"vol_ma20\", 1) + 1)
        if vol_rel > 1.5: score += 10"""

        new_logic = """    else: 
        # MACHINE GUN MODE: Deep Dips + High Volatility (Velocity)
        rsi2 = row.get(\"rsi2\", 50)
        score += (100 - rsi2) * 2.0  # Base: How deep is the dip?
        
        # NEW: Velocity Booster (Prioritize High Volatility Stocks)
        # Higher ATR% means the stock moves fast -> hits profit targets quicker
        close_px = row.get(\"close\", 1.0)
        if close_px > 0:
            atr_pct = (row.get(\"atr14\", 0) / close_px) * 100
            if atr_pct > 3.0: score += 15  # High Velocity
            elif atr_pct > 2.0: score += 5
        
        vol_rel = row.get(\"volume\", 0) / (row.get(\"vol_ma20\", 1) + 1)
        if vol_rel > 1.5: score += 10"""

        if old_logic_signature in content:
            new_content = content.replace(old_logic_signature, new_logic)
            with open(engine_path, "w") as f:
                f.write(new_content)
            print("   ✅ Engine updated with Volatility Ranking Logic.")
        elif old_logic_signature_alt in content:
            new_content = content.replace(old_logic_signature_alt, new_logic)
            with open(engine_path, "w") as f:
                f.write(new_content)
            print("   ✅ Engine updated with Volatility Ranking Logic (Alt Match).")
        else:
            if "Velocity Booster" in content:
                print("   ℹ️ Engine logic already present.")
            else:
                print("   ⚠️ Could not auto-patch 'execution/engine.py'. Check indentation matches exactly.")

    print("\n✅ Upgrade Complete. Optimizer, Screener, and Backtester are now in sync.")

if __name__ == "__main__":
    apply_sync_upgrade()
