import json
import os

def apply_precision_upgrade():
    print("🎯 Initializing Operation Precision...")

    # --- STEP 1: INJECT PROFIT TARGET (Config) ---
    config_path = "config/generated_strategies.json"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            strategies = json.load(f)
        
        modified = False
        for strat in strategies:
            if strat.get("name") == "Strategy_Apex_Alpha_MachineGun":
                print("   👉 updating 'Machine Gun' Strategy...")
                
                # Check if profit target already exists
                has_target = any(r.get("type") == "profit_target" for r in strat.get("exit_rules", []))
                
                if not has_target:
                    # Inject 6% Profit Target (Bank the bounce)
                    strat["exit_rules"].append({
                        "type": "profit_target",
                        "val": 1.06
                    })
                    print("      ✅ Injected 6% Profit Target (Exit at +6%).")
                    modified = True
                else:
                    print("      ℹ️ Profit target already present.")
        
        if modified:
            with open(config_path, "w") as f:
                json.dump(strategies, f, indent=4)
            print("   💾 Configuration saved.")
    else:
        print(f"   ❌ Critical: {config_path} not found.")

    # --- STEP 2: UPGRADE RANKING LOGIC (Engine) ---
    engine_path = "execution/engine.py"
    if os.path.exists(engine_path):
        with open(engine_path, "r") as f:
            content = f.read()
        
        # We define the NEW logic block. 
        # Ideally, we find the block we wrote in the previous step and enhance it.
        
        # Target the 'Velocity Booster' block we added previously
        signature = "# NEW: Velocity Booster (Prioritize High Volatility Stocks)"
        
        new_logic_block = """        # NEW: Velocity Booster (Prioritize High Volatility Stocks)
        # Higher ATR% means the stock moves fast -> hits profit targets quicker
        close_px = row.get(\"close\", 1.0)
        if close_px > 0:
            atr_pct = (row.get(\"atr14\", 0) / close_px) * 100
            if atr_pct > 3.0: score += 15  # High Velocity
            elif atr_pct > 2.0: score += 5
            
        # NEW: Smart Trend Filter (Soft Filter)
        # We don't ban downtrends, but we heavily favor uptrends.
        # This pushes 'Good Stocks on Bad Days' to the top of the list.
        if row.get(\"close\", 0) > row.get(\"sma200\", 999999):
            score += 20"""

        if signature in content:
            # We assume the previous block structure exists. We will replace the Velocity block
            # with the Enhanced Velocity + Trend block.
            
            # Use a slightly wider match to ensure we replace the old logic cleanly
            old_block_start = "        # NEW: Velocity Booster (Prioritize High Volatility Stocks)"
            # We look for the next distinct block start to bound the replacement
            next_block_start = "        vol_rel = row.get" 
            
            if old_block_start in content and next_block_start in content:
                # Splitting logic to surgically insert
                parts = content.split(old_block_start)
                pre_block = parts[0]
                # Find the rest after the logic we want to replace
                post_block_parts = parts[1].split(next_block_start)
                
                if len(post_block_parts) >= 2:
                    # Reassemble: Pre + New Logic + Rest
                    # Note: We strip the old velocity logic by taking the LAST part of the split
                    # (effectively removing everything between OLD_START and NEXT_START)
                    rest_of_code = next_block_start + post_block_parts[1]
                    
                    new_content = pre_block + new_logic_block + "\n        \n" + rest_of_code
                    
                    with open(engine_path, "w") as f:
                        f.write(new_content)
                    print("   ✅ Engine updated: Added 'Soft Trend' Ranking (+20 Score for Uptrend).")
                else:
                    print("   ⚠️ Parsing error: Could not locate end of logic block.")
            else:
                print("   ⚠️ Could not find exact code anchors in engine.py.")
        else:
            print("   ⚠️ Could not find Velocity Booster signature. Did you run the previous upgrade?")

    print("\n✅ Operation Precision Complete. Re-run Backtest.")

if __name__ == "__main__":
    apply_precision_upgrade()
