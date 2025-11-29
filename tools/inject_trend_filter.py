import json
import os

CONFIG_PATH = "config/generated_strategies.json"

def inject_trend_filter():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: {CONFIG_PATH} not found.")
        return

    with open(CONFIG_PATH, "r") as f:
        strategies = json.load(f)

    print(f"🔍 Analyzing {len(strategies)} strategies...")
    
    modified = False
    
    for strat in strategies:
        if strat["name"] == "Strategy_Apex_Alpha_MachineGun":
            print(f"🎯 Found Machine Gun Strategy. Analyzing rules...")
            
            # Check if rule already exists
            has_trend = False
            for rule in strat["entry_rules"]:
                if rule.get("col") == "close" and rule.get("ref") == "sma200" and rule.get("op") == ">":
                    has_trend = True
                    break
            
            if not has_trend:
                # Inject the Golden Rule: Close > SMA200
                new_rule = {"col": "close", "op": ">", "ref": "sma200"}
                strat["entry_rules"].append(new_rule)
                print(f"✅ Injected Trend Filter: Close > SMA200")
                modified = True
            else:
                print("ℹ️ Strategy already has Trend Filter.")

    if modified:
        # Backup
        os.rename(CONFIG_PATH, CONFIG_PATH + ".bak_trend")
        print(f"📂 Backed up original to {CONFIG_PATH}.bak_trend")
        
        # Save
        with open(CONFIG_PATH, "w") as f:
            json.dump(strategies, f, indent=4)
        print("💾 Strategy Updated Successfully.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    inject_trend_filter()
