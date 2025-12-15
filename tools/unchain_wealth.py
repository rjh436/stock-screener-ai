import json
import os


def unchain_wealth():
    config_path = "config/generated_strategies.json"
    
    if not os.path.exists(config_path):
        print("❌ Config not found!")
        return

    with open(config_path, "r") as f:
        strategies = json.load(f)

    updated = False
    for strat in strategies:
        # Target Apex Wealth specifically
        if "Wealth" in strat["name"]:
            print(f"🔓 Unchaining: {strat['name']}")
            
            # 1. PURGE OLD RULES (Remove the Double Filter)
            # We want the Python 'Dual Lane' logic to be the ONLY filter.
            strat["entry_rules"] = []
            
            # 2. INJECT NEW PARAMETERS (If missing)
            # This ensures the Optimizer sees them immediately.
            if "adx_threshold" not in strat:
                strat["adx_threshold"] = 25.0
            if "rsi_strong" not in strat:
                strat["rsi_strong"] = 40.0
            if "rsi_weak" not in strat:
                strat["rsi_weak"] = 15.0
            if "use_scalp_exit" not in strat:
                strat["use_scalp_exit"] = True
                
            updated = True

    if updated:
        with open(config_path, "w") as f:
            json.dump(strategies, f, indent=4)
        print("✅ Apex Wealth unchained. Old rules purged, new params injected.")
    else:
        print("⚠️ No 'Wealth' strategy found to update.")


if __name__ == "__main__":
    unchain_wealth()
