import json
import os


def restore_and_unchain():
    config_path = "config/generated_strategies.json"
    
    # 1. Load Config
    if not os.path.exists(config_path):
        print("❌ Config not found! Creating new one.")
        strategies = []
    else:
        with open(config_path, "r") as f:
            strategies = json.load(f)

    # 2. Check if Wealth exists
    wealth_exists = False
    for strat in strategies:
        if "Wealth" in strat["name"]:
            wealth_exists = True
            # Fix it just in case it exists but is broken
            strat["entry_rules"] = [] 
            strat["adx_threshold"] = 25.0
            strat["rsi_strong"] = 40.0
            strat["rsi_weak"] = 15.0
            strat["use_scalp_exit"] = True
            print("🔧 Found existing Wealth strategy. Repaired and Unchained.")
            break
    
    # 3. Inject if Missing (The likely case)
    if not wealth_exists:
        print("🚑 Apex Wealth is missing! Injecting Gen 14 Template...")
        new_wealth = {
            "name": "Apex Wealth (Gen 14)",
            "type": "wealth",
            "entry_rules": [],  # UNCHAINED (Empty to let Python logic drive)
            "exit_rules": [{"type": "profit_target", "val": 1.15}],
            "stop_loss_atr": 3.0,
            "time_stop": 60,
            # OPTIMIZER PARAMETERS
            "adx_threshold": 25.0,
            "rsi_strong": 40.0,
            "rsi_weak": 15.0,
            "use_scalp_exit": True,
            "use_bb_exit": False,
            "scoring_weights": {
                "sniper_bonus": 50.0,
                "rsi_factor": 2.0,
                "trend_bonus": 20.0
            }
        }
        # Insert at the top so it gets priority
        strategies.insert(0, new_wealth)

    # 4. Save
    with open(config_path, "w") as f:
        json.dump(strategies, f, indent=4)
    
    print("✅ restoration complete. 'Apex Wealth (Gen 14)' is active and unchained.")


if __name__ == "__main__":
    restore_and_unchain()
