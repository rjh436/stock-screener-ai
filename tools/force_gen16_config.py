import json
import os
import time


def force_gen16_config():
    config_path = "config/generated_strategies.json"

    # Define Gen 16.1 Template (Staircase Hunter)
    strategy = {
        "name": "Apex Wealth (Gen 16.1)",
        "type": "wealth",
        "entry_rules": [],  # Unchained
        "exit_rules": [{"type": "profit_target", "val": 1.15}],
        "stop_loss_atr": 3.0,
        "time_stop": 60,
        # GEN 16.1 PARAMETERS
        "adx_threshold": 25.0,
        "rsi_strong": 40.0,
        "rsi_weak": 15.0,
        "limit_ratio": 0.98,       # 2% Discount Entry
        "trail_atr": 3.0,          # 3x ATR Trail
        "trail_activation": 1.03,  # (Unused by Staircase logic, but kept for compatibility)
        "use_bb_exit": False,
        "scoring_weights": {
            "sniper_bonus": 50.0,
            "rsi_factor": 2.0,
            "trend_bonus": 20.0,
            "vol_bonus": 10.0,
            "atr_high_bonus": 15.0
        }
    }

    print("Writing Gen 16.1 Config...")
    with open(config_path, "w") as f:
        json.dump([strategy], f, indent=4)

    time.sleep(0.5)

    # Verify
    with open(config_path, "r") as f:
        content = json.load(f)

    if len(content) > 0 and content[0]["name"] == "Apex Wealth (Gen 16.1)":
        print("✅ SUCCESS: Config forced to Gen 16.1")
    else:
        print("❌ FAILURE: Config write failed")


if __name__ == "__main__":
    force_gen16_config()
