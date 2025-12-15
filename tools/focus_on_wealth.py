import json
import os
import shutil
import datetime
import time


def focus_on_wealth():
    config_path = "config/generated_strategies.json"
    backup_path = f"config/strategies_backup_income_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 1. Create Backup (Save the Income Strategy!)
    if os.path.exists(config_path):
        shutil.copy(config_path, backup_path)
        print(f"💾 Income Strategies Backed Up: {backup_path}")

    # 2. Define The Wealth Template (Gen 14)
    wealth_strategy = {
        "name": "Apex Wealth (Gen 14)",
        "type": "wealth",
        "entry_rules": [],  # UNCHAINED: Python logic drives everything
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
            "trend_bonus": 20.0,
            "vol_bonus": 10.0,
            "atr_high_bonus": 15.0
        }
    }

    # 3. FORCE LOCK (Overwrite)
    print("☢️  INITIATING WEALTH FOCUS...")
    with open(config_path, "w") as f:
        json.dump([wealth_strategy], f, indent=4)
    
    # 4. Verify
    time.sleep(0.5)
    with open(config_path, "r") as f:
        content = json.load(f)
    
    print("\n🔍 VERIFICATION:")
    if len(content) == 1 and "Wealth" in content[0]["name"]:
        print("✅ SUCCESS: Config is locked on Apex Wealth.")
        print("🚀 You may now run the optimizer.")
    else:
        print("❌ FAILURE: Config write failed.")


if __name__ == "__main__":
    focus_on_wealth()
