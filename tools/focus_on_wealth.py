import json
import os
import shutil
import datetime
import time

def focus_on_wealth():
    config_path = "config/generated_strategies.json"
    backup_path = f"config/strategies_backup_income_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 1. Backup Existing Config
    if os.path.exists(config_path):
        shutil.copy(config_path, backup_path)
        print(f"💾 Income Strategies Backed Up: {backup_path}")

    # 2. Define Gen 15 Template (The Limit Runner)
    wealth_strategy = {
        "name": "Apex Wealth (Gen 15)",
        "type": "wealth",
        "entry_rules": [],  # UNCHAINED
        "exit_rules": [{"type": "profit_target", "val": 1.15}],
        "stop_loss_atr": 3.0,
        "time_stop": 60,
        # OPTIMIZER PARAMETERS (Gen 15)
        "adx_threshold": 25.0,
        "rsi_strong": 40.0,
        "rsi_weak": 15.0,
        "limit_ratio": 0.98,  # Entry @ 2% Discount
        "trail_atr": 3.0,     # Trailing Stop
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
    print("☢️  INITIATING GEN 15 RESET...")
    with open(config_path, "w") as f:
        json.dump([wealth_strategy], f, indent=4)
    
    time.sleep(0.5)
    print("✅ SUCCESS: Config locked on Apex Wealth (Gen 15).")

if __name__ == "__main__":
    focus_on_wealth()
