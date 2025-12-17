import json
import os
import shutil
import datetime

def force_reset():
    config_path = "config/generated_strategies.json"
    
    # 1. Backup existing (just in case)
    if os.path.exists(config_path):
        backup_name = f"config/strategies_backup_scalper_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy(config_path, backup_name)
        print(f"💾 Backup saved: {backup_name}")

    # 2. Define The Gen 16.2 Template (Delayed Bridge)
    gen_16_2 = {
        "name": "Apex Wealth (Gen 16.2)",
        "type": "wealth",
        "entry_rules": [],
        "exit_rules": [{"type": "profit_target", "val": 1.25}],
        "stop_loss_atr": 3.0,
        "time_stop": 60,
        # OPTIMIZER PARAMETERS
        "adx_threshold": 25.0,
        "rsi_strong": 40.0,
        "rsi_weak": 15.0,
        "limit_ratio": 0.98,
        "trail_atr": 4.0,      # Start looser (4.0) to encourage holding
        "trail_activation": 1.03, 
        "scoring_weights": {
            "sniper_bonus": 60.0,
            "rsi_factor": 3.0,
            "trend_bonus": 30.0,
            "vol_bonus": 10.0,
            "atr_high_bonus": 20.0
        }
    }

    # 3. Overwrite Config
    print("☢️  PURGING OLD POPULATION...")
    with open(config_path, "w") as f:
        json.dump([gen_16_2], f, indent=4)
    
    print("✅ Gen 16.2 Template Installed. You may now run the optimizer.")

if __name__ == "__main__":
    force_reset()

