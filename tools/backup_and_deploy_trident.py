import json
import os
import shutil
import datetime


def backup_and_deploy_trident():
    # 1. CREATE BACKUP
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backups/PRE_OPTIMIZATION_{timestamp}"
    
    if not os.path.exists(backup_name):
        os.makedirs(backup_name)
    
    files_to_save = [
        "config/generated_strategies.json",
        "execution/engine.py",
        "optimization/evolution.py",
        "optimization/optimizer.py",
        "app.py"
    ]
    
    print(f"📦 CREATING BACKUP: {backup_name}...")
    for f in files_to_save:
        if os.path.exists(f):
            shutil.copy(f, backup_name)
            print(f"   ✅ Saved: {f}")

    # 2. DEPLOY TRIDENT PORTFOLIO
    print("\n🔱 DEPLOYING TRIDENT CONFIGURATION...")
    config_path = "config/generated_strategies.json"
    
    strategies = [
        # 1. THE CLASSIC (Wealth Builder)
        {
            "name": "Strategy_Apex_Gen12_Sniper",
            "type": "hybrid",
            "entry_rules": [
                {"col": "cci", "op": "<", "val": 0},
                {"col": "bb_width", "op": ">", "val": 0.1},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [], # No Target
            "stop_loss_atr": 4.4,
            "time_stop": 71 
        },
        # 2. THE EVOLVED (High Precision Gen 9)
        {
            "name": "Strategy_Apex_Gen9_Evolved",
            "type": "hybrid",
            "entry_rules": [
                {"col": "cci", "op": "<", "val": 0},
                {"col": "bb_width", "op": ">", "val": 0.1},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [
                {"type": "profit_target", "val": 1.08} # 8% Target (The 82% Win Rate Secret)
            ],
            "stop_loss_atr": 5.3, # Looser stop as evolved
            "time_stop": 71
        },
        # 3. THE MACHINE GUN (Cash Flow Baseline)
        {
            "name": "Strategy_Apex_Alpha_MachineGun",
            "type": "hybrid",
            "entry_rules": [
                {"col": "adx", "op": "<", "ref": "rsi14"},
                {"col": "volume", "op": ">", "val": 0},
                {"col": "sma50", "op": ">", "val": 0}
            ],
            "exit_rules": [
                {"type": "profit_target", "val": 1.06}
            ],
            "stop_loss_atr": 4.3,
            "time_stop": 10
        }
    ]
    
    with open(config_path, "w") as f:
        json.dump(strategies, f, indent=4)
        
    print("   ✅ Configuration Updated: Gen 12 + Gen 9 + Machine Gun.")
    print("   🚀 Ready for Backtest.")


if __name__ == "__main__":
    backup_and_deploy_trident()
