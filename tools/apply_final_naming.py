import json
import os


def apply_final_naming():
    print("🏷️ APPLYING FINAL NAMING (Connecting Engine to Portfolio)...")
    
    config_path = "config/generated_strategies.json"
    
    # EXACT PARAMETERS from your uploaded file, just Renaming.
    portfolio = [
        {
            # OLD: Strategy_Apex_Gen12_Sniper
            # NEW: Apex Wealth (Gen 12) -> Triggers Sniper Logic (No Trend Filter)
            "name": "Apex Wealth (Gen 12)",
            "type": "hybrid",
            "entry_rules": [
                {"col": "cci", "op": "<", "val": 0},
                {"col": "bb_width", "op": ">", "val": 0.1},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [], 
            "stop_loss_atr": 4.4,
            "time_stop": 71 
        },
        {
            # OLD: Strategy_Apex_Gen9_Evolved
            # NEW: Apex Income (Gen 9) -> Triggers Income Logic (+20 Trend Bonus)
            "name": "Apex Income (Gen 9)",
            "type": "hybrid",
            "entry_rules": [
                {"col": "close", "op": ">", "val": 0}, 
                {"col": "bb_width", "op": ">", "val": 0.17},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [
                {"type": "profit_target", "val": 1.08}
            ],
            "stop_loss_atr": 5.1,
            "time_stop": 45
        }
    ]
    
    with open(config_path, "w") as f:
        json.dump(portfolio, f, indent=4)
        
    print("   ✅ Renaming Complete.")
    print("   - 'Apex Income' will now correctly trigger the Trend Bonus.")
    print("   - 'Apex Wealth' will correctly trigger the Sniper Priority.")


if __name__ == "__main__":
    apply_final_naming()
