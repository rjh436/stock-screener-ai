import json
import os


def restore_proven_strategies():
    print("💎 EXECUTING FORENSIC RESTORE (From CSV Data)...")
    
    config_path = "config/generated_strategies.json"
    
    # DATA SOURCE: 2025-11-30T15-21_export.csv (Row 0)
    # This is the strategy that achieved 49% CAGR / 18% Avg Profit
    gen12_sniper = {
        "name": "Strategy_Apex_Gen12_Sniper",
        "type": "hybrid",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},          # Momentum Dip
            {"col": "bb_width", "op": ">", "val": 0.1},   # Volatility Filter (The Missing Link)
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},# Oversold Confluence
            {"col": "volume", "op": ">", "ref": "vol_ma20"} # Volume Confirmation
        ],
        "exit_rules": [], # No Target (Wealth Builder)
        "stop_loss_atr": 4.4,
        "time_stop": 71 
    }

    # DATA SOURCE: 2025-11-30T16-34_export.csv (Row 1)
    # This is the strategy that achieved 43% CAGR / 1.5% Avg Profit
    machine_gun = {
        "name": "Strategy_Apex_Alpha_MachineGun",
        "type": "hybrid",
        "entry_rules": [
            {"col": "adx", "op": "<", "ref": "rsi14"},
            {"col": "volume", "op": ">", "val": 0},
            {"col": "sma50", "op": ">", "val": 0}
        ],
        "exit_rules": [
            {"type": "profit_target", "val": 1.06} # 6% Bank (Cash Flow)
        ],
        "stop_loss_atr": 4.3,
        "time_stop": 10
    }

    final_portfolio = [gen12_sniper, machine_gun]
    
    with open(config_path, "w") as f:
        json.dump(final_portfolio, f, indent=4)
        
    print(f"   ✅ Restored Exact Gen 12 Logic (CCI/BB) and Machine Gun (ADX/Target).")
    print("   🚀 System is now mathematically identical to your best runs.")

if __name__ == "__main__":
    restore_proven_strategies()
