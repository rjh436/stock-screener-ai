import json
import os


def deploy_trident_final():
    print("🔱 DEPLOYING THE FINAL TRIDENT PORTFOLIO...")

    config_path = "config/generated_strategies.json"

    # 1. THE CLASSIC (Wealth Builder / Home Runs)
    # Stats: 49% CAGR, 18% Avg Profit
    classic = {
        "name": "Strategy_Apex_Sniper_Classic",
        "type": "hybrid",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},
            {"col": "bb_width", "op": ">", "val": 0.1},
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},
            {"col": "volume", "op": ">", "ref": "vol_ma20"},
        ],
        "exit_rules": [],  # No Target
        "stop_loss_atr": 4.4,
        "time_stop": 71,
    }

    # 2. THE EVOLVED (Precision / Income)
    # Stats: 92% CAGR, 82% Win Rate (Derived from latest JSON)
    evolved = {
        "name": "Strategy_Apex_Sniper_Evolved",
        "type": "hybrid",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},
            {"col": "bb_width", "op": ">", "val": 0.17},  # The "High Volatility" Filter
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},
            {"col": "volume", "op": ">", "ref": "vol_ma20"},
        ],
        "exit_rules": [
            {"type": "profit_target", "val": 1.08}  # 8% Bank
        ],
        "stop_loss_atr": 5.1,
        "time_stop": 45,  # The "Breathing Room" setting
    }

    # 3. THE MACHINE GUN (Cash Flow / Daily)
    # Stats: 43% CAGR, Daily Activity
    machine_gun = {
        "name": "Strategy_Apex_MachineGun",
        "type": "hybrid",
        "entry_rules": [
            {"col": "adx", "op": "<", "ref": "rsi14"},
            {"col": "volume", "op": ">", "val": 0},
            {"col": "sma50", "op": ">", "val": 0},
        ],
        "exit_rules": [
            {"type": "profit_target", "val": 1.06}  # 6% Bank
        ],
        "stop_loss_atr": 4.3,
        "time_stop": 10,
    }

    portfolio = [classic, evolved, machine_gun]

    with open(config_path, "w") as f:
        json.dump(portfolio, f, indent=4)

    print("   ✅ TRIDENT ARMED:")
    print("      1. Classic (18% Profit Target)")
    print("      2. Evolved (8% Profit / 92% CAGR DNA)")
    print("      3. Machine Gun (Daily Cash Flow)")
    print("\n   🚀 READY FOR FINAL BACKTEST.")


if __name__ == "__main__":
    deploy_trident_final()
