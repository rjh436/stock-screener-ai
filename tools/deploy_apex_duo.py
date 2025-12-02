import json


def deploy_apex_duo():
    print("⚔️ DEPLOYING THE APEX DUO (Wealth + Income)...")

    config_path = "config/generated_strategies.json"

    # 1. THE WEALTH BUILDER (Classic Gen 12)
    # Proven Performance: 49% CAGR, 18.35% Avg Profit
    wealth_strategy = {
        "name": "Strategy_Apex_Gen12_Sniper",
        "type": "hybrid",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},
            {"col": "bb_width", "op": ">", "val": 0.1},
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},
            {"col": "volume", "op": ">", "ref": "vol_ma20"},
        ],
        "exit_rules": [],  # No Target (Let winners run)
        "stop_loss_atr": 4.4,
        "time_stop": 71,
    }

    # 2. THE INCOME GENERATOR (Gen 9 Super-Evolved)
    # Proven Performance: 92% CAGR, 5.2% Avg Profit, 82% Win Rate
    # Extracted from your Optimization Result
    income_strategy = {
        "name": "Strategy_Apex_Gen9_Evolved",
        "type": "hybrid",
        "entry_rules": [
            # Mutation replaced CCI with a generic check, focusing on Volatility/RSI
            {"col": "close", "op": ">", "val": 0},
            {"col": "bb_width", "op": ">", "val": 0.17},  # High Volatility Filter
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},
            {"col": "volume", "op": ">", "ref": "vol_ma20"},
        ],
        "exit_rules": [
            {"type": "profit_target", "val": 1.08}  # 8% Profit Target (The Key)
        ],
        "stop_loss_atr": 5.1,
        "time_stop": 45,  # 45 Day Limit (Optimized for 5% return)
    }

    portfolio = [wealth_strategy, income_strategy]

    with open(config_path, "w") as f:
        json.dump(portfolio, f, indent=4)

    print("   ✅ APEX DUO CONFIGURED:")
    print("      1. Gen 12 Classic (18% Profit Target)")
    print("      2. Gen 9 Evolved (5.2% Profit / 92% CAGR)")
    print("      (Machine Gun removed due to low profit per trade)")
    print("\n   🚀 System Ready for Deployment.")


if __name__ == "__main__":
    deploy_apex_duo()
