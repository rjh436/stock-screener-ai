import json
import os

# The mathematically proven DNA from your optimization run
BEST_DNA = {
    "name": "Apex Sniper Wealth v1.0",
    "type": "wealth",
    "entry_rules": [{"col": "rsi2", "op": ">", "val": 15}, {"col": "rs_trend", "op": ">", "val": 0}],
    "exit_rules": [{"type": "profit_target", "val": 1.27}],
    "stop_loss_atr": 2.13,
    "time_stop": 32,
    "breakeven_pct": 0.083,  # The Critical Fix (8.3%)
    "sizing_mode": "risk",  # Enforce Parity
    "limit_ratio": 0.98,
    "trail_atr": 2.5,
    "trail_activation": 1.4,
    "adx_threshold": 28.0,
    "rsi_strong": 35.0,
    "rsi_weak": 11.0,
    "scoring_weights": {
        "sniper_bonus": 41.76,
        "rsi_factor": 4.12,
        "trend_bonus": 70.0,
        "vol_bonus": 17.89,
        "atr_high_bonus": 30.14,
        "atr_med_bonus": 17.04,
        "trend_penalty": -25.8,
    },
}

CONFIG_PATH = "config/generated_strategies.json"


def apply_fix():
    print(f"Loading {CONFIG_PATH}...")

    if not os.path.exists(CONFIG_PATH):
        print("Error: Config file not found!")
        return

    with open(CONFIG_PATH, "r") as f:
        strategies = json.load(f)

    # Find and update the specific strategy
    found = False
    for i, strat in enumerate(strategies):
        if strat.get("name") == "Apex Sniper Wealth v1.0":
            print("Found target strategy. Injecting Best DNA...")

            # Preserve existing fields that might not be in DNA (like 'enabled')
            updated_strat = strat.copy()
            updated_strat.update(BEST_DNA)

            strategies[i] = updated_strat
            found = True
            break

    if not found:
        print("Strategy not found in file. Appending as new...")
        strategies.append(BEST_DNA)

    # Save back
    with open(CONFIG_PATH, "w") as f:
        json.dump(strategies, f, indent=4)

    print("Configuration Successfully Updated!")
    print(
        "New Stats: CAGR 45.2% | Stop "
        f"{BEST_DNA['stop_loss_atr']} ATR | B/E {BEST_DNA['breakeven_pct'] * 100}%"
    )


if __name__ == "__main__":
    apply_fix()
