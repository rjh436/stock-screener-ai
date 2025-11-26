import json
import os

CONFIG_PATH = "config/generated_strategies.json"


def clean_and_rename():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: {CONFIG_PATH} not found.")
        return

    with open(CONFIG_PATH, "r") as f:
        strategies = json.load(f)

    print(f"🔍 Analyzing {len(strategies)} strategies...")

    # Define the targets we want to keep and their new names
    # Based on Backtest Analysis:
    # 1. Gen7_Fresh_9928_mut_mut -> The "Sniper" (VIX > 33, CCI < 0)
    # 2. Cross_Gen7_Gen7       -> The "Momentum" (Stoch > RSI, VIX > 33)
    targets = {
        "Gen7_Fresh_9928_mut_mut": "Strategy_Apex_VIX_Sniper",
        "Cross_Gen7_Gen7": "Strategy_Apex_Stoch_RSI",
    }

    kept_strategies = []
    seen_names = set()

    for strat in strategies:
        old_name = strat.get("name")

        if old_name in targets:
            new_name = targets[old_name]

            # Deduplication check
            if new_name in seen_names:
                continue

            print(f"✅ Keeping & Renaming: {old_name} -> {new_name}")
            strat["name"] = new_name
            kept_strategies.append(strat)
            seen_names.add(new_name)

    if not kept_strategies:
        print("⚠️  Warning: No target strategies found in the file. Check exact names.")
        return

    # Backup original
    os.rename(CONFIG_PATH, CONFIG_PATH + ".bak_full")
    print(f"📂 Backed up original to {CONFIG_PATH}.bak_full")

    # Save cleaned version
    with open(CONFIG_PATH, "w") as f:
        json.dump(kept_strategies, f, indent=4)

    print(f"💾 Saved {len(kept_strategies)} clean strategies to {CONFIG_PATH}")


if __name__ == "__main__":
    clean_and_rename()
