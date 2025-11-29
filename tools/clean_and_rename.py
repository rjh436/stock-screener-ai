import json
import os

CONFIG_PATH = "config/generated_strategies.json"


def clean_and_rename():
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Error: {CONFIG_PATH} not found.")
        return

    with open(CONFIG_PATH, "r") as f:
        strategies = json.load(f)

    print(f"🔍 Scanning {len(strategies)} strategies in database...")

    # Define the Portfolio Targets
    target_map = {
        # 1. The Sniper (Shield) - Keep existing
        "Strategy_Apex_VIX_Sniper": "Strategy_Apex_VIX_Sniper",

        # 2. The New Machine Gun (Sword) - Upgrade to the 55% CAGR Winner
        "Cross_Cross_Strategy_mut": "Strategy_Apex_Alpha_MachineGun",

        # Fallback: If exact name isn't found, check for previous version to avoid losing slot
        "Strategy_Apex_Alpha_MachineGun": "Strategy_Apex_Alpha_MachineGun"
    }

    kept_strategies = []
    seen_names = set()

    # Sort input strategies by "score" if available, or assume file is already sorted
    # (Optimizer saves sorted, so top items are best)

    for strat in strategies:
        name = strat.get("name")
        new_name = None

        # Check for matches
        if name in target_map:
            new_name = target_map[name]

        if new_name:
            # DEDUPLICATION LOGIC:
            # We want the NEWEST/BEST version. Since the file is sorted by Score (High -> Low),
            # the first time we see "Strategy_Apex_Alpha_MachineGun", it will be the 55% winner.
            if new_name in seen_names:
                continue

            print(f"✅ KEEPING: {name} -> {new_name}")
            strat["name"] = new_name
            kept_strategies.append(strat)
            seen_names.add(new_name)

    if not kept_strategies:
        print("⚠️  Warning: No target strategies found. Check generated_strategies.json")
        return

    # Backup
    if os.path.exists(CONFIG_PATH):
        os.rename(CONFIG_PATH, CONFIG_PATH + ".bak_upgrade")
        print(f"📂 Backup created: {CONFIG_PATH}.bak_upgrade")

    # Save
    with open(CONFIG_PATH, "w") as f:
        json.dump(kept_strategies, f, indent=4)

    print(f"💾 SUCCESS: Portfolio Updated.")
    print("   1. Strategy_Apex_VIX_Sniper (The Shield)")
    print("   2. Strategy_Apex_Alpha_MachineGun (The Sword - 55% CAGR)")


if __name__ == "__main__":
    clean_and_rename()
