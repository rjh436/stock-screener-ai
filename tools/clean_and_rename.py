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

    # Define the Portfolio Targets
    # We map multiple potential names to the final destination to catch the exact variant present
    target_map = {
        # The Sniper (Already renamed or finding the original mutant)
        "Strategy_Apex_VIX_Sniper": "Strategy_Apex_VIX_Sniper",
        "Gen7_Fresh_9928_mut_mut": "Strategy_Apex_VIX_Sniper",

        # The Machine Gun (The Gen2 / Cross_Gen2 lineage)
        "Cross_Gen2_Gen2": "Strategy_Apex_Alpha_MachineGun",
        "Gen2_Fresh_3401_mut_mut_mut_mut": "Strategy_Apex_Alpha_MachineGun",
        "Gen2_Fresh_3401_mut_mut_mut": "Strategy_Apex_Alpha_MachineGun"
    }

    kept_strategies = []
    seen_names = set()

    for strat in strategies:
        name = strat.get("name")

        # Check for exact match or substring match
        new_name = None
        for key, target in target_map.items():
            if key == name:  # Exact match preferred
                new_name = target
                break

        if not new_name:
            for key, target in target_map.items():
                if key in name and "Fresh" in key:  # Substring match for lengthy mutation names
                    new_name = target
                    break

        if new_name:
            # Deduplication: Only keep the FIRST instance (highest score) of each type
            if new_name in seen_names:
                continue

            print(f"✅ Keeping & Renaming: {name} -> {new_name}")
            strat["name"] = new_name
            kept_strategies.append(strat)
            seen_names.add(new_name)

    if not kept_strategies:
        print("⚠️  Warning: No target strategies found. Check the names in generated_strategies.json")
        # List available names to help debug
        print("Available strategies:", [s["name"] for s in strategies[:5]])
        return

    # Backup
    os.rename(CONFIG_PATH, CONFIG_PATH + ".bak_portfolio_final")
    print(f"📂 Backed up to {CONFIG_PATH}.bak_portfolio_final")

    # Save Portfolio
    with open(CONFIG_PATH, "w") as f:
        json.dump(kept_strategies, f, indent=4)

    print(f"💾 Portfolio Saved: {len(kept_strategies)} Strategies Locked.")
    print("   1. Strategy_Apex_VIX_Sniper (The Shield)")
    print("   2. Strategy_Apex_Alpha_MachineGun (The Sword)")


if __name__ == "__main__":
    clean_and_rename()
