import json
import os

CONFIG_PATH = "config/generated_strategies.json"


def restore_portfolio():
    # 1. Define the Lost "Sniper" Strategy (Recovered from Backtest Logs)
    sniper_strategy = {
        "name": "Strategy_Apex_VIX_Sniper",
        "type": "random",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},
            {"col": "vix", "op": ">", "val": 33},
            {"col": "rsi14", "op": "<", "ref": "stoch_k"},
            {"col": "volume", "op": ">", "ref": "vol_ma20"}
        ],
        "exit_rules": [],
        "stop_loss_atr": 5.3,
        "time_stop": 90
    }

    # 2. Load the Current "Machine Gun" Strategy
    if not os.path.exists(CONFIG_PATH):
        print("❌ Error: generated_strategies.json not found.")
        return

    with open(CONFIG_PATH, "r") as f:
        current_strategies = json.load(f)

    # Find the Machine Gun (It might be named differently or already renamed)
    machine_gun = None

    # Check for already renamed version
    for s in current_strategies:
        if s.get("name") == "Strategy_Apex_Alpha_MachineGun":
            machine_gun = s
            break

    # If not found, look for the raw mutant (Gen2 lineage)
    if not machine_gun:
        for s in current_strategies:
            if "Gen2" in s.get("name", "") or "Cross_Gen2" in s.get("name", ""):
                machine_gun = s
                # Rename it instantly
                machine_gun["name"] = "Strategy_Apex_Alpha_MachineGun"
                break

    if not machine_gun:
        print("⚠️  Warning: Could not auto-detect the Machine Gun strategy.")
        print("Saving ONLY the Sniper. You may need to re-run optimization if this is wrong.")
        final_portfolio = [sniper_strategy]
    else:
        print(f"✅ Found Machine Gun: {machine_gun['name']}")
        final_portfolio = [sniper_strategy, machine_gun]

    # 3. Save the Master Portfolio
    # Backup first
    if os.path.exists(CONFIG_PATH):
        os.rename(CONFIG_PATH, CONFIG_PATH + ".bak_before_restore")

    with open(CONFIG_PATH, "w") as f:
        json.dump(final_portfolio, f, indent=4)

    print("\n🏆 Portfolio Restored Successfully!")
    print("------------------------------------------------")
    print(f"1. {sniper_strategy['name']} (The Shield)")
    if machine_gun:
        print(f"2. {machine_gun['name']} (The Sword)")
    print("------------------------------------------------")


if __name__ == "__main__":
    restore_portfolio()
