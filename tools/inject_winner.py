import json
import os

GEN_CONFIG = "config/generated_strategies.json"


def inject():
    print("🏆 Strategy Injector Initialized...")
    
    if not os.path.exists(GEN_CONFIG):
        print("❌ No generated strategies found.")
        return

    with open(GEN_CONFIG, "r") as f:
        population = json.load(f)

    if not population:
        print("❌ Strategy list is empty.")
        return

    # 1. Identify the Winner (First in list = Highest Score)
    champion = population[0]
    original_name = champion.get("name", "Unknown")
    
    print(f"   Top Strategy Found: {original_name}")
    
    # 2. Clone and Rename
    # We create a "Prime" version that is easy to spot in the Backtester
    prime_strat = champion.copy()
    prime_strat["name"] = "Apex_Sniper_PRIME_v1"
    
    # 3. Insert at Top (Preserving others)
    # We remove any old "Prime" versions to avoid clutter
    new_pop = [s for s in population if "Apex_Sniper_PRIME" not in s["name"]]
    new_pop.insert(0, prime_strat)
    
    # 4. Save
    with open(GEN_CONFIG, "w") as f:
        json.dump(new_pop, f, indent=4)
        
    print(f"✅ Injected 'Apex_Sniper_PRIME_v1' (derived from {original_name})")
    print("🚀 Go to 'Backtest' in the App and select 'Apex_Sniper_PRIME_v1'.")


if __name__ == "__main__":
    inject()
