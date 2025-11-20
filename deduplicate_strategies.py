import json
import os

path = "config/generated_strategies.json"
with open(path, "r") as f:
    strategies = json.load(f)

unique_strategies = []
seen_rules = set()

print(f"Total strategies before deduplication: {len(strategies)}")

for strat in strategies:
    # Create a hashable representation of the rules
    # We sort the rules to ensure order doesn't matter for uniqueness
    entry_rules_str = json.dumps(sorted(strat.get("entry_rules", []), key=lambda x: x.get("col", "")), sort_keys=True)
    exit_rules_str = json.dumps(sorted(strat.get("exit_rules", []), key=lambda x: x.get("col", "")), sort_keys=True)
    
    # Combine entry/exit/stop parameters into a unique signature
    signature = (
        entry_rules_str,
        exit_rules_str,
        strat.get("stop_loss_atr"),
        strat.get("time_stop")
    )
    
    if signature not in seen_rules:
        seen_rules.add(signature)
        unique_strategies.append(strat)

print(f"Unique strategies found: {len(unique_strategies)}")

# Rename the top 5 unique strategies
names = ["Strategy_Apex_Gen9_Alpha", "Strategy_Apex_Gen9_Beta", "Strategy_Apex_Gen9_Gamma", "Strategy_Apex_Gen9_Delta", "Strategy_Apex_Gen9_Epsilon"]

# Keep the rest as is, or rename them back to generic if needed. 
# Actually, just renaming the top 5 is enough for the UI.
for i in range(min(len(unique_strategies), 5)):
    unique_strategies[i]["name"] = names[i]

# Save back to file
# We overwrite the file with ONLY the unique strategies to keep it clean?
# Or just put the unique ones at the top?
# Let's keep all unique ones.
with open(path, "w") as f:
    json.dump(unique_strategies, f, indent=4)

print("Saved deduplicated strategies.")
