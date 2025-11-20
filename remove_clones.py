import json
import os

path = "config/generated_strategies.json"
with open(path, "r") as f:
    strategies = json.load(f)

# Filter out Beta and Epsilon
new_strategies = [s for s in strategies if s["name"] not in ["Strategy_Apex_Gen9_Beta", "Strategy_Apex_Gen9_Epsilon"]]

print(f"Removed Beta and Epsilon. Remaining: {len(new_strategies)}")

with open(path, "w") as f:
    json.dump(new_strategies, f, indent=4)
