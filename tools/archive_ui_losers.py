#!/usr/bin/env python3
import json

CONFIG_FILE = "config/generated_strategies.json"
KEEP_LIST = [
    "Superperformance Alpha B4",
    "Superperformance",
    "Superperformance Practical Selective V4 Candidate",
    "Superperformance Practical Selective V3"
]

def main():
    with open(CONFIG_FILE, "r") as f:
        strats = json.load(f)
    
    modified = 0
    for strat in strats:
        # If it's not our top 2-3 surviving candidates, explicitly disable it from UI defaults
        if strat.get("name") not in KEEP_LIST:
            if strat.get("enabled_by_default") is not False:
                strat["enabled_by_default"] = False
                modified += 1
                print(f"Archived from UI: {strat.get('name')}")
        else:
            strat["enabled_by_default"] = True
    
    if modified > 0:
        with open(CONFIG_FILE, "w") as f:
            json.dump(strats, f, indent=2)
        print(f"Successfully archived {modified} strategies.")
    else:
        print("No changes needed. Archival already matches criteria.")

if __name__ == "__main__":
    main()
