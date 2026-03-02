import json

with open("config/generated_strategies.json", "r") as f:
    strategies = json.load(f)

for strat in strategies:
    if "max_total_exposure_pct_bull" in strat and strat["max_total_exposure_pct_bull"] > 1.0:
        strat["max_total_exposure_pct_bull"] = 1.0
    if "max_total_exposure_pct_bear" in strat and strat["max_total_exposure_pct_bear"] > 1.0:
        strat["max_total_exposure_pct_bear"] = 1.0
    if "max_pos_size_pct" in strat and strat["max_pos_size_pct"] > 1.0:
        strat["max_pos_size_pct"] = 1.0

with open("config/generated_strategies.json", "w") as f:
    json.dump(strategies, f, indent=2)

print("Fixed max_total_exposure_pct limits in generated_strategies.json")
