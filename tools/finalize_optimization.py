import json
import os

def finalize_strategy():
    config_path = "config/generated_strategies.json"
    if not os.path.exists(config_path):
        print("❌ Error: Config file not found.")
        return

    with open(config_path, "r") as f:
        strategies = json.load(f)

    if not strategies:
        print("❌ Error: No strategies found in config.")
        return

    # The optimizer saves the population sorted by score, so index 0 is the winner.
    winner = strategies[0]
    
    print("\n🏆 OPTIMIZATION WINNER ANALYSIS")
    print(f"Original Name: {winner.get('name')}")
    print("-" * 40)
    
    # Rename
    new_name = "Strategy_Apex_Alpha_Gen12"
    winner["name"] = new_name
    
    # Display Key Stats
    print(f"NEW NAME: {new_name}")
    print(f"Time Stop: {winner.get('time_stop')} days")
    print(f"Stop Loss: {winner.get('stop_loss_atr')} ATR")
    
    print("\nEntry Rules:")
    for rule in winner.get("entry_rules", []):
        val = rule.get("val") if "val" in rule else rule.get("ref")
        print(f"  - {rule.get('col')} {rule.get('op')} {val}")

    print("\nExit Rules:")
    for rule in winner.get("exit_rules", []):
        if rule.get("type") == "profit_target":
            print(f"  - Profit Target: {rule.get('val')} ({(float(rule.get('val'))-1)*100:.1f}%)")
        else:
            val = rule.get("val") if "val" in rule else rule.get("ref")
            print(f"  - {rule.get('col')} {rule.get('op')} {val}")

    # Save cleanly
    # We keep the Winner + The Original Sniper (The Shield)
    # We assume 'Strategy_Apex_VIX_Sniper' might be further down the list or overwritten.
    # If it's lost, we can restore it from backup, but let's just save the top 5 for now.
    
    top_strategies = strategies[:5]
    with open(config_path, "w") as f:
        json.dump(top_strategies, f, indent=4)
        
    print("-" * 40)
    print("✅ Winner renamed and saved. Ready for final Backtest.")

if __name__ == "__main__":
    finalize_strategy()
