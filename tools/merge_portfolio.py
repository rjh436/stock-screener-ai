import json
import os

def merge_portfolio():
    print("⚔️ Constructing Final Barbell Portfolio (Smart Scan)...")

    # 1. Load Current Winner (Gen 12) from the active config
    current_config = "config/generated_strategies.json"
    if not os.path.exists(current_config):
        print("❌ Error: Active configuration file not found.")
        return

    with open(current_config, "r") as f:
        current_strats = json.load(f)
    
    # Identify Gen 12 (The new winner created by the Optimizer)
    gen12 = None
    # Usually the first one, or we search by name characteristics if renamed
    if current_strats:
        gen12 = current_strats[0]
        # Rename it to be clear this is the Long Term component
        gen12["name"] = "Strategy_Apex_Gen12_Sniper" 
    
    if not gen12:
        print("❌ Could not find a valid strategy in current config.")
        return

    print(f"   ✅ Loaded Current Winner: {gen12['name']}")

    # 2. Find the "Machine Gun" Strategy in Backups
    # We scan ALL folders in 'backups/', sorted by newest first
    backup_root = "backups"
    if not os.path.exists(backup_root):
        print("❌ Error: 'backups' directory not found.")
        return

    # Get all subdirectories in backups/
    all_backups = sorted(
        [d for d in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, d))],
        reverse=True
    )
    
    machine_gun = None
    found_in = None

    print(f"   🔍 Scanning {len(all_backups)} backup folders for 'Strategy_Apex_Alpha_MachineGun'...")

    for backup_dir in all_backups:
        strat_file = os.path.join(backup_root, backup_dir, "generated_strategies.json")
        
        if os.path.exists(strat_file):
            try:
                with open(strat_file, "r") as f:
                    strategies = json.load(f)
                
                # Search for the specific strategy name
                for s in strategies:
                    if s.get("name") == "Strategy_Apex_Alpha_MachineGun":
                        machine_gun = s
                        found_in = backup_dir
                        break
            except Exception:
                continue
        
        if machine_gun:
            break

    if not machine_gun:
        print("❌ CRITICAL: Could not find 'Strategy_Apex_Alpha_MachineGun' in any backup.")
        print("   Please check if you have a backup containing the Precision Build.")
        return

    print(f"   ✅ Recovered 'The Sword' from: {found_in}")

    # 3. Merge & Save
    # The portfolio is: [Long Term Winner (Gen 12), Daily Active (Machine Gun)]
    final_portfolio = [gen12, machine_gun]
    
    with open(current_config, "w") as f:
        json.dump(final_portfolio, f, indent=4)
        
    print("\n🏆 Portfolio Merged Successfully!")
    print(f"   1. {gen12['name']} (The Shield/Wealth - 49% CAGR)")
    print(f"   2. {machine_gun['name']} (The Sword/Cash - 43% CAGR)")
    print("   🚀 System Ready for Deployment.")

if __name__ == "__main__":
    merge_portfolio()
