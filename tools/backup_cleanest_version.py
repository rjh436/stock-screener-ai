import os
import shutil
import datetime


def backup_cleanest_version():
    # 1. Setup Backup Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backups/CLEANEST_VERSION_{timestamp}"
    
    if not os.path.exists(backup_name):
        os.makedirs(backup_name)
    
    # 2. Critical System Files
    files_to_save = [
        "config/generated_strategies.json",  # The Portfolio
        "execution/engine.py",               # The Brain
        "optimization/evolution.py",         # The Builder
        "optimization/optimizer.py",         # The Judge
        "app.py",                            # The Pro Dashboard (4 Modes)
        "simulation/paper_trader.py",        # The Real-Time Simulator
        "data/schwab_client.py",             # The Data Client
        "data/loader.py"                     # The Data Loader
    ]
    
    print(f"💎 CREATING CLEANEST VERSION BACKUP: {backup_name}...")
    
    for f in files_to_save:
        if os.path.exists(f):
            # Handle subdirectories
            dest = os.path.join(backup_name, os.path.basename(f))
            shutil.copy(f, dest)
            print(f"   ✅ Secured: {f}")
        else:
            print(f"   ⚠️ Warning: {f} not found!")
            
    # 3. Create Manifest
    manifest_path = f"{backup_name}/MANIFEST.txt"
    with open(manifest_path, "w") as f:
        f.write("APEX SNIPER - CLEANEST VERSION\n")
        f.write("==================================================\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write("SYSTEM STATE:\n")
        f.write("- Portfolio: Apex Duo (Gen 12 Classic + Gen 9 Evolved)\n")
        f.write("- Engine: Unified Velocity Ranking (Unclamped, Trend Bonus)\n")
        f.write("- Dashboard: Full Suite (Live Screener, Backtest, Super Signal Lab, Pro Simulator)\n")
        f.write("- Simulator: Real-Time Quotes + Next Open Execution + Super Signal Priority\n")
        f.write("- UI: Persistent Results + Download Buttons + Stop Loss Columns\n")
        
    print("\n✅ BACKUP COMPLETE. You have a permanent restore point for this version.")


if __name__ == "__main__":
    backup_cleanest_version()
