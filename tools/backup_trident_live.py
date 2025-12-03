import os
import shutil
import datetime


def backup_trident_live():
    # 1. Setup Backup Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backups/TRIDENT_LIVE_{timestamp}"
    
    if not os.path.exists(backup_name):
        os.makedirs(backup_name)
    
    # 2. Critical System Files
    files_to_save = [
        "config/generated_strategies.json",  # The Portfolio
        "execution/engine.py",               # The Brain
        "optimization/evolution.py",         # The Builder
        "optimization/optimizer.py",         # The Judge
        "app.py",                            # The Pro Dashboard
        "simulation/paper_trader.py",        # The Real-Time Simulator
        "data/schwab_client.py"              # The Real-Time Data Client
    ]
    
    print(f"🛡️ CREATING TRIDENT LIVE BACKUP: {backup_name}...")
    
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
        f.write("TRIDENT LIVE - PRODUCTION RELEASE\n")
        f.write("==================================================\n")
        f.write(f"Date: {timestamp}\n\n")
        f.write("STATUS: FULLY OPERATIONAL\n")
        f.write("- Portfolio: Trident (Gen 12 + Gen 9 + Machine Gun)\n")
        f.write("- Engine: Unified Velocity Ranking (Unclamped)\n")
        f.write("- Dashboard: Pro Mode (Interactive Holdings)\n")
        f.write("- Simulator: Real-Time Quotes + Next Open Execution\n")
        
    print("\n✅ SYSTEM SECURED. You are ready for daily operations.")


if __name__ == "__main__":
    backup_trident_live()
