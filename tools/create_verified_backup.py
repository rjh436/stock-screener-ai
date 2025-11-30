import os
import shutil
import datetime


def create_verified_backup():
    # 1. Setup Verified Directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backups/GOLDEN_STATE_VERIFIED_{timestamp}"
    
    if not os.path.exists(backup_name):
        os.makedirs(backup_name)
    
    # 2. Critical System Files to Freeze
    files_to_save = [
        "config/generated_strategies.json",  # The Portfolio (Gen 12 + Machine Gun)
        "execution/engine.py",               # The Brain (Unified Velocity Ranking)
        "optimization/evolution.py",         # The Trainer
        "optimization/optimizer.py",         # The Loop
        "app.py"                             # The Dashboard
    ]
    
    print(f"🔐 CREATING VERIFIED BACKUP: {backup_name}...")
    
    for f in files_to_save:
        if os.path.exists(f):
            shutil.copy(f, backup_name)
            print(f"   ✅ Secured: {f}")
        else:
            print(f"   ⚠️ Warning: {f} not found!")
            
    # 3. Create Manifest with Performance Evidence
    manifest_path = f"{backup_name}/MANIFEST.txt"
    with open(manifest_path, "w") as f:
        f.write("APEX SNIPER - VERIFIED GOLDEN STATE\n")
        f.write("==================================================\n")
        f.write(f"Date: {timestamp}\n")
        f.write("Verification Source: 2025-11-30T22-40_export.csv\n\n")
        
        f.write("STRATEGY 1: THE SHIELD (Gen 12 Sniper)\n")
        f.write("   - Performance: 48.96% CAGR\n")
        f.write("   - Avg Profit:  18.35%\n")
        f.write("   - Trades:      49\n")
        f.write("   - Logic:       CCI < 0, BB Width > 0.1 (Deep Value)\n\n")
        
        f.write("STRATEGY 2: THE SWORD (Machine Gun)\n")
        f.write("   - Performance: 45.98% CAGR\n")
        f.write("   - Avg Profit:  1.61%\n")
        f.write("   - Trades:      432\n")
        f.write("   - Logic:       ADX < RSI14, Profit Target 1.06 (Velocity)\n")
        
    print("\n✅ BACKUP COMPLETE. This state is now frozen in time.")


if __name__ == "__main__":
    create_verified_backup()
