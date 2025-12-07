import os
import shutil

def restore_golden_state():
    print("🏆 Initiating GOLDEN STATE RESTORE...")
    
    backup_root = "backups"
    if not os.path.exists(backup_root):
        print(f"❌ Error: {backup_root} directory not found.")
        return

    # Find folders starting with GOLDEN_STATE_
    # We sort reverse to get the NEWEST one first
    golden_backups = sorted(
        [d for d in os.listdir(backup_root) if d.startswith("GOLDEN_STATE_")],
        reverse=True
    )
    
    if not golden_backups:
        print("❌ CRITICAL: No 'GOLDEN_STATE' backup found!")
        print("   Checking for 'Precision_Build' as fallback...")
        # Fallback to Precision Build if Golden State is missing (similar state)
        golden_backups = sorted(
            [d for d in os.listdir(backup_root) if d.startswith("Precision_Build_")],
            reverse=True
        )
        
    if not golden_backups:
        print("❌ No valid backups found. Cannot restore.")
        return

    target_backup = golden_backups[0]
    source_path = os.path.join(backup_root, target_backup)
    print(f"   ✅ Found Backup: {target_backup}")
    
    # Files to restore
    # We map the filename in the backup to its destination
    # Assuming backup structure is flat or matches source
    files_to_restore = {
        "generated_strategies.json": "config/generated_strategies.json",
        "engine.py": "execution/engine.py",
        "evolution.py": "optimization/evolution.py",
        "optimizer.py": "optimization/optimizer.py",
        "app.py": "app.py"
    }
    
    print("   ♻️  Restoring Files...")
    for filename, dest_path in files_to_restore.items():
        src = os.path.join(source_path, filename)
        
        # Check if file exists in backup
        if os.path.exists(src):
            # Create destination directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(src, dest_path)
            print(f"      - Restored {filename} -> {dest_path}")
        else:
            print(f"      ⚠️  File missing in backup: {filename}")
            
    print("\n✅ RESTORE COMPLETE.")
    print("   The system has been reverted to the 'Golden State'.")
    print("   Please run the Backtest immediately to confirm performance.")

if __name__ == "__main__":
    restore_golden_state()
