import os
import shutil
import datetime


def create_backup():
    # 1. Define what defines the "Current State"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/baseline_{timestamp}"
    
    # Files that constitute the "Brain" of the system
    critical_files = [
        "config/generated_strategies.json",  # The winning strategies
        "optimization/evolution.py",         # The current breeding logic
        "execution/engine.py",               # The validated math engine
        "optimization/optimizer.py"          # The optimization loop
    ]
    
    # 2. Create Backup Directory
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"📂 Created backup directory: {backup_dir}")
    
    # 3. Copy Files
    for file_path in critical_files:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            dest_path = os.path.join(backup_dir, file_name)
            shutil.copy2(file_path, dest_path)
            print(f"   ✅ Backed up: {file_name}")
        else:
            print(f"   ⚠️  Warning: Could not find {file_path}")

    # 4. Create a 'Restore' instruction file
    with open(f"{backup_dir}/RESTORE_INSTRUCTIONS.txt", "w") as f:
        f.write(f"Baseline Backup created on {timestamp}.\n")
        f.write("To restore this version:\n")
        f.write("1. Copy 'generated_strategies.json' back to 'config/'\n")
        f.write("2. Copy 'evolution.py' back to 'optimization/'\n")
        f.write("3. Copy 'engine.py' back to 'execution/'\n")
        
    print("\n🚀 Restore Point Saved. You may now proceed with changes safely.")


if __name__ == "__main__":
    create_backup()
