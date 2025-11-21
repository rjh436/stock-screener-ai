import json
import shutil
import os
import sys

# Determine paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "generated_strategies.json")
BACKUP_PATH = os.path.join(BASE_DIR, "config", "generated_strategies.json.bak")

def purge():
    # 1. Backup
    if os.path.exists(CONFIG_PATH):
        try:
            shutil.copy(CONFIG_PATH, BACKUP_PATH)
            print(f"Backed up {CONFIG_PATH} to {BACKUP_PATH}")
        except Exception as e:
            print(f"Warning: Failed to backup: {e}")
    else:
        print(f"No existing file at {CONFIG_PATH}, skipping backup.")

    # 2. Overwrite with empty list
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump([], f, indent=4)
        print("Purged old strategies to prevent pollution from flawed evolutionary logic.")
    except Exception as e:
        print(f"Error purging file: {e}")

if __name__ == "__main__":
    purge()
