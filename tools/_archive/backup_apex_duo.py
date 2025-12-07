"""Create a timestamped backup for the Apex Duo portfolio and engine files."""

import datetime
import os
import shutil


FILES_TO_SAVE = [
    "config/generated_strategies.json",  # Apex Duo portfolio
    "execution/engine.py",               # Unified Velocity engine
    "optimization/evolution.py",         # Builder
    "optimization/optimizer.py",         # Judge
    "app.py",                            # Dashboard
]


def backup_apex_duo() -> str:
    """Create backup directory, copy critical files, and write manifest."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backups", f"APEX_DUO_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    print(f"Creating Apex Duo backup at {backup_dir}...")

    for path in FILES_TO_SAVE:
        if os.path.exists(path):
            shutil.copy(path, backup_dir)
            print(f"  Secured: {path}")
        else:
            print(f"  Warning: {path} not found")

    readme_path = os.path.join(backup_dir, "README.txt")
    with open(readme_path, "w") as readme:
        readme.write("APEX DUO PORTFOLIO BACKUP\n")
        readme.write("============================================\n")
        readme.write(f"Date: {timestamp}\n\n")
        readme.write(
            "This backup contains the Apex Duo portfolio "
            "(Gen 12 Classic + Gen 9 Evolved) and the Unified Velocity engine.\n\n"
        )
        readme.write("Files captured:\n")
        for path in FILES_TO_SAVE:
            readme.write(f"- {path}\n")

    print("Backup complete.")
    return backup_dir


if __name__ == "__main__":
    backup_apex_duo()
