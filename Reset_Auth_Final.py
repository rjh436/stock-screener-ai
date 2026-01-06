import os
import shutil
from pathlib import Path

print("INITIATING NUCLEAR AUTH RESET...")

# 1. Define all possible hiding spots for the Ghost Token
possible_paths = [
    # Project paths
    Path("data/schwab_tokens.json"),
    Path("data/.schwab_creds.json"),
    Path(".schwab_token.json"),
    Path("tokens.json"),
    # SYSTEM DEFAULT PATH (The likely culprit!)
    Path.home() / ".schwab" / "token.json",
    Path.home() / ".schwab",
]

found = False
for p in possible_paths:
    if p.exists():
        found = True
        try:
            if p.is_dir():
                shutil.rmtree(p)
                print(f"DELETED DIRECTORY: {p}")
            else:
                p.unlink()
                print(f"DELETED FILE: {p}")
        except Exception as e:
            print(f"FAILED to delete {p}: {e}")
    else:
        print(f"Checked {p} (Nothing found)")

if not found:
    print("No tokens found. Check for hidden files in root.")

# 2. Clear the Data Cache (Preserving the folder structure)
# This removes empty files that might be causing the "Incomplete History" error
cache_dir = Path("data/cache")
if cache_dir.exists():
    print("Cleaning Data Cache (removing small/empty files)...")
    count = 0
    for f in cache_dir.glob("*.parquet"):
        try:
            # Delete files smaller than 1KB (likely failed downloads)
            if f.stat().st_size < 1024:
                f.unlink()
                count += 1
        except Exception:
            pass
    print(f"Removed {count} corrupted cache files.")

print("\nRESET COMPLETE.")
print("Run './Run_Screener.command' immediately.")
print("You should see a 'Please log in' prompt in this terminal.")
