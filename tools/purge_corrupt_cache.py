import glob
import os

import pandas as pd

CACHE_DIR = "data/cache_indices"


def purge_corrupt():
    print(f"Scanning {CACHE_DIR} for corrupt parquet files...")

    files = glob.glob(os.path.join(CACHE_DIR, "*.parquet"))
    deleted = 0

    for f in files:
        try:
            if os.path.getsize(f) == 0:
                print(f"Removing 0-byte file: {f}")
                os.remove(f)
                deleted += 1
                continue

            try:
                pd.read_parquet(f)
            except Exception as e:
                print(f"Removing unreadable file ({os.path.basename(f)}): {e}")
                os.remove(f)
                deleted += 1

        except Exception as e:
            print(f"Could not access {f}: {e}")

    print(f"\nScan complete. Removed {deleted} corrupt files.")
    print("Restart the screener to re-download fresh data.")


if __name__ == "__main__":
    purge_corrupt()
