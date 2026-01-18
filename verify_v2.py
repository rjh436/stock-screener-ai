import os
import sys

import numpy as np
import pandas as pd

# Setup path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack
from execution.engine import _compute_indicators


def verify():
    print("Verifying V2 engine data integrity...")

    # Load small sample
    symbols = ["AAPL", "TSLA", "NVDA"]
    data = fetch_data_pack(symbols)

    if not data:
        print("Failed to load data.")
        return

    for sym, df in data.items():
        print(f"\nProcessing {sym}...")
        try:
            enriched = _compute_indicators(df)

            # Check 1: ADR
            if "adr_pct" not in enriched.columns:
                print("   ADR column missing")
            else:
                adr_mean = enriched["adr_pct"].mean()
                print(f"   ADR column found (mean: {adr_mean:.2f}%)")

            # Check 2: Prev High
            if "prev_high" not in enriched.columns:
                print("   Prev_high column missing")
            else:
                print("   Prev_high column found")

        except Exception as e:
            print(f"   Crash: {e}")


if __name__ == "__main__":
    verify()
