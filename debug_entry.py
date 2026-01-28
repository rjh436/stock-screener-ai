import os
import sys
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data


def run_debug(symbol="TSLA"):
    start_date = "2020-01-01"
    end_date = "2020-12-31"

    data = fetch_data_pack([symbol], days=3000, backtest_mode=True)
    g_data = fetch_data_pack(["SPY"], days=3000, backtest_mode=True) or {}

    if not data or symbol not in data or data[symbol] is None or data[symbol].empty:
        print(f"No data for {symbol}")
        return

    prepared = prepare_backtest_data(data, [symbol], None, g_data)
    enriched = prepared.enriched
    if symbol not in enriched:
        print(f"Symbol {symbol} not enriched")
        return

    df = enriched[symbol].df.copy()
    df = df.loc[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]

    print("Date,Close,Pivot,Buy")
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        pivot = row.get("high_20_prev", np.nan)
        price_today = row.get("close", np.nan)
        price_yesterday = prev_row.get("close", np.nan)
        vol_today = row.get("volume", np.nan)
        vol_ma50 = row.get("vol_ma50", np.nan)
        rs_rating = row.get("rs_rating", np.nan)

        buy = False
        if (
            np.isfinite(pivot)
            and np.isfinite(price_today)
            and np.isfinite(price_yesterday)
            and np.isfinite(vol_today)
            and np.isfinite(vol_ma50)
            and vol_ma50 > 0
            and np.isfinite(rs_rating)
        ):
            if rs_rating >= 90:
                if vol_today >= (vol_ma50 * 1.2):
                    if price_today > pivot and price_yesterday < pivot:
                        buy = True

        date_str = str(row.name)[:10]
        pivot_str = f"{pivot:.2f}" if np.isfinite(pivot) else "nan"
        close_str = f"{price_today:.2f}" if np.isfinite(price_today) else "nan"
        print(f"{date_str},{close_str},{pivot_str},{'BUY' if buy else ''}")


if __name__ == "__main__":
    run_debug("TSLA")
