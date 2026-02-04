import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data


def contraction_ok(row: pd.Series) -> bool:
    rp5 = float(row.get("range_pct_5", 999.0) or 999.0)
    rp10 = float(row.get("range_pct_10", 999.0) or 999.0)
    rp20 = float(row.get("range_pct_20", 999.0) or 999.0)
    rp40 = float(row.get("range_pct_40", 999.0) or 999.0)

    last_contraction = min(rp5, rp10)
    if last_contraction > 10.0:
        return False

    contractions = 0
    if rp10 <= 20.0:
        contractions += 1
    if rp20 <= 25.0:
        contractions += 1
    if rp40 <= 30.0:
        contractions += 1
    return contractions >= 2


def main() -> None:
    symbols = ["AAPL", "NVDA", "TSLA", "MSFT", "AMD", "AMZN", "META", "GOOGL", "NFLX", "AVGO"]
    data = fetch_data_pack(symbols, days=900, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=900, backtest_mode=True)
    prepared = prepare_backtest_data(data, symbols, start_date=None, global_data=g_data)

    for sym in symbols:
        if sym not in prepared.enriched:
            print(f"{sym}: missing")
            continue
        df = prepared.enriched[sym].df.copy()
        df = df.dropna(subset=["range_pct_5", "range_pct_10", "range_pct_20", "range_pct_40"])
        if df.empty:
            print(f"{sym}: no VCP rows")
            continue

        vcp_flag = df.apply(contraction_ok, axis=1)
        vol_prev = df["volume"].shift(1)
        vol_ma50_prev = df["vol_ma50"].shift(1)
        vol_dry = (vol_prev <= (vol_ma50_prev * 0.75))
        base_depth = df.get("range_pct_20", 999.0) <= 30.0
        tightness = (df.get("vcp_tightness", 999.0) > 0) & (df.get("vcp_tightness", 999.0) < 1.0)

        total = len(df)
        print(f"\n{sym}")
        print(f"Rows: {total}")
        print(f"VCP contraction ok: {vcp_flag.sum()} ({(vcp_flag.mean()*100):.2f}%)")
        print(f"Vol dry-up (<75% prev MA50): {vol_dry.sum()} ({(vol_dry.mean()*100):.2f}%)")
        print(f"Base depth <=30%: {base_depth.sum()} ({(base_depth.mean()*100):.2f}%)")
        print(f"Tightness (0-1): {tightness.sum()} ({(tightness.mean()*100):.2f}%)")

        both = vcp_flag & vol_dry & base_depth
        print(f"VCP+VolDry+BaseDepth: {both.sum()} ({(both.mean()*100):.2f}%)")


if __name__ == "__main__":
    main()
