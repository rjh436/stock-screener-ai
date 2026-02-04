import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from execution.engine import prepare_backtest_data


def _summarize(series: pd.Series, name: str) -> None:
    if series is None:
        print(f"{name}: MISSING")
        return
    s = pd.to_numeric(series, errors="coerce")
    print(
        f"{name}: count={s.count()} nan={s.isna().sum()} "
        f"min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f}"
    )


def main() -> None:
    symbols = get_universe_symbols("RUSSELL3000")
    target = "NVDA"
    print(f"Loading raw data for Russell 3000 (target={target})")
    data = fetch_data_pack(symbols, days=900, backtest_mode=True)
    df_raw = data.get(target)
    if df_raw is None or df_raw.empty:
        print("No raw data found.")
        return

    df_raw = df_raw.copy()
    df_raw.columns = df_raw.columns.str.lower()
    print("\nRAW DATA (head)")
    cols = [c for c in ["close", "volume", "rs_rating", "momentum_rank", "rs_ratio"] if c in df_raw.columns]
    print(df_raw[cols].head(5))
    print("\nRAW DATA (tail)")
    print(df_raw[cols].tail(5))
    for c in cols:
        _summarize(df_raw[c], f"raw.{c}")

    print("\nPreparing backtest data (inject RS/momentum)...")
    g_data = fetch_data_pack(["SPY", "VIX"], days=900, backtest_mode=True)
    prepared = prepare_backtest_data(data, symbols, start_date=None, global_data=g_data)
    if not prepared.enriched or target not in prepared.enriched:
        print("Prepared data missing target.")
        return

    df = prepared.enriched[target].df
    print("\nPREPARED DATA (head)")
    cols2 = [c for c in ["close", "rs_rating", "momentum_rank", "rs_ratio"] if c in df.columns]
    print(df[cols2].head(5))
    print("\nPREPARED DATA (tail)")
    print(df[cols2].tail(5))
    for c in cols2:
        _summarize(df[c], f"prep.{c}")

    # Cross-sectional snapshot for last date
    all_dates = prepared.all_dates
    if len(all_dates):
        last_idx = len(all_dates) - 1
        rs_vals = []
        mom_vals = []
        for sym, sdata in prepared.enriched.items():
            if len(sdata.rsrating) == 0:
                continue
            # Map symbol's last available gidx to global index
            gidx = int(sdata.gidx[-1]) if len(sdata.gidx) else -1
            if gidx == last_idx:
                rs_vals.append(float(sdata.rsrating[-1]))
                mom_vals.append(float(sdata.momrank[-1]))
        if rs_vals:
            rs_arr = np.array(rs_vals, dtype=float)
            mom_arr = np.array(mom_vals, dtype=float)
            print("\nCROSS-SECTION (last date) RS distribution")
            print(
                f"rs_rating: count={rs_arr.size} min={rs_arr.min():.2f} "
                f"p50={np.percentile(rs_arr, 50):.2f} p90={np.percentile(rs_arr, 90):.2f} max={rs_arr.max():.2f}"
            )
            print(
                f"momentum_rank: count={mom_arr.size} min={mom_arr.min():.2f} "
                f"p50={np.percentile(mom_arr, 50):.2f} p90={np.percentile(mom_arr, 90):.2f} max={mom_arr.max():.2f}"
            )


if __name__ == "__main__":
    main()
