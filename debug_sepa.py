from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data

SYMBOLS = ["TSLA", "NVDA", "AAPL"]
TARGET_DATES = ["2020-11-13", "2020-11-16"]
DAYS = 5000


def _fmt(val: float | int | None) -> str:
    try:
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            return "NaN"
        return f"{float(val):.4f}"
    except Exception:
        return "NaN"


def _to_date(val: np.datetime64) -> str:
    try:
        return str(pd.Timestamp(val).date())
    except Exception:
        return str(val)


def _pick_vix(g_data: dict) -> Optional[pd.DataFrame]:
    for key in ("$VIX", "VIX"):
        df = g_data.get(key)
        if df is not None and hasattr(df, "empty") and not df.empty:
            return df
    return None


def _get_index_for_date(idx: np.ndarray, target_date: str) -> Optional[int]:
    target_ns = pd.Timestamp(target_date).to_datetime64()
    hit = np.where(idx == target_ns)[0]
    if hit.size:
        return int(hit[0])
    pos = int(np.searchsorted(idx, target_ns))
    if pos <= 0 or pos >= len(idx):
        return None
    return pos - 1


def _report_for_date(sd, date_str: str) -> None:
    idx = sd.index
    loc = _get_index_for_date(idx, date_str)
    if loc is None:
        print(f"\n[{date_str}] ERROR: Date not found in available data.")
        return

    actual_date = _to_date(idx[loc])

    close = sd.close[loc]
    sma50 = sd.sma50[loc]
    sma150 = sd.sma150[loc]
    sma200 = sd.sma200[loc]
    sma200_slope = sd.sma200slope[loc]
    high52 = sd.high52w[loc]
    low52 = sd.low52w[loc]
    rs_rating = sd.rsrating[loc] if loc < len(sd.rsrating) else np.nan
    bb_width = sd.bbwidth[loc] if loc < len(sd.bbwidth) else np.nan
    trend_mask = bool(sd.trend_mask[loc]) if loc < len(sd.trend_mask) else False

    slope_is_nan = not np.isfinite(sma200_slope)
    slope_is_zero = np.isfinite(sma200_slope) and float(sma200_slope) == 0.0

    print(f"\n=== Forensic Report: {date_str} (Bar: {actual_date}) ===")
    print("-- Trend Metrics --")
    print(f"Close:        {_fmt(close)}")
    print(f"SMA50:        {_fmt(sma50)}")
    print(f"SMA150:       {_fmt(sma150)}")
    print(f"SMA200:       {_fmt(sma200)}")
    print(f"SMA200 Slope: {_fmt(sma200_slope)}")
    print(f"52w High:     {_fmt(high52)}")
    print(f"52w Low:      {_fmt(low52)}")
    print(f"trend_mask:   {trend_mask}")

    print("\n-- Soft Metrics --")
    print(f"RS Rating:    {_fmt(rs_rating)} (Pass > 80: {np.isfinite(rs_rating) and rs_rating > 80})")
    print(f"BB Width:     {_fmt(bb_width)} (Pass < 0.20: {np.isfinite(bb_width) and bb_width < 0.20})")

    print("\n-- Data Integrity --")
    print(f"sma200_slope NaN: {slope_is_nan}")
    print(f"sma200_slope == 0: {slope_is_zero}")

    print("\n-- Manual Logic Simulation (Engine Filters) --")
    reasons = []

    # Hard gates from trend_mask definition in engine.py
    if not (np.isfinite(close) and np.isfinite(sma50) and close > sma50):
        reasons.append("REJECTED: Close <= SMA50")
    if not (np.isfinite(sma50) and np.isfinite(sma150) and sma50 > sma150):
        reasons.append("REJECTED: SMA50 <= SMA150")
    if not (np.isfinite(sma150) and np.isfinite(sma200) and sma150 > sma200):
        reasons.append("REJECTED: SMA150 <= SMA200")
    if not (np.isfinite(sma200_slope) and sma200_slope > 0):
        reasons.append("REJECTED: SMA200 Slope <= 0 or NaN")
    if not (np.isfinite(high52) and np.isfinite(close) and close > 0.75 * high52):
        reasons.append("REJECTED: Close <= 0.75 * 52w High")
    if not (np.isfinite(low52) and np.isfinite(close) and close > 1.30 * low52):
        reasons.append("REJECTED: Close <= 1.30 * 52w Low")

    # Soft gates (engine loop)
    if not (np.isfinite(rs_rating) and rs_rating > 80):
        reasons.append("REJECTED: RS Rating <= 80 or NaN")
    if not (np.isfinite(bb_width) and bb_width < 0.20):
        reasons.append("REJECTED: BB Width >= 0.20 or NaN")

    if reasons:
        for r in reasons:
            print(r)
    else:
        print("PASS: All gates satisfied for this bar.")


def main() -> int:
    print("=== SEPA V18.2 Surgical Biopsy ===")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Days: {DAYS}")

    data_map = fetch_data_pack(SYMBOLS, days=DAYS, backtest_mode=True) or {}
    if not data_map:
        print("ERROR: No data returned for symbols.")
        return 1

    g_data = fetch_data_pack(["SPY", "VIX", "$VIX"], days=DAYS, backtest_mode=True) or {}
    spy_df = g_data.get("SPY")
    if spy_df is None or (hasattr(spy_df, "empty") and spy_df.empty):
        print("ERROR: SPY data missing. RS ratio and regime checks may be invalid.")

    vix_df = _pick_vix(g_data)
    if vix_df is None:
        print("WARNING: VIX data missing. Proceeding with defaults.")

    global_data = {"SPY": spy_df, "VIX": vix_df}

    prepared = prepare_backtest_data(data_map, symbol_universe=SYMBOLS, start_date=None, global_data=global_data)
    sd = prepared.enriched.get("TSLA")
    if sd is None:
        print("ERROR: TSLA not found in prepared data.")
        return 1

    for dt in TARGET_DATES:
        _report_for_date(sd, dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
