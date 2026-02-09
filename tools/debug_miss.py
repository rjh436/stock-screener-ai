#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data
from strategies.minervini_sepa import MinerviniSEPAStrategy, _contraction_ok


SYMBOL = "TSLA"
START_DATE = "2020-01-01"
END_DATE = "2020-12-31"
START_TS = pd.Timestamp(START_DATE)
END_TS = pd.Timestamp(END_DATE)


def _f(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _tf(flag: bool) -> str:
    return "T" if bool(flag) else "F"


def evaluate_minervini_row(df: pd.DataFrame, i: int, params: Dict[str, Any]) -> Dict[str, Any]:
    row = df.iloc[i]
    date = pd.Timestamp(df.index[i])

    close_px = _f(row.get("close"), 0.0)
    sma50 = _f(row.get("sma50"), np.nan)
    sma150 = _f(row.get("sma150"), np.nan)
    sma200 = _f(row.get("sma200"), np.nan)
    high_52w = _f(row.get("high_52w"), np.nan)
    rs_rank = _f(row.get("rs_rating"), np.nan)

    warmup = int(params.get("warmup_bars", 200) or 200)
    rs_min = float(params.get("rs_min", 80.0) or 80.0)
    stop_buy_ref = str(params.get("stop_buy_ref", "high_20_prev") or "high_20_prev")
    vol_mult = float(params.get("vol_mult", 1.5) or 1.5)
    max_stop_pct = float(params.get("max_stop_pct", 0.05) or 0.05)

    trend_pass = False
    vcp_pass = False
    reason = "UNKNOWN"

    if i < warmup:
        reason = f"WARMUP<{warmup}"
    elif close_px <= 0:
        reason = "Invalid Close"
    elif not (np.isfinite(sma50) and np.isfinite(sma150) and np.isfinite(sma200)):
        reason = "Missing Trend MAs"
    elif not (close_px > sma50 > sma150 > sma200):
        reason = "Trend Template Failed"
    elif not np.isfinite(high_52w) or high_52w <= 0:
        reason = "Missing 52W High"
    elif close_px < (0.75 * high_52w):
        reason = "Below 75% of 52W High"
    elif not np.isfinite(rs_rank) or rs_rank < rs_min:
        reason = "Primary RS Gate"
    else:
        trend_pass = True
        if not _contraction_ok(df, i, params):
            reason = "VCP Tightness"
        else:
            vcp_pass = True
            pivot = _f(row.get(stop_buy_ref), _f(row.get("high_20_prev"), 0.0))
            if pivot <= 0:
                reason = "Missing Pivot"
            elif close_px <= pivot:
                reason = "Breakout Close <= Pivot"
            else:
                vol = _f(row.get("volume"), 0.0)
                vol_ma50 = _f(row.get("vol_ma50"), np.nan)
                if np.isfinite(vol_ma50) and vol_ma50 > 0 and vol < (vol_ma50 * vol_mult):
                    reason = "Volume Expansion Failed"
                else:
                    low_px = _f(row.get("low"), 0.0)
                    trigger = pivot * float(params.get("stop_buy_mult", 1.0) or 1.0)
                    if trigger <= 0 or low_px <= 0:
                        reason = "Invalid Trigger/Low"
                    else:
                        stop_width = (trigger - low_px) / trigger
                        if stop_width > max_stop_pct:
                            reason = "Stop Width Failed"
                        else:
                            reason = "ENTRY_READY"

    return {
        "date": date,
        "price": close_px,
        "ma200": sma200,
        "rs_rank": rs_rank,
        "trend_pass": trend_pass,
        "vcp_pass": vcp_pass,
        "reason": reason,
    }


def main() -> None:
    params = {
        "name": "Minervini SEPA Debug",
        "warmup_bars": 200,
        "rs_min": 80.0,
        "vol_mult": 1.5,
        "stop_buy_ref": "high_20_prev",
        "stop_buy_mult": 1.0,
        "max_stop_pct": 0.05,
        "stop_limit_pct": 0.02,
        "vcp_lookback_bars": 80,
        "vcp_pivot_span": 1,
        "vcp_required_contractions": 2,
        "vcp_max_base_depth_pct": 35.0,
        "vcp_last_contraction_max_pct": 10.0,
        "vcp_damping_ratio": 0.85,
        "vcp_require_pivot_volume_dryup": True,
    }

    print("Flight Recorder: TSLA 2020 Minervini SEPA Gate Trace")
    today = pd.Timestamp.now().normalize()
    days = int(max((today - START_TS).days, (END_TS - START_TS).days) + 420)

    data = fetch_data_pack([SYMBOL], days=days, backtest_mode=False, force_fresh=True) or {}
    global_raw = fetch_data_pack(["SPY"], days=days, backtest_mode=False, force_fresh=True) or {}
    spy_df = global_raw.get("SPY")
    global_data = {"SPY": spy_df, "VIX": None}

    prev_disable_cache = os.environ.get("APEX_DISABLE_INDICATOR_CACHE")
    os.environ["APEX_DISABLE_INDICATOR_CACHE"] = "1"
    try:
        prepared = prepare_backtest_data(
            data,
            symbol_universe=[SYMBOL],
            start_date=None,
            global_data=global_data,
        )
    finally:
        if prev_disable_cache is None:
            os.environ.pop("APEX_DISABLE_INDICATOR_CACHE", None)
        else:
            os.environ["APEX_DISABLE_INDICATOR_CACHE"] = prev_disable_cache
    sym_data = prepared.enriched.get(SYMBOL)
    if sym_data is None or sym_data.df.empty:
        raise RuntimeError("TSLA dataframe unavailable after preparation.")

    df = sym_data.df
    strat = MinerviniSEPAStrategy(params)
    total_days = 0
    entry_ready_days = 0
    strategy_signal_days = 0
    reasons: Dict[str, int] = {}
    signal_dates: List[str] = []

    for i, ts in enumerate(df.index):
        dt = pd.Timestamp(ts)
        if dt < START_TS or dt > END_TS:
            continue
        total_days += 1
        diag = evaluate_minervini_row(df, i, params)
        reasons[diag["reason"]] = reasons.get(diag["reason"], 0) + 1
        if diag["reason"] == "ENTRY_READY":
            entry_ready_days += 1

        decision = strat.entry(df, i)
        if decision:
            strategy_signal_days += 1
            signal_dates.append(str(dt.date()))

        ma200 = diag["ma200"]
        rs_rank = diag["rs_rank"]
        ma200_s = f"${ma200:.2f}" if np.isfinite(ma200) else "NaN"
        rs_s = f"{rs_rank:.2f}" if np.isfinite(rs_rank) else "NaN"
        print(
            f"[{dt.date()}] Price: ${diag['price']:.2f} | MA200: {ma200_s} | "
            f"RS_Rank: {rs_s} | Trend_Pass: {_tf(diag['trend_pass'])} | "
            f"VCP_Pass: {_tf(diag['vcp_pass'])} | REASON: {diag['reason']}"
        )

    top_reason = max(reasons.items(), key=lambda x: x[1])[0] if reasons else "N/A"
    print("\nSummary")
    print(f"Total days checked: {total_days}")
    print(f"ENTRY_READY days: {entry_ready_days}")
    print(f"Strategy entry signals: {strategy_signal_days}")
    print(f"Top rejection reason: {top_reason}")
    if signal_dates:
        print(f"Signal dates: {', '.join(signal_dates[:10])}")


if __name__ == "__main__":
    main()
