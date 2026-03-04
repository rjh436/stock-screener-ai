from __future__ import annotations

from typing import Any, Dict, Optional
import math

import numpy as np
import pandas as pd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def evaluate_recovery(
    df: pd.DataFrame,
    i: int,
    params: Dict[str, Any],
    rs_percentile: float,
) -> Optional[Dict[str, Any]]:
    """
    Recovery/re-entry sleeve:
    - Looser template to catch post-correction reclaims
    - Requires recent weakness then reclaim above sma50
    - Keeps RS quality floor to avoid weak dead-cat bounces
    """
    if i < 10:
        return None

    row = df.iloc[i]

    close_px = _as_float(row.get("close"), 0.0)
    low_px = _as_float(row.get("low"), 0.0)
    sma50 = _as_float(row.get("sma50"), 0.0)
    sma200 = _as_float(row.get("sma200"), 0.0)
    volume = _as_float(row.get("volume"), 0.0)
    vol_ma50 = _as_float(row.get("vol_ma50"), float("nan"))

    if not (close_px > 0 and sma50 > 0 and sma200 > 0):
        return None
    if not (close_px > sma50 > sma200):
        return None

    rs_min = float(params.get("recovery_rs_min", 65.0) or 65.0)
    if rs_percentile < rs_min:
        return None

    reclaim_lookback = max(5, int(params.get("recovery_reclaim_lookback_bars", 15) or 15))
    start = max(0, i - reclaim_lookback)
    recent = df.iloc[start : i + 1]
    closes = pd.to_numeric(recent.get("close"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    sma50s = pd.to_numeric(recent.get("sma50"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    if closes.size < 3 or sma50s.size < 3:
        return None

    prior_weakness = bool(np.any((closes[:-1] < sma50s[:-1]) & np.isfinite(closes[:-1]) & np.isfinite(sma50s[:-1])))
    if not prior_weakness:
        return None

    reclaim_buffer = float(params.get("recovery_reclaim_buffer_pct", 0.005) or 0.005)
    if close_px < (sma50 * (1.0 + max(0.0, reclaim_buffer))):
        return None

    if math.isfinite(vol_ma50) and vol_ma50 > 0:
        vol_min_mult = float(params.get("recovery_volume_min_mult", 1.0) or 1.0)
        if volume < (vol_ma50 * vol_min_mult):
            return None

    trigger_px = close_px
    stop_buffer_pct = float(params.get("recovery_stop_below_sma50_pct", 0.05) or 0.05)
    stop_anchor = sma50 * (1.0 - max(0.0, stop_buffer_pct))
    stop_px = min(low_px if low_px > 0 else stop_anchor, stop_anchor)
    if stop_px <= 0 or stop_px >= trigger_px:
        return None

    max_stop_pct = float(params.get("recovery_max_stop_pct", 0.12) or 0.12)
    stop_width = (trigger_px - stop_px) / trigger_px
    if stop_width > max_stop_pct:
        return None

    reclaim_distance = (close_px - sma50) / sma50 if sma50 > 0 else 0.0
    rs_component = min(1.0, rs_percentile / 100.0)
    reclaim_component = max(0.0, min(1.0, 1.0 - (reclaim_distance / 0.08)))
    strength = (0.60 * rs_component) + (0.40 * reclaim_component)

    return {
        "trigger_price": trigger_px,
        "stop_price": stop_px,
        "stop_limit_pct": float(params.get("stop_limit_pct", 0.02) or 0.02),
        "stop_loss_type": "low_or_pct",
        "entry_type": "recovery",
        "sleeve": "recovery",
        "max_stop_pct": max_stop_pct,
        "entry_timing": "next_day",
        "signal_mode": "after_close",
        "signal_strength": float(strength),
    }
