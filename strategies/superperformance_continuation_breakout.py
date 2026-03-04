from __future__ import annotations

from typing import Any, Dict, Optional
import math

import pandas as pd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def evaluate_continuation_breakout(
    df: pd.DataFrame,
    i: int,
    params: Dict[str, Any],
    rs_percentile: float,
) -> Optional[Dict[str, Any]]:
    """
    Trend-continuation breakout sleeve:
    - Confirmed uptrend
    - Strong relative strength
    - Fresh breakout above short consolidation pivot
    - Volume confirmation (configurable)
    """
    lookback = max(
        5,
        int(params.get("continuation_breakout_lookback_bars", 20) or 20),
    )
    if i <= lookback:
        return None

    row = df.iloc[i]
    prev_row = df.iloc[i - 1]

    close_px = _as_float(row.get("close"), 0.0)
    open_px = _as_float(row.get("open"), 0.0)
    low_px = _as_float(row.get("low"), 0.0)
    high_px = _as_float(row.get("high"), 0.0)
    sma20 = _as_float(row.get("sma20"), 0.0)
    sma50 = _as_float(row.get("sma50"), 0.0)
    sma150 = _as_float(row.get("sma150"), 0.0)
    sma200 = _as_float(row.get("sma200"), 0.0)
    volume = _as_float(row.get("volume"), 0.0)
    vol_ma50 = _as_float(row.get("vol_ma50"), float("nan"))
    prev_close = _as_float(prev_row.get("close"), 0.0)

    if not (close_px > 0 and high_px > 0 and low_px > 0 and sma50 > 0 and sma150 > 0 and sma200 > 0):
        return None

    strict_trend = bool(params.get("continuation_breakout_strict_trend", False))
    if strict_trend:
        if not (close_px > sma20 > sma50 > sma150 > sma200):
            return None
    else:
        if not (close_px > sma50 > sma150 > sma200):
            return None
        if sma20 > 0 and sma20 < sma50:
            return None

    rs_min = float(params.get("continuation_breakout_rs_min", 75.0) or 75.0)
    if rs_percentile < rs_min:
        return None

    runup_min = float(params.get("continuation_breakout_runup_min_pct", 8.0) or 8.0)
    ret_1m = _as_float(row.get("ret_1m"), float("nan"))
    ret_3m = _as_float(row.get("ret_3m"), float("nan"))
    runup = max(v for v in (ret_1m, ret_3m) if math.isfinite(v)) if (math.isfinite(ret_1m) or math.isfinite(ret_3m)) else 0.0
    if runup < runup_min:
        return None

    prior_highs = pd.to_numeric(df.iloc[i - lookback : i].get("high"), errors="coerce")
    if prior_highs.empty:
        return None
    pivot_high = float(prior_highs.max(skipna=True))
    if not (math.isfinite(pivot_high) and pivot_high > 0):
        return None

    breakout_buffer = max(
        0.0,
        float(params.get("continuation_breakout_buffer_pct", 0.0) or 0.0),
    )
    breakout_level = pivot_high * (1.0 + breakout_buffer)
    if close_px < breakout_level:
        return None

    require_up_close = bool(params.get("continuation_breakout_require_up_close", True))
    if require_up_close:
        if close_px < open_px:
            return None
        if prev_close > 0 and close_px < prev_close:
            return None

    if math.isfinite(vol_ma50) and vol_ma50 > 0:
        vol_min_mult = float(params.get("continuation_breakout_volume_min_mult", 1.0) or 1.0)
        if volume < (vol_ma50 * vol_min_mult):
            return None

    high_52w = _as_float(row.get("high_52w"), float("nan"))
    if not math.isfinite(high_52w):
        high_52w = _as_float(row.get("high52w"), float("nan"))
    near_high_max = max(
        0.0,
        float(params.get("continuation_breakout_near_high_52w_max_pct", 0.15) or 0.15),
    )
    if math.isfinite(high_52w) and high_52w > 0:
        dist_from_high = (high_52w - close_px) / high_52w
        if dist_from_high < 0.0 or dist_from_high > near_high_max:
            return None

    trigger_px = close_px
    stop_sma20_buffer = max(
        0.0,
        float(params.get("continuation_breakout_stop_below_sma20_pct", 0.03) or 0.03),
    )
    stop_pivot_buffer = max(
        0.0,
        float(params.get("continuation_breakout_stop_below_pivot_pct", 0.02) or 0.02),
    )
    sma20_stop = sma20 * (1.0 - stop_sma20_buffer) if sma20 > 0 else float("nan")
    pivot_stop = pivot_high * (1.0 - stop_pivot_buffer)
    stop_px = max(v for v in (sma20_stop, pivot_stop) if math.isfinite(v) and v > 0)
    if low_px > 0:
        stop_px = min(stop_px, low_px * (1.0 - 0.0005))
    if stop_px <= 0 or stop_px >= trigger_px:
        return None

    max_stop_pct = float(params.get("continuation_breakout_max_stop_pct", 0.12) or 0.12)
    stop_width = (trigger_px - stop_px) / trigger_px
    if stop_width > max_stop_pct:
        return None

    rs_component = min(1.0, rs_percentile / 100.0)
    vol_component = 0.5
    if math.isfinite(vol_ma50) and vol_ma50 > 0:
        vol_component = min(1.0, max(0.0, (volume / vol_ma50) / 2.0))
    extension_component = min(
        1.0,
        max(0.0, (close_px - pivot_high) / max(trigger_px * 0.03, 1e-6)),
    )
    strength = (0.55 * rs_component) + (0.25 * vol_component) + (0.20 * extension_component)

    return {
        "trigger_price": trigger_px,
        "stop_price": stop_px,
        "stop_limit_pct": float(params.get("stop_limit_pct", 0.02) or 0.02),
        "stop_loss_type": "low_or_pct",
        "entry_type": "continuation",
        "sleeve": "continuation",
        "max_stop_pct": max_stop_pct,
        "entry_timing": "next_day",
        "signal_mode": "after_close",
        "signal_strength": float(strength),
    }
