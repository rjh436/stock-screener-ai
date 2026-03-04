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


def evaluate_continuation(
    df: pd.DataFrame,
    i: int,
    params: Dict[str, Any],
    rs_percentile: float,
) -> Optional[Dict[str, Any]]:
    """
    Pullback continuation sleeve:
    - Confirmed uptrend (close > sma50 > sma150 > sma200)
    - Strong but not necessarily elite RS
    - Pullback into support followed by reclaim/breakout confirmation
    """
    if i < 1:
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

    if not (close_px > 0 and sma20 > 0 and sma50 > 0 and sma150 > 0 and sma200 > 0):
        return None
    if not (close_px > sma50 > sma150 > sma200):
        return None

    rs_min = float(params.get("continuation_rs_min", 75.0) or 75.0)
    if rs_percentile < rs_min:
        return None

    runup_min = float(params.get("continuation_runup_min_pct", 10.0) or 10.0)
    ret_1m = _as_float(row.get("ret_1m"), float("nan"))
    ret_3m = _as_float(row.get("ret_3m"), float("nan"))
    runup = max(v for v in (ret_1m, ret_3m) if math.isfinite(v)) if (math.isfinite(ret_1m) or math.isfinite(ret_3m)) else 0.0
    if runup < runup_min:
        return None

    # Keep continuation sleeve focused on true leaders near highs.
    high_52w = _as_float(row.get("high_52w"), float("nan"))
    if not math.isfinite(high_52w):
        high_52w = _as_float(row.get("high52w"), float("nan"))
    near_high_max = max(
        0.0,
        float(params.get("continuation_near_high_52w_max_pct", 0.12) or 0.12),
    )
    if math.isfinite(high_52w) and high_52w > 0:
        dist_from_high = (high_52w - close_px) / high_52w
        if dist_from_high < 0 or dist_from_high > near_high_max:
            return None

    require_recent_high_pullback = bool(
        params.get("continuation_require_recent_high_pullback", True)
    )
    if require_recent_high_pullback:
        lookback_bars = max(
            15,
            int(params.get("continuation_recent_high_lookback_bars", 80) or 80),
        )
        max_age_bars = max(
            3,
            int(params.get("continuation_recent_high_max_age_bars", 35) or 35),
        )
        pullback_min = max(
            0.0,
            float(
                params.get("continuation_recent_high_pullback_min_pct", 0.02) or 0.02
            ),
        )
        pullback_max = max(
            pullback_min,
            float(
                params.get("continuation_recent_high_pullback_max_pct", 0.18) or 0.18
            ),
        )

        start = max(0, i - lookback_bars)
        prior = (
            pd.to_numeric(df.iloc[start:i].get("close"), errors="coerce")
            .to_numpy(copy=False)
        )
        if prior.size == 0:
            return None
        finite_mask = pd.notna(prior)
        if not finite_mask.any():
            return None
        prior_valid = prior[finite_mask]
        recent_high = float(prior_valid.max())
        if not (recent_high > 0 and math.isfinite(recent_high)):
            return None

        # Map the recent-high index back to global index for age gating.
        valid_indices = pd.RangeIndex(start, i)[finite_mask]
        high_idx = int(valid_indices[int(prior_valid.argmax())])
        high_age = max(0, i - high_idx)
        if high_age > max_age_bars:
            return None

        pullback_from_high = (recent_high - close_px) / recent_high
        if not (pullback_min <= pullback_from_high <= pullback_max):
            return None

    # Demand that price actually pulled into support in the recent window.
    require_support_touch = bool(params.get("continuation_require_support_touch", True))
    if require_support_touch:
        support_lookback = max(
            3,
            int(params.get("continuation_support_touch_lookback_bars", 8) or 8),
        )
        support_tol = max(
            0.0,
            float(params.get("continuation_support_touch_tolerance_pct", 0.015) or 0.015),
        )
        start = max(0, i - support_lookback + 1)
        lows = pd.to_numeric(df.iloc[start : i + 1].get("low"), errors="coerce")
        sma20s = pd.to_numeric(df.iloc[start : i + 1].get("sma20"), errors="coerce")
        sma50s = pd.to_numeric(df.iloc[start : i + 1].get("sma50"), errors="coerce")
        if lows.empty or sma20s.empty or sma50s.empty:
            return None
        support20 = sma20s * (1.0 + support_tol)
        support50 = sma50s * (1.0 + support_tol)
        touched = ((lows <= support20) | (lows <= support50)) & lows.notna()
        if not bool(touched.any()):
            return None

    pullback_max_pct = float(params.get("continuation_pullback_max_pct", 0.035) or 0.035)
    anchor = sma20 if close_px >= sma20 else sma50
    if anchor <= 0:
        return None
    pullback_dist = abs(close_px - anchor) / anchor
    if pullback_dist > pullback_max_pct:
        return None

    # Confirmation: require continuation day to reclaim short-term pivot.
    require_reclaim_breakout = bool(
        params.get("continuation_require_reclaim_breakout", True)
    )
    prior_pivot_high = high_px
    if require_reclaim_breakout:
        reclaim_lookback = max(
            2,
            int(params.get("continuation_reclaim_lookback_bars", 5) or 5),
        )
        if i < reclaim_lookback:
            return None
        prior_highs = pd.to_numeric(
            df.iloc[i - reclaim_lookback : i].get("high"),
            errors="coerce",
        )
        if prior_highs.empty:
            return None
        prior_pivot_high = float(prior_highs.max(skipna=True))
        if not (math.isfinite(prior_pivot_high) and prior_pivot_high > 0):
            return None
        reclaim_buffer = max(
            0.0,
            float(params.get("continuation_reclaim_buffer_pct", 0.0) or 0.0),
        )
        if close_px < (prior_pivot_high * (1.0 + reclaim_buffer)):
            return None

    if math.isfinite(vol_ma50) and vol_ma50 > 0:
        vol_max_mult = float(params.get("continuation_pullback_volume_max_mult", 1.20) or 1.20)
        if volume > (vol_ma50 * vol_max_mult):
            return None

    require_up_close = bool(params.get("continuation_require_up_close", True))
    if require_up_close:
        if close_px < open_px:
            return None
        if prev_close > 0 and close_px < prev_close:
            return None

    reclaim_strength_min = float(params.get("continuation_reclaim_strength_min", 0.0) or 0.0)
    day_strength = (close_px - open_px) / close_px if close_px > 0 else 0.0
    if day_strength < reclaim_strength_min:
        return None

    trigger_px = close_px
    stop_lookback = max(
        2,
        int(params.get("continuation_stop_lookback_bars", 6) or 6),
    )
    recent_lows = pd.to_numeric(
        df.iloc[max(0, i - stop_lookback + 1) : i + 1].get("low"),
        errors="coerce",
    )
    swing_low = float(recent_lows.min(skipna=True)) if not recent_lows.empty else float("nan")
    swing_buffer = max(
        0.0,
        float(params.get("continuation_stop_below_swing_low_pct", 0.002) or 0.002),
    )
    swing_stop = swing_low * (1.0 - swing_buffer) if math.isfinite(swing_low) and swing_low > 0 else float("nan")
    stop_buffer_sma20 = float(params.get("continuation_stop_below_sma20_pct", 0.02) or 0.02)
    stop_buffer_sma50 = float(params.get("continuation_stop_below_sma50_pct", 0.03) or 0.03)
    sma20_floor = sma20 * (1.0 - max(0.0, stop_buffer_sma20))
    sma50_floor = sma50 * (1.0 - max(0.0, stop_buffer_sma50))
    structure_floor = max(sma20_floor, sma50_floor)
    if math.isfinite(swing_stop):
        stop_px = max(swing_stop, structure_floor)
    else:
        stop_px = structure_floor
    if low_px > 0:
        stop_px = min(stop_px, low_px * (1.0 - 0.0005))
    if stop_px <= 0 or stop_px >= trigger_px:
        return None

    max_stop_pct = float(params.get("continuation_max_stop_pct", 0.10) or 0.10)
    stop_width = (trigger_px - stop_px) / trigger_px
    if stop_width > max_stop_pct:
        return None

    rs_component = min(1.0, rs_percentile / 100.0)
    proximity_component = 1.0 - min(1.0, pullback_dist / max(pullback_max_pct, 1e-6))
    reclaim_component = min(
        1.0,
        max(
            0.0,
            (close_px - prior_pivot_high) / max(trigger_px * 0.03, 1e-6),
        ),
    )
    strength = (0.55 * rs_component) + (0.25 * proximity_component) + (0.20 * reclaim_component)

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
