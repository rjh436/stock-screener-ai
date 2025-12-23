from __future__ import annotations
from typing import Dict, Tuple, Any
import math
import pandas as pd
from execution.engine import DEFAULT_SCORING_WEIGHTS

def resolve_signal_index(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Returns (signal_idx, current_idx) to ensure strict parity.
    - If a LIVE (injected) bar exists, Signal is yesterday (n-2).
    - If only HISTORICAL bars exist, Signal is the last closed bar (n-1).
    """
    if df is None or df.empty:
        return -1, -1

    n = len(df)
    current_i = n - 1

    # Check for the live bar flag injected by loader.py
    is_live = False
    if "is_live_bar" in df.columns:
        # Check the last row for the flag
        if bool(df.iloc[-1].get("is_live_bar", False)):
            is_live = True

    if is_live:
        # We are mid-day. The last bar is forming. Signal was Yesterday.
        return n - 2, current_i
    else:
        # We are post-close or pre-open. The last bar is the confirmed Signal.
        return n - 1, current_i

def get_strategy_weights(params: Dict[str, Any]) -> Dict[str, float]:
    """Merge default weights with strategy-specific overrides."""
    w = dict(DEFAULT_SCORING_WEIGHTS)
    if not params:
        return w

    strat_weights = params.get("scoring_weights")
    if isinstance(strat_weights, dict):
        for k, v in strat_weights.items():
            try:
                w[k] = float(v)
            except (TypeError, ValueError):
                continue
    return w

def apply_wealth_boost(score: float, strategy_name: str) -> float:
    """Centralize the Wealth Strategy multiplier logic."""
    # Matches the logic previously hardcoded in app.py
    if "wealth" in str(strategy_name).lower():
        return score * 1.3
    return score

def apply_gap_atr_stop_penalty(stop_mult: float, gap_pct: float, atr_pct: float) -> float:
    """Apply the 1.2x stop multiplier when gap/ATR thresholds are met."""
    try:
        stop_mult_val = float(stop_mult)
    except (TypeError, ValueError):
        return stop_mult
    if gap_pct > -0.03 and atr_pct > 5.0:
        return stop_mult_val * 1.2
    return stop_mult_val

def compute_limit_fill(
    prev_close: float,
    open_px: float,
    low_px: float,
    limit_ratio: Any,
) -> Tuple[bool, float]:
    """
    Determine if a limit order would fill intraday and return the entry price.
    Returns (filled, entry_px). If no limit ratio is provided, treats as a market fill.
    """
    try:
        prev_close_val = float(prev_close)
        open_val = float(open_px)
    except (TypeError, ValueError):
        return False, float("nan")
    if not (math.isfinite(prev_close_val) and math.isfinite(open_val)) or prev_close_val <= 0 or open_val <= 0:
        return False, float("nan")

    try:
        low_val = float(low_px)
        if not math.isfinite(low_val) or low_val <= 0:
            low_val = open_val
    except (TypeError, ValueError):
        low_val = open_val

    if limit_ratio is None:
        return True, open_val

    try:
        limit_ratio_val = float(limit_ratio)
    except (TypeError, ValueError):
        return True, open_val

    target_px = prev_close_val * limit_ratio_val
    if open_val < target_px:
        return True, open_val
    if low_val < target_px:
        return True, target_px
    return False, target_px

def apply_strategy_score_multipliers(score: Any, params: Dict[str, Any]) -> Any:
    """Apply strategy score multipliers based on explicit strategy metadata."""
    role = str(
        params.get("type")
        or params.get("strategy_role")
        or params.get("role")
        or params.get("category")
        or ""
    ).lower()
    if "wealth" in role:
        return score * 1.3
    return score
