from __future__ import annotations
from typing import Dict, Tuple, Any
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
