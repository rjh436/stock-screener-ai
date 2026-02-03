from typing import Dict, Optional
import math
import pandas as pd
from .generic import GenericStrategy


def _as_float(value, default=0.0):
    try:
        val = float(value)
    except Exception:
        return default
    if not math.isfinite(val):
        return default
    return val


class QullamaggieEpisodicPivotStrategy(GenericStrategy):
    """
    Qullamaggie Episodic Pivot (daily-bar approximation).
    Signal on gap + volume surge, enter on next-day buy-stop-limit.
    """

    def __init__(self, genome: Dict):
        super().__init__(genome)
        self._name = genome.get("name", "Qullamaggie EP")
        self._gate_counts = {
            "checked": 0,
            "gap_pass": 0,
            "volume_pass": 0,
            "stop_width_pass": 0,
            "signal_pass": 0,
        }

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.genome.get("warmup_bars", 60))
        if i < warmup:
            return None

        self._gate_counts["checked"] += 1
        row = df.iloc[i]

        gap_pct = _as_float(row.get("gap_pct"), 0.0)
        gap_min = float(self.genome.get("ep_gap_min", 10.0))
        if gap_pct < gap_min:
            return None
        self._gate_counts["gap_pass"] += 1

        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 1.0)
        vol_mult = float(self.genome.get("ep_vol_mult", 2.0))
        if vol < (vol_ma50 * vol_mult):
            return None
        self._gate_counts["volume_pass"] += 1

        high = _as_float(row.get("high"), 0.0)
        low = _as_float(row.get("low"), 0.0)
        if high <= 0 or low <= 0:
            return None

        stop_buy_mult = float(self.genome.get("stop_buy_mult", 1.0))
        trigger = high * stop_buy_mult

        max_stop_pct = float(self.genome.get("max_stop_pct", 0.10))
        stop_width = (trigger - low) / trigger
        if stop_width > max_stop_pct:
            return None
        self._gate_counts["stop_width_pass"] += 1
        self._gate_counts["signal_pass"] += 1

        return {
            "trigger_price": trigger,
            "stop_price": low,
            "stop_limit_pct": float(self.genome.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_of_day",
        }
