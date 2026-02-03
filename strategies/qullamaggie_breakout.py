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


def _contraction_ok(row: pd.Series) -> bool:
    rp5 = _as_float(row.get("range_pct_5"), 999.0)
    rp10 = _as_float(row.get("range_pct_10"), 999.0)
    rp20 = _as_float(row.get("range_pct_20"), 999.0)
    rp40 = _as_float(row.get("range_pct_40"), 999.0)
    return rp5 < rp10 < rp20 < rp40


class QullamaggieBreakoutStrategy(GenericStrategy):
    """
    Qullamaggie Breakout (daily-bar approximation).
    Scans after close and places next-day buy-stop-limit orders.
    """

    def __init__(self, genome: Dict):
        super().__init__(genome)
        self._name = genome.get("name", "Qullamaggie Breakout")
        self._gate_counts = {
            "checked": 0,
            "momentum_pass": 0,
            "base_pass": 0,
            "vol_dryup_pass": 0,
            "adr_pass": 0,
            "stop_width_pass": 0,
            "signal_pass": 0,
        }

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.genome.get("warmup_bars", 120))
        if i < warmup:
            return None

        self._gate_counts["checked"] += 1
        row = df.iloc[i]

        # 1) Momentum / pre-move filter
        mom_rank = _as_float(row.get("momentum_rank"), _as_float(row.get("rs_rating"), 0.0))
        mom_floor = float(self.genome.get("q_momentum_rank_pct", 98.0))
        premove_min = float(self.genome.get("q_premove_min", 30.0))
        ret_3m = _as_float(row.get("ret_3m"), 0.0)
        if mom_rank < mom_floor and ret_3m < premove_min:
            return None
        self._gate_counts["momentum_pass"] += 1

        # 2) Base contraction + volume dry-up
        if not _contraction_ok(row):
            return None
        self._gate_counts["base_pass"] += 1

        if _as_float(row.get("vol_dryup"), 0.0) < 1.0:
            return None
        self._gate_counts["vol_dryup_pass"] += 1

        # 3) ADR (Qullamaggie definition)
        adr = _as_float(row.get("adr_pct_q"), _as_float(row.get("adr_pct"), 0.0))
        adr_min = float(self.genome.get("adr_min", 3.0))
        if adr < adr_min:
            return None
        self._gate_counts["adr_pass"] += 1

        # 4) Trigger and stop
        stop_buy_ref = self.genome.get("stop_buy_ref", "high_20_prev")
        pivot = _as_float(row.get(stop_buy_ref), _as_float(row.get("high_20_prev"), 0.0))
        if pivot <= 0:
            return None

        stop_buy_mult = float(self.genome.get("stop_buy_mult", 1.0))
        trigger = pivot * stop_buy_mult

        low = _as_float(row.get("low"), 0.0)
        if low <= 0 or trigger <= 0:
            return None

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
