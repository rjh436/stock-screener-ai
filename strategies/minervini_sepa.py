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


class MinerviniSEPAStrategy(GenericStrategy):
    """
    Minervini SEPA (daily-bar approximation).
    Trend Template + VCP proxy + buy-stop-limit entry.
    """

    def __init__(self, genome: Dict):
        super().__init__(genome)
        self._name = genome.get("name", "Minervini SEPA")
        self._gate_counts = {
            "checked": 0,
            "trend_template_pass": 0,
            "vcp_pass": 0,
            "stop_width_pass": 0,
            "signal_pass": 0,
        }

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.genome.get("warmup_bars", 200))
        if i < warmup:
            return None

        self._gate_counts["checked"] += 1
        row = df.iloc[i]

        close_px = _as_float(row.get("close"), 0.0)
        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)
        sma200_slope = _as_float(row.get("sma200_slope_1m"), _as_float(row.get("sma200_slope"), 0.0))

        if close_px <= 0 or sma50 <= 0 or sma150 <= 0 or sma200 <= 0:
            return None

        if not (close_px > sma150 > sma200):
            return None
        if sma50 <= sma150 or sma50 <= sma200:
            return None
        if close_px <= sma50:
            return None
        if sma200_slope <= 0:
            return None

        pct_off_high = _as_float(row.get("pct_off_high_52w"), 100.0)
        pct_above_low = _as_float(row.get("pct_above_low_52w"), 0.0)
        if pct_off_high > 25.0:
            return None
        if pct_above_low < 30.0:
            return None

        rs_rating = _as_float(row.get("rs_rating"), 0.0)
        mom_rank = _as_float(row.get("momentum_rank"), 0.0)
        rs_min = float(self.genome.get("rs_min", 70.0))
        if rs_rating < rs_min and mom_rank < rs_min:
            ret_6m = _as_float(row.get("ret_6m"), 0.0)
            if ret_6m < 20.0:
                return None

        self._gate_counts["trend_template_pass"] += 1

        # VCP proxy: contraction + volume dry-up
        if not _contraction_ok(row):
            return None
        if _as_float(row.get("vol_dryup"), 0.0) < 1.0:
            return None

        self._gate_counts["vcp_pass"] += 1

        stop_buy_ref = self.genome.get("stop_buy_ref", "high_20_prev")
        pivot = _as_float(row.get(stop_buy_ref), _as_float(row.get("high_20_prev"), 0.0))
        if pivot <= 0:
            return None

        stop_buy_mult = float(self.genome.get("stop_buy_mult", 1.0))
        trigger = pivot * stop_buy_mult

        low = _as_float(row.get("low"), 0.0)
        if low <= 0 or trigger <= 0:
            return None

        max_stop_pct = float(self.genome.get("max_stop_pct", 0.08))
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
