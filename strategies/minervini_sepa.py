from typing import Dict, Optional
import math
import pandas as pd

from .base import BaseStrategy


def _as_float(value, default=0.0) -> float:
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


class MinerviniSEPAStrategy(BaseStrategy):
    """
    Minervini SEPA (daily-bar approximation).
    Trend Template + VCP proxy + buy-stop entry.
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = self.params.get("name", "Minervini SEPA")
        self._gate_counts = {
            "checked": 0,
            "trend_template_pass": 0,
            "vcp_pass": 0,
            "stop_width_pass": 0,
            "signal_pass": 0,
        }
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.params.get("warmup_bars", 200))
        if i < warmup:
            return None

        self._gate_counts["checked"] += 1
        row = df.iloc[i]

        close_px = _as_float(row.get("close"), 0.0)
        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)

        if close_px <= 0 or sma50 <= 0 or sma150 <= 0 or sma200 <= 0:
            return None

        # Minervini Trend Filter (Strict)
        if not (close_px > sma50 > sma150 > sma200):
            return None

        high_52w = _as_float(row.get("high_52w"), 0.0)
        if high_52w <= 0 or close_px < (0.75 * high_52w):
            return None

        rs_min = float(self.params.get("rs_min", 80))
        rs_rating = _as_float(row.get("rs_rating"), 0.0)
        if rs_rating < rs_min:
            return None

        self._gate_counts["trend_template_pass"] += 1

        # VCP proxy: contraction + volume dry-up
        if not _contraction_ok(row):
            return None
        if _as_float(row.get("vol_dryup"), 0.0) < 1.0:
            return None

        self._gate_counts["vcp_pass"] += 1

        stop_buy_ref = self.params.get("stop_buy_ref", "high_20_prev")
        pivot = _as_float(row.get(stop_buy_ref), _as_float(row.get("high_20_prev"), 0.0))
        if pivot <= 0:
            return None

        # Require breakout day confirmation (close > pivot) + volume expansion
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), vol)
        vol_mult = float(self.params.get("vol_mult", 1.5))
        if close_px <= pivot:
            return None
        if vol_ma50 > 0 and vol < (vol_ma50 * vol_mult):
            return None

        trigger = pivot * float(self.params.get("stop_buy_mult", 1.0))
        low_px = _as_float(row.get("low"), 0.0)
        if low_px <= 0 or trigger <= 0:
            return None

        max_stop_pct = float(self.params.get("max_stop_pct", 0.05))
        stop_width = (trigger - low_px) / trigger
        if stop_width > max_stop_pct:
            return None

        self._gate_counts["stop_width_pass"] += 1
        self._gate_counts["signal_pass"] += 1

        stop_px = max(low_px, trigger * (1.0 - max_stop_pct))

        return {
            "trigger_price": trigger,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
        }

    def exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_i: int,
        entry_price: float,
        stop_price: float,
    ) -> bool:
        return False
