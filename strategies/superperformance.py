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


class SuperperformanceStrategy(BaseStrategy):
    """
    Superperformance Strategy
    - Minervini Trend Filter (Stage 2)
    - Qullamaggie Breakout + Episodic Pivot triggers
    - Hard stop: Low of breakout day or max -5%
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = self.params.get("name", "Superperformance")
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.params.get("warmup_bars", 200))
        if i < warmup:
            return None

        row = df.iloc[i]
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return None

        # Optional trend filter (off by default for Qullamaggie)
        require_trend = bool(self.params.get("require_trend", False))
        rs_min = float(self.params.get("rs_min", 0.0))
        rs_rating = _as_float(row.get("rs_rating"), 0.0)
        if require_trend:
            sma50 = _as_float(row.get("sma50"), 0.0)
            sma150 = _as_float(row.get("sma150"), 0.0)
            sma200 = _as_float(row.get("sma200"), 0.0)
            if not (close_px > sma50 > sma150 > sma200):
                return None
            high_52w = _as_float(row.get("high_52w"), 0.0)
            if high_52w <= 0 or close_px < (0.75 * high_52w):
                return None

        if rs_min > 0 and rs_rating < rs_min:
            return None

        mom_min = float(self.params.get("mom_rank_min", 0.0))
        mom_rank = _as_float(row.get("momentum_rank"), 0.0)
        if mom_min > 0 and mom_rank < mom_min:
            return None

        entry_mode = str(self.params.get("entry_mode", "") or "").lower()
        if entry_mode not in {"breakout", "ep", "both"}:
            entry_mode = "both"

        # --- Breakout Trigger ---
        prior_high = _as_float(row.get("highest10_1"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), vol)
        vol_mult = float(self.params.get("vol_mult", 1.5))

        is_breakout = False
        if entry_mode in {"breakout", "both"}:
            # Qullamaggie Visual Tightness (NATR < threshold for last N days)
            natr_max = float(self.params.get("natr_max", 3.0))
            natr_days = int(self.params.get("natr_days", 10))
            if i >= natr_days:
                natr_window = df["natr"].iloc[i - natr_days + 1 : i + 1]
                if not natr_window.isna().any() and float(natr_window.max()) < natr_max:
                    if prior_high > 0 and close_px > prior_high:
                        if vol_ma50 > 0 and vol >= (vol_ma50 * vol_mult):
                            is_breakout = True

        # --- Episodic Pivot (High Volume Gap) ---
        is_ep = False
        if entry_mode in {"ep", "both"} and i >= 1:
            open_px = _as_float(row.get("open"), 0.0)
            prev_close = _as_float(df.iloc[i - 1].get("close"), 0.0)
            if prev_close > 0 and open_px > 0:
                gap_pct = (open_px - prev_close) / prev_close
                ep_gap = float(self.params.get("ep_gap_pct", 0.10))
                ep_vol_mult = float(self.params.get("ep_vol_mult", 1.5))
                if gap_pct >= ep_gap and vol_ma50 > 0 and vol >= (vol_ma50 * ep_vol_mult):
                    is_ep = True

        if not (is_breakout or is_ep):
            return None

        # --- Entry + Risk ---
        breakout_buffer = float(self.params.get("breakout_buffer", 0.0))
        if is_breakout:
            trigger_px = prior_high * (1.0 + breakout_buffer)
        else:
            trigger_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        if trigger_px <= 0 or low_px <= 0:
            return None

        max_stop_pct = float(self.params.get("max_stop_pct", 0.05))
        hard_stop = trigger_px * (1.0 - max_stop_pct)
        stop_px = max(low_px, hard_stop)

        if stop_px >= trigger_px:
            return None

        return {
            "trigger_price": trigger_px,
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
        # Engine uses centralized exit logic; keep a safe default.
        return False
