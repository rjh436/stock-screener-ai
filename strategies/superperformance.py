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

        # Candle quality (Squat filter)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        clv = _as_float(row.get("clv"), float("nan"))
        if not math.isfinite(clv):
            rng = high_px - low_px
            if rng <= 0:
                return None
            clv = (close_px - low_px) / rng

        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), vol)
        if clv <= 0.70:
            return None
        if vol_ma50 > 0 and vol < (vol_ma50 * 1.25):
            return None
        # Sector confirmation
        sector_rs = _as_float(row.get("sector_rs"), 50.0)
        if sector_rs < 50.0:
            return None

        # Trend/RS filters are applied selectively:
        # - Breakout/VCP: strict Minervini trend + RS filters
        # - EP: relaxed (only require price > SMA50)
        require_trend = bool(self.params.get("require_trend", False))
        rs_min = float(self.params.get("rs_min", 0.0))
        rs_rating = _as_float(row.get("rs_rating"), 0.0)
        mom_min = float(self.params.get("mom_rank_min", 0.0))
        mom_rank = _as_float(row.get("momentum_rank"), 0.0)

        entry_mode = str(self.params.get("entry_mode", "") or "").lower()
        if entry_mode not in {"breakout", "ep", "both"}:
            entry_mode = "both"

        # --- Breakout Trigger ---
        prior_high = _as_float(row.get("highest10_1"), 0.0)
        vol_mult = float(self.params.get("vol_mult", 1.5))

        is_breakout = False
        if entry_mode in {"breakout", "both"}:
            # Qullamaggie Visual Tightness (NATR < threshold for last N days)
            natr_max = float(self.params.get("natr_max", 3.0))
            natr_days = int(self.params.get("natr_days", 10))
            if i >= natr_days:
                natr_window = df["natr"].iloc[i - natr_days + 1 : i + 1]
                if not natr_window.isna().any() and float(natr_window.max()) < natr_max:
                    # Volatility Clamp (base depth)
                    base_ok = True
                    base_depth_pct = _as_float(row.get("range_pct_20"), float("nan"))
                    if not math.isfinite(base_depth_pct):
                        if i >= 19:
                            hi = float(df["high"].iloc[i - 19 : i + 1].max())
                            lo = float(df["low"].iloc[i - 19 : i + 1].min())
                            if lo > 0:
                                base_depth_pct = ((hi - lo) / lo) * 100.0
                    if math.isfinite(base_depth_pct) and base_depth_pct > 30.0:
                        base_ok = False
                    if base_ok and prior_high > 0 and close_px > prior_high:
                        req_vol_mult = max(vol_mult, 1.25)
                        if vol_ma50 > 0 and vol >= (vol_ma50 * req_vol_mult):
                            # Enforce strict Minervini trend on breakouts
                            if require_trend:
                                sma50 = _as_float(row.get("sma50"), 0.0)
                                sma150 = _as_float(row.get("sma150"), 0.0)
                                sma200 = _as_float(row.get("sma200"), 0.0)
                                if not (close_px > sma50 > sma150 > sma200):
                                    pass
                                else:
                                    high_52w = _as_float(row.get("high_52w"), 0.0)
                                    if high_52w > 0 and close_px >= (0.75 * high_52w):
                                        if (rs_min <= 0 or rs_rating >= rs_min) and (mom_min <= 0 or mom_rank >= mom_min):
                                            is_breakout = True
                            else:
                                if (rs_min <= 0 or rs_rating >= rs_min) and (mom_min <= 0 or mom_rank >= mom_min):
                                    is_breakout = True

        # --- Episodic Pivot (High Volume Gap) ---
        is_ep = False
        if entry_mode in {"ep", "both"} and i >= 1:
            open_px = _as_float(row.get("open"), 0.0)
            prev_close = _as_float(df.iloc[i - 1].get("close"), 0.0)
            if prev_close > 0 and open_px > 0:
                gap_pct = (open_px - prev_close) / prev_close
                ep_gap = max(float(self.params.get("ep_gap_pct", 0.10)), 0.04)
                ep_vol_mult = max(float(self.params.get("ep_vol_mult", 1.5)), 1.5)
                req_ep_mult = max(ep_vol_mult, 1.25)
                if gap_pct >= ep_gap and vol_ma50 > 0 and vol >= (vol_ma50 * req_ep_mult):
                    sma50 = _as_float(row.get("sma50"), 0.0)
                    if sma50 > 0 and close_px > sma50:
                        # EP quality floor
                        natr_val = _as_float(row.get("natr"), 0.0)
                        if rs_rating >= 60 and natr_val <= 5.0:
                            is_ep = True

        if not (is_breakout or is_ep):
            return None

        # --- Entry + Risk ---
        breakout_buffer = float(self.params.get("breakout_buffer", 0.0))
        if is_breakout:
            trigger_px = prior_high * (1.0 + breakout_buffer)
        elif is_ep:
            trigger_px = close_px
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

    def pyramid(self, df: pd.DataFrame, i: int, position: Dict) -> Optional[Dict]:
        """
        Pyramiding signal:
        - If profit exceeds threshold, add fraction to position.
        - Engine enforces max positions and sizing constraints.
        """
        if position is None:
            return None

        entry_price = _as_float(position.get("entry_price"), 0.0)
        if entry_price <= 0:
            return None

        max_adds = int(self.params.get("pyramid_max_adds", 1) or 1)
        pyramids = int(position.get("pyramids", 0) or 0)
        if pyramids >= max_adds:
            return None

        close_px = _as_float(df.iloc[i].get("close"), 0.0)
        if close_px <= 0:
            return None

        threshold = float(self.params.get("pyramid_threshold", 0.05) or 0.05)
        profit_pct = (close_px - entry_price) / entry_price
        if profit_pct < threshold:
            return None

        sma20 = _as_float(df.iloc[i].get("sma20"), 0.0)
        if sma20 > 0 and close_px <= sma20:
            return None

        rs_rating = _as_float(df.iloc[i].get("rs_rating"), 0.0)
        if rs_rating < 80:
            return None

        add_fraction = float(self.params.get("pyramid_fraction", 0.5) or 0.5)
        if add_fraction <= 0:
            return None

        return {
            "add_fraction": add_fraction,
            "stop_to_avg_cost": bool(self.params.get("pyramid_stop_to_avg_cost", True)),
        }
