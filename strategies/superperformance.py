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

        # OHLCV
        open_px = _as_float(row.get("open"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), vol)
        # Market cap filter (if available)
        market_cap_min = float(self.params.get("market_cap_min", 0.0) or 0.0)
        if market_cap_min > 0:
            market_cap = _as_float(
                row.get("market_cap")
                or row.get("mkt_cap")
                or row.get("marketcap")
                or row.get("mktcap"),
                0.0,
            )
            if market_cap > 0 and market_cap < market_cap_min:
                return None

        # Shared indicators
        rs_rating = _as_float(row.get("rs_rating"), 0.0)
        sma50_val = _as_float(row.get("sma50"), 0.0)
        sma200_val = _as_float(row.get("sma200"), 0.0)
        sector_rs = _as_float(row.get("sector_rs"), 50.0)

        vcp_rs_min = float(self.params.get("vcp_rs_min", 80.0))
        vcp_sector_min = float(self.params.get("vcp_sector_min", 50.0))
        ep_rs_min = float(self.params.get("ep_rs_min", 60.0))

        entry_mode = str(self.params.get("entry_mode", "") or "").lower()
        if entry_mode not in {"breakout", "ep", "both"}:
            entry_mode = "both"

        # --- VCP / Trend Trigger ---
        prior_high = _as_float(
            row.get("high_20_prev")
            or row.get("highest10_1")
            or row.get("prev_high"),
            0.0,
        )
        is_breakout = False
        if entry_mode in {"breakout", "both"}:
            base_ok = True
            if close_px <= sma200_val:
                base_ok = False
            if rs_rating < vcp_rs_min:
                base_ok = False
            if vcp_sector_min > 0 and sector_rs < vcp_sector_min:
                base_ok = False

            # VCP structure
            base_depth_pct = _as_float(row.get("range_pct_20"), float("nan"))
            if not math.isfinite(base_depth_pct):
                if i >= 19:
                    hi = float(df["high"].iloc[i - 19 : i + 1].max())
                    lo = float(df["low"].iloc[i - 19 : i + 1].min())
                    if lo > 0:
                        base_depth_pct = ((hi - lo) / lo) * 100.0
            rp5 = _as_float(row.get("range_pct_5"), float("nan"))
            rp10 = _as_float(row.get("range_pct_10"), float("nan"))
            rp20 = _as_float(row.get("range_pct_20"), float("nan"))
            rp40 = _as_float(row.get("range_pct_40"), float("nan"))
            last_contraction = min(rp5, rp10)

            contractions = 0
            if math.isfinite(rp10) and rp10 <= 20.0:
                contractions += 1
            if math.isfinite(rp20) and rp20 <= 25.0:
                contractions += 1
            if math.isfinite(rp40) and rp40 <= 30.0:
                contractions += 1

            if not math.isfinite(last_contraction) or last_contraction >= 10.0:
                base_ok = False
            if contractions < 2:
                base_ok = False

            # Volume dry-up
            dry_ok = True
            if i > 0:
                prev_row = df.iloc[i - 1]
                vol_prev = _as_float(prev_row.get("volume"), 0.0)
                vol_ma50_prev = _as_float(prev_row.get("vol_ma50"), 0.0)
                if vol_ma50_prev > 0 and vol_prev > (vol_ma50_prev * 0.75):
                    dry_ok = False
            if not dry_ok:
                base_ok = False

            if base_ok and prior_high > 0:
                is_breakout = True

        # --- Episodic Pivot (High Volume Gap) ---
        is_ep = False
        if entry_mode in {"ep", "both"} and i >= 1:
            prev_close = _as_float(df.iloc[i - 1].get("close"), 0.0)
            if prev_close > 0 and open_px > 0:
                gap_pct = (open_px - prev_close) / prev_close
                ep_gap = max(float(self.params.get("ep_gap_pct", 0.06)), 0.06)
                ep_vol_mult = max(float(self.params.get("ep_vol_mult", 2.0)), 2.0)
                if (
                    gap_pct >= ep_gap
                    and vol_ma50 > 0
                    and vol >= (vol_ma50 * ep_vol_mult)
                    and close_px > open_px
                    and rs_rating >= ep_rs_min
                ):
                    day_range = _as_float(row.get("true_range"), 0.0)
                    if day_range <= 0:
                        day_range = max(high_px - low_px, 0.0)
                    tr_ma50 = _as_float(row.get("tr_ma50"), 0.0)
                    if tr_ma50 > 0 and day_range > (1.25 * tr_ma50):
                        is_ep = True

        if not (is_breakout or is_ep):
            return None

        # --- Entry + Risk ---
        breakout_buffer = float(self.params.get("breakout_buffer", 0.001))
        if is_breakout:
            trigger_px = prior_high * (1.0 + breakout_buffer)
            entry_type = "vcp"
        elif is_ep:
            ep_entry_mode = str(self.params.get("ep_entry_mode", "close")).lower()
            trigger_px = open_px if ep_entry_mode == "open" else close_px
            entry_type = "ep"
        else:
            trigger_px = _as_float(row.get("high"), 0.0)
            entry_type = "unknown"
        low_px = _as_float(row.get("low"), 0.0)
        if trigger_px <= 0 or low_px <= 0:
            return None

        max_stop_pct = float(self.params.get("max_stop_pct", 0.05))
        if entry_type == "ep":
            stop_px = low_px  # EP: low-of-day stop
            ep_max_stop_pct = float(self.params.get("ep_max_stop_pct", 0.12))
            if trigger_px > 0:
                stop_width = (trigger_px - stop_px) / trigger_px
                if stop_width > ep_max_stop_pct:
                    return None
        else:
            hard_stop = trigger_px * (1.0 - max_stop_pct)
            stop_px = max(low_px, hard_stop)

        if stop_px >= trigger_px:
            return None

        return {
            "trigger_price": trigger_px,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
            "entry_type": entry_type,
            "max_stop_pct": float(self.params.get("ep_max_stop_pct", 0.12)) if entry_type == "ep" else None,
            # Allow EP to enter on the gap day (same-day open)
            "entry_timing": "same_day_close" if entry_type == "ep" else "next_day",
            "signal_mode": "close" if entry_type == "ep" else None,
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
