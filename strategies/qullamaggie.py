from __future__ import annotations

from typing import Dict, Optional
import math

import pandas as pd

from .base import BaseStrategy


def _as_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _pct(value, default: float) -> float:
    out = _as_float(value, default)
    if out > 1.0:
        out /= 100.0
    return max(0.0, out)


class QullamaggieStrategy(BaseStrategy):
    """
    Daily-bar approximation of Qullamaggie-style breakout / episodic-pivot setups.

    Signals are produced after the close and executed on the next session via the
    engine's existing stop / next-day execution model.
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = str(self.params.get("name", "Qullamaggie Breakout")).strip() or "Qullamaggie Breakout"
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    def _passes_global_gates(self, row: pd.Series) -> bool:
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return False

        min_price = _as_float(self.params.get("min_price", 5.0), 5.0)
        if close_px < min_price:
            return False

        min_avg_volume = _as_float(self.params.get("min_avg_volume_30", 250000.0), 250000.0)
        vol_ma30 = _as_float(row.get("vol_ma30"), float("nan"))
        if math.isfinite(vol_ma30) and vol_ma30 < min_avg_volume:
            return False

        market_cap_min = _as_float(self.params.get("market_cap_min", 0.0), 0.0)
        if market_cap_min > 0:
            market_cap = _as_float(
                row.get("market_cap")
                or row.get("mkt_cap")
                or row.get("marketcap")
                or row.get("mktcap"),
                0.0,
            )
            if market_cap > 0 and market_cap < market_cap_min:
                return False

        if bool(self.params.get("require_market_uptrend", True)):
            spy_close = _as_float(row.get("spy_close"), 0.0)
            spy_sma200 = _as_float(row.get("spy_sma200"), 0.0)
            if spy_close > 0 and spy_sma200 > 0 and spy_close < spy_sma200:
                return False

        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)
        if bool(self.params.get("require_close_above_sma50", True)) and sma50 > 0 and close_px < sma50:
            return False
        if bool(self.params.get("require_sma50_above_sma200", True)):
            if not (sma50 > 0 and sma200 > 0 and sma50 > sma200):
                return False
        if bool(self.params.get("require_trend_template", False)):
            if not (close_px > sma50 > sma150 > sma200):
                return False

        near_high_min = _pct(self.params.get("near_high_min_pct", 0.85), 0.85)
        high_52w = _as_float(row.get("high_52w"), 0.0)
        if high_52w > 0 and close_px < (high_52w * near_high_min):
            return False

        rs_min = _as_float(self.params.get("rs_min", 90.0), 90.0)
        rs_rating = max(_as_float(row.get("rs_rating"), 0.0), _as_float(row.get("momentum_rank"), 0.0))
        if rs_rating < rs_min:
            return False

        momentum_rank_min = _as_float(self.params.get("momentum_rank_min", 0.0), 0.0)
        momentum_rank = _as_float(row.get("momentum_rank"), 0.0)
        if momentum_rank_min > 0 and momentum_rank < momentum_rank_min:
            return False

        adr_min = _pct(self.params.get("adr_min_pct", 0.03), 0.03)
        adr = _as_float(row.get("adr_pct"), 0.0)
        if adr_min > 0 and adr < adr_min:
            return False

        return True

    def _breakout_candidate(self, row: pd.Series) -> Optional[Dict]:
        close_px = _as_float(row.get("close"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma20 = _as_float(row.get("vol_ma20"), float("nan"))
        vol_ma10 = _as_float(row.get("vol_ma10"), float("nan"))
        vol_ma50 = _as_float(row.get("vol_ma50"), float("nan"))
        bb_width = _as_float(row.get("bb_width"), float("nan"))
        high_20_prev = _as_float(row.get("high_20_prev"), 0.0)
        if not (close_px > 0 and high_px > 0 and low_px > 0 and high_20_prev > 0):
            return None

        bb_width_max = _as_float(self.params.get("bb_width_max", 0.22), 0.22)
        if math.isfinite(bb_width) and bb_width > bb_width_max:
            return None

        rp5 = _as_float(row.get("range_pct_5"), float("nan"))
        rp10 = _as_float(row.get("range_pct_10"), float("nan"))
        rp20 = _as_float(row.get("range_pct_20"), float("nan"))
        rp40 = _as_float(row.get("range_pct_40"), float("nan"))
        if not all(math.isfinite(x) for x in (rp5, rp10, rp20, rp40)):
            return None
        if not (rp5 < rp10 < rp20 < rp40):
            return None

        if bool(self.params.get("require_volume_dryup", True)):
            if math.isfinite(vol_ma10) and math.isfinite(vol_ma50) and not (vol_ma10 < vol_ma50):
                return None
            dryup_max = _as_float(self.params.get("setup_volume_max_mult", 1.0), 1.0)
            if math.isfinite(vol_ma50) and vol_ma50 > 0 and vol > (vol_ma50 * dryup_max):
                return None

        breakout_buffer = _pct(self.params.get("breakout_buffer", 0.001), 0.001)
        trigger = high_20_prev * (1.0 + breakout_buffer)
        if close_px >= trigger:
            return None

        setup_proximity = _pct(self.params.get("setup_proximity_pct", 0.03), 0.03)
        if close_px < (trigger * (1.0 - setup_proximity)):
            return None

        setup_high_proximity = _pct(self.params.get("setup_high_proximity_pct", 0.012), 0.012)
        if high_px < (trigger * (1.0 - setup_high_proximity)):
            return None

        clv_min = _as_float(self.params.get("setup_clv_min", 0.55), 0.55)
        clv = _as_float(row.get("clv"), 0.0)
        if clv < clv_min:
            return None

        max_stop_pct = _pct(self.params.get("max_stop_pct", 0.08), 0.08)
        stop_floor = trigger * (1.0 - max_stop_pct)
        stop_px = max(low_px, stop_floor)
        if stop_px <= 0 or stop_px >= trigger:
            return None
        stop_width = (trigger - stop_px) / trigger
        if stop_width > max_stop_pct:
            return None

        rs_component = max(_as_float(row.get("rs_rating"), 0.0), _as_float(row.get("momentum_rank"), 0.0))
        adr_component = min(15.0, _as_float(row.get("adr_pct"), 0.0) * 200.0)
        tight_component = min(15.0, max(0.0, rp40 - rp5))
        strength = (0.60 * rs_component) + adr_component + tight_component

        return {
            "trigger_price": trigger,
            "stop_price": stop_px,
            "stop_limit_pct": _pct(self.params.get("stop_limit_pct", 0.03), 0.03),
            "entry_type": "vcp",
            "sleeve": "breakout",
            "max_stop_pct": max_stop_pct,
            "entry_timing": "next_day",
            "signal_mode": "after_close",
            "signal_strength": float(strength),
        }

    def _ep_candidate(self, row: pd.Series) -> Optional[Dict]:
        close_px = _as_float(row.get("close"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        sma20 = _as_float(row.get("sma20"), float("nan"))
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), float("nan"))
        gap_pct = _as_float(row.get("gap_pct"), 0.0)
        if not (close_px > 0 and high_px > 0 and low_px > 0):
            return None

        gap_min = _pct(self.params.get("ep_gap_pct", 0.10), 0.10)
        if gap_pct < gap_min:
            return None

        vol_mult = _as_float(self.params.get("ep_vol_mult", 2.0), 2.0)
        if math.isfinite(vol_ma50) and vol_ma50 > 0 and vol < (vol_ma50 * vol_mult):
            return None

        clv_min = _as_float(self.params.get("ep_close_near_high_min", 0.7), 0.7)
        clv = _as_float(row.get("clv"), 0.0)
        if clv < clv_min:
            return None

        max_extension = _pct(self.params.get("ep_max_extension_above_sma20", 0.10), 0.10)
        if math.isfinite(sma20) and sma20 > 0 and ((close_px - sma20) / sma20) > max_extension:
            return None

        breakout_buffer = _pct(self.params.get("ep_breakout_buffer", 0.0), 0.0)
        trigger = high_px * (1.0 + breakout_buffer)
        max_stop_pct = _pct(self.params.get("ep_max_stop_pct", self.params.get("max_stop_pct", 0.10)), 0.10)
        stop_floor = trigger * (1.0 - max_stop_pct)
        stop_px = max(low_px, stop_floor)
        if stop_px <= 0 or stop_px >= trigger:
            return None
        stop_width = (trigger - stop_px) / trigger
        if stop_width > max_stop_pct:
            return None

        rs_component = max(_as_float(row.get("rs_rating"), 0.0), _as_float(row.get("momentum_rank"), 0.0))
        gap_component = min(25.0, gap_pct * 200.0)
        volume_component = 0.0
        if math.isfinite(vol_ma50) and vol_ma50 > 0:
            volume_component = min(20.0, ((vol / vol_ma50) - 1.0) * 8.0)
        strength = (0.55 * rs_component) + gap_component + volume_component

        return {
            "trigger_price": trigger,
            "stop_price": stop_px,
            "stop_limit_pct": _pct(self.params.get("stop_limit_pct", 0.03), 0.03),
            "entry_type": "ep",
            "sleeve": "breakout",
            "max_stop_pct": max_stop_pct,
            "entry_timing": "next_day",
            "signal_mode": "after_close",
            "signal_strength": float(strength),
        }

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.params.get("warmup_bars", 200) or 200)
        if i < warmup:
            return None
        row = df.iloc[i]
        if not self._passes_global_gates(row):
            return None

        mode = str(self.params.get("entry_mode", "both") or "both").lower()
        candidates = []
        if mode in {"both", "breakout", "vcp"}:
            breakout = self._breakout_candidate(row)
            if breakout is not None:
                candidates.append(breakout)
        if mode in {"both", "ep", "episodic_pivot"}:
            ep = self._ep_candidate(row)
            if ep is not None:
                candidates.append(ep)
        if not candidates:
            return None

        selected = max(candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
        selected = dict(selected)
        selected.pop("signal_strength", None)
        return selected

    def exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_i: int,
        entry_price: float,
        stop_price: float,
    ) -> bool:
        # Engine uses centralized next-day-safe exit handling.
        return False
