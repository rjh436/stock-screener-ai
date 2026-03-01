from __future__ import annotations

from typing import Dict, List, Optional
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


def _first_finite(values, default: float = float("nan")) -> float:
    for value in values:
        out = _as_float(value, float("nan"))
        if math.isfinite(out):
            return out
    return default


def _pct(value, default: float = 0.0) -> float:
    out = _as_float(value, default)
    if out > 1.0:
        out /= 100.0
    return max(0.0, out)


class ApexFusionStrategy(BaseStrategy):
    """
    Non-leveraged, EOD-safe trend strategy with two entry archetypes:
    1) Predictive VCP setup trigger (place stop-buy next session).
    2) Strict Episodic Pivot continuation.
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = str(self.params.get("name", "Apex Fusion")).strip() or "Apex Fusion"
        self.last_reject_reason = ""
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    def _reject(self, reason: str) -> bool:
        self.last_reject_reason = str(reason or "")
        return False

    def _resolve_rs(self, row: pd.Series) -> float:
        return max(
            _first_finite([row.get("rs_rating"), row.get("momentum_rank")], default=0.0),
            0.0,
        )

    def _resolve_momentum(self, row: pd.Series) -> float:
        return max(_as_float(row.get("momentum_rank"), 0.0), 0.0)

    def _passes_global_gates(self, df: pd.DataFrame, i: int) -> bool:
        row = df.iloc[i]

        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return self._reject("close_invalid")

        min_price = _as_float(self.params.get("min_price", 10.0), 10.0)
        if close_px < min_price:
            return self._reject(f"min_price {close_px:.2f}<{min_price:.2f}")

        min_avg_volume = _as_float(self.params.get("min_avg_volume_30", 150000.0), 150000.0)
        vol_ma30 = _as_float(row.get("vol_ma30"), float("nan"))
        if min_avg_volume > 0 and math.isfinite(vol_ma30) and vol_ma30 < min_avg_volume:
            return self._reject(f"liq vol_ma30={vol_ma30:.0f}<{min_avg_volume:.0f}")

        adr_min = _pct(self.params.get("adr_min_pct", 0.035), 0.035)
        if adr_min > 0:
            adr = _first_finite([row.get("adr_pct"), row.get("adr_pct_q")], default=0.0)
            if adr < adr_min:
                return self._reject(f"adr {adr:.3f}<{adr_min:.3f}")

        if bool(self.params.get("require_market_uptrend", True)):
            spy_close = _as_float(row.get("spy_close"), 0.0)
            spy_sma200 = _as_float(row.get("spy_sma200"), 0.0)
            spy_sma50 = _as_float(row.get("spy_sma50"), 0.0)
            if spy_close > 0 and spy_sma200 > 0 and spy_close < spy_sma200:
                return self._reject("market_below_sma200")
            if bool(self.params.get("require_spy_sma50_above_sma200", False)):
                if spy_sma50 > 0 and spy_sma200 > 0 and spy_sma50 < spy_sma200:
                    return self._reject("market_sma50_below_sma200")

        sma20 = _as_float(row.get("sma20"), 0.0)
        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)
        if not (close_px > sma50 > sma150 > sma200):
            return self._reject("trend_template_failed")
        if bool(self.params.get("require_sma20_support", True)) and sma20 > 0 and close_px < sma20:
            return self._reject("close_below_sma20")

        lookback = int(self.params.get("sma200_rising_lookback_bars", 20) or 20)
        if (i - lookback) >= 0:
            sma200_prev = _as_float(df.iloc[i - lookback].get("sma200"), float("nan"))
            if math.isfinite(sma200_prev) and sma200 <= sma200_prev:
                return self._reject("sma200_not_rising")

        high_52w = _as_float(row.get("high_52w"), 0.0)
        near_high_min = _pct(self.params.get("near_high_min_pct", 0.85), 0.85)
        if high_52w > 0 and close_px < (high_52w * near_high_min):
            return self._reject("not_near_52w_high")

        runup_min = _as_float(self.params.get("prior_runup_min_pct", 30.0), 30.0)
        prior_runup = max(
            _as_float(row.get("ret_1m"), 0.0),
            _as_float(row.get("ret_3m"), 0.0),
        )
        if prior_runup < runup_min:
            return self._reject(f"runup {prior_runup:.2f}<{runup_min:.2f}")

        rs = self._resolve_rs(row)
        rs_gate = _as_float(self.params.get("rs_gate_min", 90.0), 90.0)
        if rs < rs_gate:
            return self._reject(f"rs {rs:.2f}<{rs_gate:.2f}")

        mom = self._resolve_momentum(row)
        mom_gate = _as_float(self.params.get("momentum_gate_min", 80.0), 80.0)
        if mom > 0 and mom < mom_gate:
            return self._reject(f"momentum {mom:.2f}<{mom_gate:.2f}")

        sector_rs_min = _as_float(self.params.get("sector_rs_min", 50.0), 50.0)
        sector_rs = _as_float(row.get("sector_rs"), float("nan"))
        if math.isfinite(sector_rs) and sector_rs < sector_rs_min:
            return self._reject(f"sector_rs {sector_rs:.1f}<{sector_rs_min:.1f}")

        max_vix = _as_float(self.params.get("max_vix", 40.0), 40.0)
        vix = _as_float(row.get("vix"), float("nan"))
        if math.isfinite(vix) and max_vix > 0 and vix > max_vix:
            return self._reject(f"vix {vix:.2f}>{max_vix:.2f}")

        growth_min = _as_float(self.params.get("fundamental_growth_min_pct", 15.0), 15.0)
        eps_yoy = _as_float(row.get("eps_growth_yoy"), float("nan"))
        sales_yoy = _as_float(row.get("sales_growth_yoy"), float("nan"))
        eps_ok = math.isfinite(eps_yoy) and eps_yoy >= growth_min
        sales_ok = math.isfinite(sales_yoy) and sales_yoy >= growth_min
        if not (eps_ok or sales_ok):
            if not bool(self.params.get("fundamental_override_enabled", True)):
                return self._reject("fundamental_gate_failed")
            override_rs = _as_float(self.params.get("fundamental_override_rs_min", 95.0), 95.0)
            override_ret3m = _as_float(self.params.get("fundamental_override_ret3m_min", 35.0), 35.0)
            if not (rs >= override_rs and _as_float(row.get("ret_3m"), 0.0) >= override_ret3m):
                return self._reject("fundamental_override_failed")

        inst_min = _as_float(self.params.get("institutional_sponsorship_min", 0.0), 0.0)
        if inst_min > 0:
            inst = _as_float(row.get("institutional_sponsorship"), float("nan"))
            if math.isfinite(inst) and inst < inst_min:
                return self._reject("institutional_sponsorship_low")

        return True

    def _build_vcp_setup_candidate(self, row: pd.Series) -> Optional[Dict]:
        close_px = _as_float(row.get("close"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 0.0)
        if close_px <= 0 or high_px <= 0 or low_px <= 0:
            return None

        pivot = _first_finite(
            [row.get("highest20_1"), row.get("highest10_1"), row.get("prev_high")],
            default=0.0,
        )
        if pivot <= 0:
            return None

        breakout_buffer = _pct(self.params.get("breakout_buffer", 0.0015), 0.0015)
        trigger = pivot * (1.0 + breakout_buffer)
        if close_px > trigger:
            return None

        setup_proximity_pct = _pct(self.params.get("setup_proximity_pct", 0.025), 0.025)
        if close_px < (trigger * (1.0 - setup_proximity_pct)):
            return None

        setup_high_proximity_pct = _pct(self.params.get("setup_high_proximity_pct", 0.012), 0.012)
        if high_px < (trigger * (1.0 - setup_high_proximity_pct)):
            return None

        setup_volume_max_mult = _as_float(self.params.get("setup_volume_max_mult", 1.20), 1.20)
        vol_mult = (vol / vol_ma50) if vol_ma50 > 0 else 0.0
        if vol_ma50 > 0 and vol_mult > setup_volume_max_mult:
            return None

        clv_min = _as_float(self.params.get("setup_clv_min", 0.55), 0.55)
        clv = _as_float(row.get("clv"), 0.0)
        if clv < clv_min:
            return None

        rp5 = _as_float(row.get("range_pct_5"), float("nan"))
        rp20 = _as_float(row.get("range_pct_20"), float("nan"))
        rp40 = _as_float(row.get("range_pct_40"), float("nan"))
        max_base_depth = _as_float(self.params.get("setup_max_base_depth_pct", 30.0), 30.0)
        contraction_ratio = _as_float(self.params.get("setup_contraction_ratio", 0.80), 0.80)
        if not (math.isfinite(rp5) and math.isfinite(rp20) and math.isfinite(rp40)):
            return None
        if rp20 > max_base_depth or rp40 > (max_base_depth * 1.2):
            return None
        if rp5 > (rp20 * contraction_ratio):
            return None

        vol_dryup = _as_float(row.get("vol_dryup"), float("nan"))
        vol_dryup_max = _as_float(self.params.get("setup_vol_dryup_max", 1.0), 1.0)
        if math.isfinite(vol_dryup) and vol_dryup > vol_dryup_max:
            return None

        max_stop_pct = _pct(self.params.get("max_stop_pct", 0.06), 0.06)
        stop_floor = trigger * (1.0 - max_stop_pct)
        raw_stop = min(low_px, _as_float(row.get("sma20"), low_px))
        stop_px = max(raw_stop, stop_floor)
        if not (stop_px > 0 and stop_px < trigger):
            return None
        stop_width = (trigger - stop_px) / trigger
        if stop_width > max_stop_pct:
            return None

        rs = self._resolve_rs(row)
        mom = self._resolve_momentum(row)
        ret_3m = max(_as_float(row.get("ret_3m"), 0.0), 0.0)
        tight_bonus = max(0.0, min(18.0, max_base_depth - rp5))
        dryup_bonus = 0.0
        if vol_ma50 > 0:
            dryup_bonus = max(0.0, min(12.0, (setup_volume_max_mult - vol_mult) * 12.0))
        strength = (0.45 * rs) + (0.20 * mom) + min(25.0, ret_3m * 0.45) + tight_bonus + dryup_bonus

        return {
            "trigger_price": trigger,
            "stop_price": stop_px,
            "stop_limit_pct": _pct(self.params.get("stop_limit_pct", 0.04), 0.04),
            "entry_type": "vcp",
            "entry_timing": "next_day",
            "signal_strength": strength,
            "max_stop_pct": max_stop_pct,
        }

    def _build_ep_candidate(self, row: pd.Series) -> Optional[Dict]:
        if not bool(self.params.get("enable_ep", False)):
            return None

        close_px = _as_float(row.get("close"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 0.0)
        gap_pct = _as_float(row.get("gap_pct"), 0.0)
        clv = _as_float(row.get("clv"), 0.0)
        prev_high = _as_float(row.get("prev_high"), float("nan"))
        sma20 = _as_float(row.get("sma20"), float("nan"))
        if close_px <= 0 or low_px <= 0:
            return None

        gap_min = _as_float(self.params.get("ep_gap_pct", 10.0), 10.0)
        vol_min = _as_float(self.params.get("ep_vol_mult", 3.5), 3.5)
        clv_min = _as_float(self.params.get("ep_close_near_high_min", 0.80), 0.80)
        vol_mult = (vol / vol_ma50) if vol_ma50 > 0 else 0.0
        if gap_pct < gap_min or vol_mult < vol_min or clv < clv_min:
            return None
        if math.isfinite(prev_high) and prev_high > 0 and close_px <= (prev_high * 1.005):
            return None

        max_ext = _pct(self.params.get("ep_max_extension_above_sma20", 0.10), 0.10)
        if math.isfinite(sma20) and sma20 > 0 and close_px > (sma20 * (1.0 + max_ext)):
            return None

        max_stop_pct = _pct(self.params.get("ep_max_stop_pct", 0.10), 0.10)
        stop_floor = close_px * (1.0 - max_stop_pct)
        stop_px = max(low_px, stop_floor)
        if not (stop_px > 0 and stop_px < close_px):
            return None

        rs = self._resolve_rs(row)
        strength = (0.45 * rs) + min(28.0, vol_mult * 7.0) + min(24.0, gap_pct * 1.4) + (clv * 12.0)

        return {
            "trigger_price": close_px,
            "stop_price": stop_px,
            "stop_limit_pct": _pct(self.params.get("stop_limit_pct", 0.04), 0.04),
            "entry_type": "ep",
            "entry_timing": "next_day",
            "signal_strength": strength,
            "max_stop_pct": max_stop_pct,
        }

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        self.last_reject_reason = ""
        warmup = int(self.params.get("warmup_bars", 220) or 220)
        if i < warmup:
            self._reject("warmup")
            return None
        if not self._passes_global_gates(df, i):
            return None

        row = df.iloc[i]
        candidates: List[Dict] = []
        if bool(self.params.get("enable_vcp", True)):
            vcp = self._build_vcp_setup_candidate(row)
            if vcp is not None:
                candidates.append(vcp)
        ep = self._build_ep_candidate(row)
        if ep is not None:
            candidates.append(ep)

        if not candidates:
            self._reject("no_candidate")
            return None

        min_signal_strength = _as_float(self.params.get("min_signal_strength", 88.0), 88.0)
        selected = max(candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
        if _as_float(selected.get("signal_strength"), 0.0) < min_signal_strength:
            self._reject("signal_strength_gate")
            return None
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
        # Engine exit state machine handles practical next-day execution.
        return False

    def pyramid(self, df: pd.DataFrame, i: int, position: Dict) -> Optional[Dict]:
        if position is None:
            return None
        if not bool(self.params.get("pyramid_enabled", True)):
            return None

        entry_price = _as_float(position.get("entry_price"), 0.0)
        if entry_price <= 0:
            return None

        max_adds = int(self.params.get("pyramid_max_adds", 1) or 1)
        pyramids = int(position.get("pyramids", 0) or 0)
        if pyramids >= max_adds:
            return None

        row = df.iloc[i]
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return None

        threshold = _pct(self.params.get("pyramid_threshold", 0.06), 0.06)
        profit_pct = (close_px - entry_price) / entry_price
        if profit_pct < threshold:
            return None

        sma20 = _as_float(row.get("sma20"), 0.0)
        if sma20 > 0 and close_px < sma20:
            return None

        breakout_ref = _first_finite([row.get("highest10_1"), row.get("highest20_1")], default=0.0)
        if breakout_ref > 0 and close_px <= breakout_ref:
            return None

        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 0.0)
        add_vol_mult = _as_float(self.params.get("pyramid_volume_min_mult", 1.1), 1.1)
        if vol_ma50 > 0 and vol < (vol_ma50 * add_vol_mult):
            return None

        add_fraction = _as_float(self.params.get("pyramid_fraction", 0.5), 0.5)
        if add_fraction <= 0:
            return None

        return {
            "add_fraction": add_fraction,
            "stop_to_avg_cost": bool(self.params.get("pyramid_stop_to_avg_cost", False)),
        }
