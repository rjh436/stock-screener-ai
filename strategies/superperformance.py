from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import BaseStrategy


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except Exception:
        return default
    if not math.isfinite(val):
        return default
    return val


def _first_finite(values: Sequence[Any], default: float = float("nan")) -> float:
    for value in values:
        out = _as_float(value, float("nan"))
        if math.isfinite(out):
            return out
    return default


def _as_percent_threshold(value: Any, floor_pct: float) -> float:
    out = _as_float(value, floor_pct)
    if out <= 1.0:
        out *= 100.0
    return max(float(floor_pct), float(out))


@dataclass(frozen=True)
class VCPDetectionResult:
    pivot_price: float
    price_contraction_1: float
    price_contraction_2: float
    volume_contraction_1: float
    volume_contraction_2: float
    breakout_volume_multiple: float


def detect_vcp_breakout(
    df: pd.DataFrame,
    i: int,
    *,
    lookback: int = 80,
    extrema_order: int = 3,
    min_contractions: int = 2,
    breakout_volume_mult: float = 1.5,
    breakout_buffer: float = 0.0,
) -> Optional[VCPDetectionResult]:
    """
    Detect a VCP breakout using local extrema from scipy.signal.argrelextrema.

    Rules:
    - Identify local highs/lows in a rolling window.
    - Require at least two H->L contraction legs.
    - Require contraction_1 > contraction_2 (tightening).
    - Require avg_volume_leg_1 > avg_volume_leg_2 (volume contraction).
    - Trigger only when close breaks above last pivot high on volume >= 1.5x MA50.
    """
    if i <= 0 or lookback < 20:
        return None

    start = max(0, i - lookback + 1)
    window = df.iloc[start : i + 1]
    if len(window) < max(25, (extrema_order * 2) + 8):
        return None

    highs = pd.to_numeric(window.get("high"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    lows = pd.to_numeric(window.get("low"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    closes = pd.to_numeric(window.get("close"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    volumes = pd.to_numeric(window.get("volume"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vol_ma50 = pd.to_numeric(window.get("vol_ma50"), errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if not (np.isfinite(closes[-1]) and closes[-1] > 0):
        return None

    high_idx = argrelextrema(highs, np.greater_equal, order=extrema_order)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=extrema_order)[0]
    if high_idx.size < min_contractions or low_idx.size < min_contractions:
        return None

    # Build pivot stream and compress consecutive same-type pivots.
    pivots: List[Tuple[int, str, float]] = []
    for idx in high_idx.tolist():
        if np.isfinite(highs[idx]):
            pivots.append((idx, "H", float(highs[idx])))
    for idx in low_idx.tolist():
        if np.isfinite(lows[idx]):
            pivots.append((idx, "L", float(lows[idx])))

    pivots.sort(key=lambda x: x[0])
    if len(pivots) < (min_contractions * 2):
        return None

    compact: List[Tuple[int, str, float]] = []
    for pivot in pivots:
        if not compact:
            compact.append(pivot)
            continue
        last = compact[-1]
        if pivot[1] != last[1]:
            compact.append(pivot)
            continue
        # Keep stronger extreme for same-type adjacency.
        if pivot[1] == "H" and pivot[2] >= last[2]:
            compact[-1] = pivot
        elif pivot[1] == "L" and pivot[2] <= last[2]:
            compact[-1] = pivot

    # Extract H->L contraction legs.
    # Each leg stores:
    # (price_contraction_pct, avg_volume, hi_idx, lo_idx, hi_price)
    legs: List[Tuple[float, float, int, int, float]] = []
    for left, right in zip(compact, compact[1:]):
        if left[1] != "H" or right[1] != "L":
            continue
        hi_idx0, _, hi_px = left
        lo_idx0, _, lo_px = right
        if not (hi_px > 0 and lo_px > 0 and lo_px < hi_px):
            continue
        contraction = ((hi_px - lo_px) / hi_px) * 100.0
        if contraction <= 0 or not math.isfinite(contraction):
            continue
        seg = volumes[hi_idx0 : lo_idx0 + 1]
        avg_vol = float(np.nanmean(seg)) if seg.size > 0 else float("nan")
        legs.append((contraction, avg_vol, int(hi_idx0), int(lo_idx0), float(hi_px)))

    if len(legs) < min_contractions:
        return None

    c1, v1, _, _, _ = legs[-2]
    c2, v2, hi_idx_last, lo_idx_last, hi_px_last = legs[-1]

    # Tightening contraction profile (C1 > C2).
    if not (c1 > c2 > 0):
        return None

    # Volume contraction profile (V1 > V2).
    if math.isfinite(v1) and math.isfinite(v2) and not (v1 > (v2 * 0.95)):
        return None

    # Use the high that started the final contraction as the breakout pivot.
    # This is less brittle than selecting the last generic local high.
    pivot_price = float(hi_px_last)
    if not (math.isfinite(pivot_price) and pivot_price > 0):
        return None

    # Ensure final contraction has completed and current bar is after the contraction low.
    if (len(window) - 1) <= lo_idx_last:
        return None

    breakout_level = pivot_price * (1.0 + max(0.0, breakout_buffer))
    if closes[-1] <= breakout_level:
        return None

    vol_now = float(volumes[-1]) if np.isfinite(volumes[-1]) else 0.0
    ma50_now = float(vol_ma50[-1]) if np.isfinite(vol_ma50[-1]) else 0.0
    volume_multiple = (vol_now / ma50_now) if ma50_now > 0 else 0.0
    if ma50_now > 0 and volume_multiple < breakout_volume_mult:
        return None

    return VCPDetectionResult(
        pivot_price=pivot_price,
        price_contraction_1=float(c1),
        price_contraction_2=float(c2),
        volume_contraction_1=float(v1) if math.isfinite(v1) else float("nan"),
        volume_contraction_2=float(v2) if math.isfinite(v2) else float("nan"),
        breakout_volume_multiple=volume_multiple,
    )


class SuperperformanceStrategy(BaseStrategy):
    """
    Gate + Archetype Superperformance model.

    Gate (must pass):
    - Trend template: close > sma10 > sma20 > sma50 > sma150 > sma200
    - RS percentile gate
    - Fundamental gate with High-Tight-Flag override for missing fundamentals

    Archetypes (union logic):
    - A: VCP breakout
    - B: Episodic Pivot (EP)
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = self.params.get("name", "Superperformance")
        self._last_reject_reason = ""
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    @property
    def last_reject_reason(self) -> str:
        return self._last_reject_reason

    def _reject(self, reason: str) -> Optional[Dict]:
        self._last_reject_reason = reason
        return None

    def _resolve_rs_percentile(self, row: pd.Series) -> float:
        candidates = [
            row.get("rs_percentile"),
            row.get("relative_strength_percentile"),
            row.get("momentum_rank"),
            row.get("rs_rating"),
        ]
        for value in candidates:
            out = _as_float(value, float("nan"))
            if math.isfinite(out):
                return out
        # Fail-open when no cross-sectional RS field exists.
        return 100.0

    def _resolve_price_action_percentile(self, row: pd.Series, rs_percentile: float) -> float:
        # If explicit price-action percentile exists, prefer it.
        explicit = _first_finite(
            [
                row.get("price_action_percentile"),
                row.get("pa_percentile"),
                row.get("breakout_percentile"),
            ],
            default=float("nan"),
        )
        if math.isfinite(explicit):
            return explicit

        # Fallback to cross-sectional momentum proxies.
        return max(
            rs_percentile,
            _first_finite(
                [
                    row.get("momentum_rank"),
                    row.get("rs_rating"),
                ],
                default=0.0,
            ),
        )

    def _ep_candidate(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 1:
            return None

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        prev_close = _as_float(prev_row.get("close"), 0.0)
        open_px = _as_float(row.get("open"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        close_px = _as_float(row.get("close"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 0.0)

        if prev_close <= 0 or open_px <= 0 or high_px <= 0 or low_px <= 0 or close_px <= 0:
            return None

        gap_pct = ((open_px - prev_close) / prev_close) * 100.0
        ep_gap_min = _as_percent_threshold(self.params.get("ep_gap_pct", 8.0), 8.0)
        ep_vol_mult = max(3.0, float(self.params.get("ep_vol_mult", 3.0) or 3.0))
        close_near_high_min = float(self.params.get("ep_close_near_high_min", 0.80) or 0.80)

        day_range = high_px - low_px
        clv = _as_float(row.get("clv"), float("nan"))
        if not math.isfinite(clv):
            clv = ((close_px - low_px) / day_range) if day_range > 0 else 0.0

        vol_multiple = (vol / vol_ma50) if vol_ma50 > 0 else 0.0
        if (
            gap_pct < ep_gap_min
            or vol_ma50 <= 0
            or vol_multiple < ep_vol_mult
            or clv < close_near_high_min
        ):
            return None

        ep_entry_mode = str(self.params.get("ep_entry_mode", "close") or "close").lower()
        trigger_px = open_px if ep_entry_mode == "open" else close_px
        stop_px = low_px

        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return None

        ep_max_stop_pct = float(self.params.get("ep_max_stop_pct", 0.15) or 0.15)
        stop_width = (trigger_px - stop_px) / trigger_px
        if stop_width > ep_max_stop_pct:
            return None

        return {
            "trigger_price": trigger_px,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
            "entry_type": "ep",
            "max_stop_pct": ep_max_stop_pct,
            "entry_timing": "same_day_open" if ep_entry_mode == "open" else "same_day_close",
            "signal_mode": "open" if ep_entry_mode == "open" else "close",
            "signal_strength": float(vol_multiple + (gap_pct / 10.0)),
        }

    def _ep_failure_reason(self, df: pd.DataFrame, i: int) -> str:
        if i < 1:
            return "ep:no_prev_bar"

        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev_close = _as_float(prev_row.get("close"), 0.0)
        open_px = _as_float(row.get("open"), 0.0)
        high_px = _as_float(row.get("high"), 0.0)
        low_px = _as_float(row.get("low"), 0.0)
        close_px = _as_float(row.get("close"), 0.0)
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), 0.0)
        if prev_close <= 0 or open_px <= 0 or high_px <= 0 or low_px <= 0 or close_px <= 0:
            return "ep:invalid_ohlc"

        gap_pct = ((open_px - prev_close) / prev_close) * 100.0
        ep_gap_min = _as_percent_threshold(self.params.get("ep_gap_pct", 8.0), 8.0)
        ep_vol_mult = max(3.0, float(self.params.get("ep_vol_mult", 3.0) or 3.0))
        close_near_high_min = float(self.params.get("ep_close_near_high_min", 0.80) or 0.80)

        day_range = high_px - low_px
        clv = _as_float(row.get("clv"), float("nan"))
        if not math.isfinite(clv):
            clv = ((close_px - low_px) / day_range) if day_range > 0 else 0.0
        vol_multiple = (vol / vol_ma50) if vol_ma50 > 0 else 0.0

        if gap_pct < ep_gap_min:
            return f"ep:gap_pct={gap_pct:.2f}<{ep_gap_min:.2f}"
        if vol_ma50 <= 0:
            return "ep:vol_ma50<=0"
        if vol_multiple < ep_vol_mult:
            return f"ep:vol_mult={vol_multiple:.2f}<{ep_vol_mult:.2f}"
        if clv < close_near_high_min:
            return f"ep:close_near_high={clv:.2f}<{close_near_high_min:.2f}"

        ep_entry_mode = str(self.params.get("ep_entry_mode", "close") or "close").lower()
        trigger_px = open_px if ep_entry_mode == "open" else close_px
        stop_px = low_px
        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return "ep:invalid_stop"
        ep_max_stop_pct = float(self.params.get("ep_max_stop_pct", 0.15) or 0.15)
        stop_width = (trigger_px - stop_px) / trigger_px
        if stop_width > ep_max_stop_pct:
            return f"ep:stop_width={stop_width:.3f}>{ep_max_stop_pct:.3f}"
        return "ep:unknown"

    def _vcp_candidate(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        row = df.iloc[i]

        lookback = int(self.params.get("vcp_lookback_bars", 80) or 80)
        extrema_order = int(self.params.get("vcp_extrema_order", 3) or 3)
        vol_mult = float(self.params.get("vcp_breakout_volume_mult", 1.5) or 1.5)
        breakout_buffer = float(self.params.get("breakout_buffer", 0.001) or 0.001)

        vcp = detect_vcp_breakout(
            df,
            i,
            lookback=lookback,
            extrema_order=max(1, extrema_order),
            min_contractions=2,
            breakout_volume_mult=max(1.5, vol_mult),
            breakout_buffer=max(0.0, breakout_buffer),
        )
        if vcp is None:
            return None

        trigger_px = float(vcp.pivot_price * (1.0 + max(0.0, breakout_buffer)))
        low_px = _as_float(row.get("low"), 0.0)
        max_stop_pct = float(self.params.get("max_stop_pct", 0.06) or 0.06)
        stop_px = max(low_px, trigger_px * (1.0 - max_stop_pct))

        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return None

        strength = float(
            vcp.breakout_volume_multiple
            + (vcp.price_contraction_1 / 10.0)
            - (vcp.price_contraction_2 / 10.0)
        )

        return {
            "trigger_price": trigger_px,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
            "entry_type": "vcp",
            "max_stop_pct": max_stop_pct,
            "entry_timing": "next_day",
            "signal_strength": strength,
        }

    def check_setup(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        self._last_reject_reason = ""

        warmup = int(self.params.get("warmup_bars", 200) or 200)
        if i < warmup:
            return self._reject("warmup")

        row = df.iloc[i]
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return self._reject("close_invalid")

        # Gate 1: Strict trend template hierarchy.
        sma10 = _as_float(row.get("sma10"), 0.0)
        sma20 = _as_float(row.get("sma20"), 0.0)
        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)

        if not (close_px > sma10 > sma20 > sma50 > sma150 > sma200):
            return self._reject(
                f"trend_gate close={close_px:.2f} sma10={sma10:.2f} sma20={sma20:.2f} "
                f"sma50={sma50:.2f} sma150={sma150:.2f} sma200={sma200:.2f}"
            )

        min_price = float(self.params.get("min_price", 2.0) or 2.0)
        if close_px < min_price:
            return self._reject(f"min_price close={close_px:.2f} < {min_price:.2f}")

        min_avg_volume = float(self.params.get("min_avg_volume_30", 0.0) or 0.0)
        vol_ma30 = _as_float(row.get("vol_ma30"), float("nan"))
        if min_avg_volume > 0 and math.isfinite(vol_ma30) and vol_ma30 < min_avg_volume:
            return self._reject(f"liquidity vol_ma30={vol_ma30:.0f} < {min_avg_volume:.0f}")

        # Gate 2: Relative strength percentile gate.
        rs_percentile = self._resolve_rs_percentile(row)
        rs_gate_min = _as_percent_threshold(
            self.params.get(
                "rs_percentile_min",
                self.params.get("rs_gate_min", 85.0),
            ),
            85.0,
        )
        if rs_percentile < rs_gate_min:
            return self._reject(f"rs_gate rs_percentile={rs_percentile:.2f} < {rs_gate_min:.2f}")

        # Gate 3: Fundamental growth gate with IPO/HTF override.
        eps_yoy = _first_finite(
            [
                row.get("eps_growth_yoy"),
                row.get("eps_yoy_growth_pct"),
                row.get("eps_yoy"),
            ],
            default=float("nan"),
        )
        sales_yoy = _first_finite(
            [
                row.get("sales_growth_yoy"),
                row.get("revenue_yoy_growth_pct"),
                row.get("sales_yoy"),
            ],
            default=float("nan"),
        )

        growth_min = _as_percent_threshold(self.params.get("fundamental_growth_min_pct", 20.0), 20.0)
        eps_available = math.isfinite(eps_yoy)
        sales_available = math.isfinite(sales_yoy)

        if eps_available or sales_available:
            eps_ok = eps_available and eps_yoy >= growth_min
            sales_ok = sales_available and sales_yoy >= growth_min
            if not (eps_ok or sales_ok):
                return self._reject(
                    f"fundamental_gate eps_yoy={eps_yoy:.2f} sales_yoy={sales_yoy:.2f} < {growth_min:.2f}"
                )
        else:
            price_action_pct = self._resolve_price_action_percentile(row, rs_percentile)
            htf_override = _as_percent_threshold(self.params.get("high_tight_flag_override_pct", 95.0), 95.0)
            if price_action_pct < htf_override:
                return self._reject(
                    f"fundamental_missing_override price_action_pct={price_action_pct:.2f} < {htf_override:.2f}"
                )

        # Archetype union logic: any trigger can produce an entry.
        entry_mode = str(self.params.get("entry_mode", "both") or "both").lower()
        allow_vcp = entry_mode in {"both", "breakout", "vcp"}
        allow_ep = entry_mode in {"both", "ep", "episodic_pivot"}

        candidates: List[Dict] = []
        vcp_candidate: Optional[Dict] = None
        ep_candidate: Optional[Dict] = None
        ep_failure = ""
        if allow_vcp:
            vcp_candidate = self._vcp_candidate(df, i)
            if vcp_candidate is not None:
                candidates.append(vcp_candidate)

        if allow_ep:
            ep_candidate = self._ep_candidate(df, i)
            if ep_candidate is not None:
                candidates.append(ep_candidate)
            else:
                ep_failure = self._ep_failure_reason(df, i)

        if not candidates:
            details: List[str] = []
            if allow_vcp and vcp_candidate is None:
                details.append("vcp:no_breakout")
            if allow_ep and ep_candidate is None:
                details.append(ep_failure or "ep:no_signal")
            if details:
                return self._reject("archetype_none " + " | ".join(details))
            return self._reject("archetype_none")

        selected = max(candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
        selected.pop("signal_strength", None)
        return selected

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        return self.check_setup(df, i)

    def exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_i: int,
        entry_price: float,
        stop_price: float,
    ) -> bool:
        # Engine uses centralized exit logic.
        return False

    def pyramid(self, df: pd.DataFrame, i: int, position: Dict) -> Optional[Dict]:
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

        rs_percentile = _first_finite(
            [
                df.iloc[i].get("rs_percentile"),
                df.iloc[i].get("momentum_rank"),
                df.iloc[i].get("rs_rating"),
            ],
            default=0.0,
        )
        if rs_percentile < 85.0:
            return None

        add_fraction = float(self.params.get("pyramid_fraction", 0.5) or 0.5)
        if add_fraction <= 0:
            return None

        return {
            "add_fraction": add_fraction,
            "stop_to_avg_cost": bool(self.params.get("pyramid_stop_to_avg_cost", True)),
        }
