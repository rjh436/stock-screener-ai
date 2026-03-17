from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from .base import BaseStrategy
from .superperformance_breakout import select_breakout_candidate
from .superperformance_continuation import evaluate_continuation
from .superperformance_continuation_breakout import evaluate_continuation_breakout
from .superperformance_recovery import evaluate_recovery

_STOP_WIDTH_TOL = 1e-4
_DEFAULT_VCP_DAMPING_RATIO = 0.75
_DEFAULT_VCP_VOLUME_DRYUP_MULT = 1.20
_DEFAULT_VCP_BREAKOUT_VOL_FLOOR = 2.0
_DEFAULT_ADR_MIN_PCT = 3.5
_DEFAULT_PRIOR_RUNUP_MIN_PCT = 30.0


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
    damping_ratio: float = _DEFAULT_VCP_DAMPING_RATIO,
    volume_dryup_mult: float = _DEFAULT_VCP_VOLUME_DRYUP_MULT,
    breakout_buffer: float = 0.0,
    require_breakout_close: bool = True,
    require_breakout_volume: bool = True,
    return_reason: bool = False,
) -> Union[Optional[VCPDetectionResult], Tuple[Optional[VCPDetectionResult], str]]:
    """
    Detect a VCP breakout using local extrema from scipy.signal.argrelextrema.

    Rules:
    - Identify local highs/lows in a rolling window.
    - Require at least two H->L contraction legs.
    - Require contraction_1 > contraction_2 (tightening).
    - Require avg_volume_leg_1 > avg_volume_leg_2 (volume contraction).
    - Trigger only when close breaks above last pivot high on volume >= 1.5x MA50.
    """
    def _pack(
        result: Optional[VCPDetectionResult],
        reason: str,
    ) -> Union[Optional[VCPDetectionResult], Tuple[Optional[VCPDetectionResult], str]]:
        if return_reason:
            return result, reason
        return result

    if i <= 0 or lookback < 20:
        return _pack(None, "invalid_window")

    start = max(0, i - lookback + 1)
    window = df.iloc[start : i + 1]
    if len(window) < max(25, (extrema_order * 2) + 8):
        return _pack(None, "insufficient_bars")

    highs = pd.to_numeric(window.get("high"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    lows = pd.to_numeric(window.get("low"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    closes = pd.to_numeric(window.get("close"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    volumes = pd.to_numeric(window.get("volume"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vol_ma50 = pd.to_numeric(window.get("vol_ma50"), errors="coerce").to_numpy(dtype=np.float64, copy=False)

    if not (np.isfinite(closes[-1]) and closes[-1] > 0):
        return _pack(None, "invalid_close")

    high_idx = argrelextrema(highs, np.greater_equal, order=extrema_order)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=extrema_order)[0]
    if high_idx.size < min_contractions or low_idx.size < min_contractions:
        return _pack(None, "extrema_insufficient")

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
        return _pack(None, "pivots_insufficient")

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
        return _pack(None, "legs_insufficient")

    if len(legs) >= 2:
        c1, v1, _, _, _ = legs[-2]
        c2, v2, hi_idx_last, lo_idx_last, hi_px_last = legs[-1]

        # Tightening contraction profile (C1 > C2) with damping ratio.
        if not (c1 > c2 > 0):
            return _pack(None, "no_tightening")
        damping_ratio = float(np.clip(_as_float(damping_ratio, _DEFAULT_VCP_DAMPING_RATIO), 0.0, 1.0))
        tight_ratio = (c2 / c1) if c1 > 0 else float("inf")
        if damping_ratio > 0 and tight_ratio > damping_ratio:
            return _pack(None, "no_tightening")

        # Volume contraction profile: prior leg volume must exceed current leg volume.
        volume_dryup_mult = max(1.0, _as_float(volume_dryup_mult, _DEFAULT_VCP_VOLUME_DRYUP_MULT))
        if math.isfinite(v1) and math.isfinite(v2) and not (v1 >= (v2 * volume_dryup_mult)):
            return _pack(None, "no_volume_contraction")
    else:
        # One-leg fallback for elite-RS overrides.
        c2, v2, hi_idx_last, lo_idx_last, hi_px_last = legs[-1]
        if not (c2 > 0):
            return _pack(None, "invalid_contraction")
        c1 = c2 * 1.05
        v1 = float("nan")

    # Use the high that started the final contraction as the breakout pivot.
    # This is less brittle than selecting the last generic local high.
    pivot_price = float(hi_px_last)
    if not (math.isfinite(pivot_price) and pivot_price > 0):
        return _pack(None, "invalid_pivot")

    # Ensure final contraction has completed and current bar is after the contraction low.
    if (len(window) - 1) <= lo_idx_last:
        return _pack(None, "contraction_incomplete")

    breakout_level = pivot_price * (1.0 + max(0.0, breakout_buffer))
    if require_breakout_close and closes[-1] <= breakout_level:
        return _pack(None, "breakout_not_triggered")

    vol_now = float(volumes[-1]) if np.isfinite(volumes[-1]) else 0.0
    ma50_now = float(vol_ma50[-1]) if np.isfinite(vol_ma50[-1]) else 0.0
    volume_multiple = (vol_now / ma50_now) if ma50_now > 0 else 0.0
    if require_breakout_volume and ma50_now > 0 and volume_multiple < breakout_volume_mult:
        return _pack(None, "breakout_volume_insufficient")

    return _pack(
        VCPDetectionResult(
            pivot_price=pivot_price,
            price_contraction_1=float(c1),
            price_contraction_2=float(c2),
            volume_contraction_1=float(v1) if math.isfinite(v1) else float("nan"),
            volume_contraction_2=float(v2) if math.isfinite(v2) else float("nan"),
            breakout_volume_multiple=volume_multiple,
        ),
        "ok",
    )


def detect_micro_vcp_breakout(
    df: pd.DataFrame,
    i: int,
    *,
    lookback: int = 12,
    micro_window: int = 5,
    max_contraction_pct: float = 6.0,
    breakout_volume_mult: float = 1.5,
    breakout_buffer: float = 0.0,
    max_bb_width: float = 0.12,
    require_breakout_close: bool = True,
    require_breakout_volume: bool = True,
    return_reason: bool = False,
) -> Union[Optional[VCPDetectionResult], Tuple[Optional[VCPDetectionResult], str]]:
    """
    Detect tight 3-5 day "T-pivot" style breakouts missed by extrema-based VCP.
    """

    def _pack(
        result: Optional[VCPDetectionResult],
        reason: str,
    ) -> Union[Optional[VCPDetectionResult], Tuple[Optional[VCPDetectionResult], str]]:
        if return_reason:
            return result, reason
        return result

    if i <= 0:
        return _pack(None, "invalid_index")
    micro_window = max(3, int(micro_window))
    lookback = max(micro_window + 2, int(lookback))

    start = max(0, i - lookback + 1)
    window = df.iloc[start : i + 1]
    if len(window) < (micro_window + 2):
        return _pack(None, "insufficient_bars")

    def _col_to_np(name: str) -> np.ndarray:
        if name in window.columns:
            col = window[name]
        else:
            col = pd.Series(np.nan, index=window.index)
        return pd.to_numeric(col, errors="coerce").to_numpy(dtype=np.float64, copy=False)

    highs = _col_to_np("high")
    lows = _col_to_np("low")
    closes = _col_to_np("close")
    volumes = _col_to_np("volume")
    vol_ma50 = _col_to_np("vol_ma50")
    bb_width = _col_to_np("bb_width")

    if not (np.isfinite(closes[-1]) and closes[-1] > 0):
        return _pack(None, "invalid_close")

    seg_high = highs[-(micro_window + 1) : -1]
    seg_low = lows[-(micro_window + 1) : -1]
    if seg_high.size < micro_window or seg_low.size < micro_window:
        return _pack(None, "segment_insufficient")
    if not np.isfinite(seg_high).any() or not np.isfinite(seg_low).any():
        return _pack(None, "segment_invalid")

    pivot_price = float(np.nanmax(seg_high))
    trough_price = float(np.nanmin(seg_low))
    if not (math.isfinite(pivot_price) and math.isfinite(trough_price) and pivot_price > 0 and trough_price > 0):
        return _pack(None, "invalid_pivot")
    if trough_price >= pivot_price:
        return _pack(None, "invalid_contraction")

    contraction_pct = ((pivot_price - trough_price) / pivot_price) * 100.0
    if not math.isfinite(contraction_pct) or contraction_pct <= 0:
        return _pack(None, "invalid_contraction")
    if contraction_pct > max_contraction_pct:
        return _pack(None, "micro_contraction_too_wide")

    if math.isfinite(max_bb_width) and max_bb_width > 0:
        bb_seg = bb_width[-(micro_window + 1) : -1]
        if np.isfinite(bb_seg).any():
            bb_last = float(np.nanmax(bb_seg))
            if math.isfinite(bb_last) and bb_last > max_bb_width:
                return _pack(None, "bb_width_not_tight")

    breakout_level = pivot_price * (1.0 + max(0.0, breakout_buffer))
    if require_breakout_close and closes[-1] <= breakout_level:
        return _pack(None, "breakout_not_triggered")

    vol_now = float(volumes[-1]) if np.isfinite(volumes[-1]) else 0.0
    ma50_now = float(vol_ma50[-1]) if np.isfinite(vol_ma50[-1]) else 0.0
    volume_multiple = (vol_now / ma50_now) if ma50_now > 0 else 0.0
    if require_breakout_volume and ma50_now > 0 and volume_multiple < breakout_volume_mult:
        return _pack(None, "breakout_volume_insufficient")

    first_half = volumes[-(micro_window + 1) : -(micro_window // 2 + 1)]
    second_half = volumes[-(micro_window // 2 + 1) : -1]
    v1 = float(np.nanmean(first_half)) if first_half.size else float("nan")
    v2 = float(np.nanmean(second_half)) if second_half.size else float("nan")
    c1 = min(max_contraction_pct * 1.2, contraction_pct * 1.4)
    c2 = contraction_pct

    return _pack(
        VCPDetectionResult(
            pivot_price=pivot_price,
            price_contraction_1=float(max(c1, c2 * 1.01)),
            price_contraction_2=float(c2),
            volume_contraction_1=float(v1) if math.isfinite(v1) else float("nan"),
            volume_contraction_2=float(v2) if math.isfinite(v2) else float("nan"),
            breakout_volume_multiple=volume_multiple,
        ),
        "ok_micro",
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
        self._last_vcp_failure_reason = ""
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
        # Fail-closed by default: missing RS should not pass elite gates.
        if bool(self.params.get("rs_fail_open", False)):
            return 100.0
        return 0.0

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

    def _is_adaptive_market_green(self, row: pd.Series) -> bool:
        if not bool(self.params.get("adaptive_breakout_gates_enabled", False)):
            return False
        spy_close = _first_finite(
            [
                row.get("spy_close"),
                row.get("spyclose"),
            ],
            default=float("nan"),
        )
        spy_sma200 = _first_finite(
            [
                row.get("spy_sma200"),
                row.get("spysma200"),
            ],
            default=float("nan"),
        )
        return bool(
            math.isfinite(spy_close)
            and math.isfinite(spy_sma200)
            and spy_sma200 > 0
            and spy_close > spy_sma200
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
        ep_gap_min = _as_percent_threshold(self.params.get("ep_gap_pct", 8.0), 0.0)
        ep_vol_mult = max(3.0, float(self.params.get("ep_vol_mult", 3.0) or 3.0))
        close_near_high_min = float(self.params.get("ep_close_near_high_min", 0.80) or 0.80)
        ep_entry_mode = str(self.params.get("ep_entry_mode", "close") or "close").lower()
        ep_max_stop_pct = float(self.params.get("ep_max_stop_pct", 0.15) or 0.15)

        vol_multiple = 0.0
        if ep_entry_mode == "open":
            # Open-mode EP must use only information available before/at the open.
            if gap_pct < ep_gap_min:
                return None
            trigger_px = open_px
            stop_px = trigger_px * (1.0 - ep_max_stop_pct)
        else:
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
            trigger_px = close_px
            stop_px = low_px

        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return None

        stop_width = (trigger_px - stop_px) / trigger_px
        if (stop_width - ep_max_stop_pct) > _STOP_WIDTH_TOL:
            return None

        force_next_day = bool(self.params.get("ep_force_next_day", False))
        if force_next_day:
            # Enforce next-session execution before assigning any same-day mode.
            entry_timing = "next_day"
            signal_mode = "after_close"
        elif ep_entry_mode == "open":
            entry_timing = "next_day"
            signal_mode = "after_close"
        else:
            entry_timing = "same_day_close"
            signal_mode = "close"

        return {
            "trigger_price": trigger_px,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
            "entry_type": "ep",
            "max_stop_pct": ep_max_stop_pct,
            "entry_timing": entry_timing,
            "signal_mode": signal_mode,
            "signal_strength": float((gap_pct / 10.0) if ep_entry_mode == "open" else (vol_multiple + (gap_pct / 10.0))),
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
        ep_gap_min = _as_percent_threshold(self.params.get("ep_gap_pct", 8.0), 0.0)
        ep_vol_mult = max(3.0, float(self.params.get("ep_vol_mult", 3.0) or 3.0))
        close_near_high_min = float(self.params.get("ep_close_near_high_min", 0.80) or 0.80)
        ep_entry_mode = str(self.params.get("ep_entry_mode", "close") or "close").lower()
        ep_max_stop_pct = float(self.params.get("ep_max_stop_pct", 0.15) or 0.15)

        if gap_pct < ep_gap_min:
            return f"ep:gap_pct={gap_pct:.2f}<{ep_gap_min:.2f}"

        if ep_entry_mode == "open":
            trigger_px = open_px
            stop_px = trigger_px * (1.0 - ep_max_stop_pct)
        else:
            day_range = high_px - low_px
            clv = _as_float(row.get("clv"), float("nan"))
            if not math.isfinite(clv):
                clv = ((close_px - low_px) / day_range) if day_range > 0 else 0.0
            vol_multiple = (vol / vol_ma50) if vol_ma50 > 0 else 0.0

            if vol_ma50 <= 0:
                return "ep:vol_ma50<=0"
            if vol_multiple < ep_vol_mult:
                return f"ep:vol_mult={vol_multiple:.2f}<{ep_vol_mult:.2f}"
            if clv < close_near_high_min:
                return f"ep:close_near_high={clv:.2f}<{close_near_high_min:.2f}"
            trigger_px = close_px
            stop_px = low_px

        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return "ep:invalid_stop"
        stop_width = (trigger_px - stop_px) / trigger_px
        if (stop_width - ep_max_stop_pct) > _STOP_WIDTH_TOL:
            return f"ep:stop_width={stop_width:.3f}>{ep_max_stop_pct:.3f}"
        return "ep:unknown"

    def _vcp_candidate(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        row = df.iloc[i]
        self._last_vcp_failure_reason = ""

        lookback = int(self.params.get("vcp_lookback_bars", 80) or 80)
        extrema_order = int(self.params.get("vcp_extrema_order", 3) or 3)
        vol_mult = float(self.params.get("vcp_breakout_volume_mult", _DEFAULT_VCP_BREAKOUT_VOL_FLOOR) or _DEFAULT_VCP_BREAKOUT_VOL_FLOOR)
        vcp_damping_ratio = float(self.params.get("vcp_damping_ratio", _DEFAULT_VCP_DAMPING_RATIO) or _DEFAULT_VCP_DAMPING_RATIO)
        vcp_volume_dryup_mult = float(
            self.params.get("vcp_volume_dryup_mult", _DEFAULT_VCP_VOLUME_DRYUP_MULT)
            or _DEFAULT_VCP_VOLUME_DRYUP_MULT
        )
        breakout_buffer = float(self.params.get("breakout_buffer", 0.001) or 0.001)
        breakout_vol_req = max(_DEFAULT_VCP_BREAKOUT_VOL_FLOOR, vol_mult)
        vcp_entry_mode = str(self.params.get("vcp_entry_mode", "next_day") or "next_day").lower()
        vcp_trigger_mode = str(self.params.get("vcp_trigger_mode", "close_confirmed") or "close_confirmed").lower()
        use_setup_trigger = vcp_trigger_mode in {"setup", "prebreakout", "next_day_setup"}
        # Default remains close-confirmed breakout for selectivity. Setup-trigger mode
        # is available explicitly and applies stricter pre-breakout proximity checks.
        require_breakout_close = not use_setup_trigger
        require_breakout_volume = not use_setup_trigger

        vcp, vcp_reason = detect_vcp_breakout(
            df,
            i,
            lookback=lookback,
            extrema_order=max(1, extrema_order),
            min_contractions=2,
            breakout_volume_mult=breakout_vol_req,
            damping_ratio=vcp_damping_ratio,
            volume_dryup_mult=vcp_volume_dryup_mult,
            breakout_buffer=max(0.0, breakout_buffer),
            require_breakout_close=require_breakout_close,
            require_breakout_volume=require_breakout_volume,
            return_reason=True,
        )
        used_elite_override = False
        if vcp is None:
            micro_enabled = bool(self.params.get("vcp_micro_enabled", True))
            if micro_enabled:
                micro_window = int(self.params.get("vcp_micro_window", 5) or 5)
                micro_max_contraction = float(self.params.get("vcp_micro_max_contraction_pct", 6.0) or 6.0)
                micro_bb_width = float(self.params.get("vcp_micro_max_bb_width", 0.12) or 0.12)
                vcp, micro_reason = detect_micro_vcp_breakout(
                    df,
                    i,
                    lookback=max(lookback // 2, micro_window + 2),
                    micro_window=max(3, micro_window),
                    max_contraction_pct=max(1.0, micro_max_contraction),
                    breakout_volume_mult=max(1.25, breakout_vol_req * 0.85),
                    breakout_buffer=max(0.0, breakout_buffer),
                    max_bb_width=max(0.0, micro_bb_width),
                    require_breakout_close=require_breakout_close,
                    require_breakout_volume=require_breakout_volume,
                    return_reason=True,
                )
                if vcp is not None:
                    vcp_reason = str(micro_reason or "ok_micro")

        if vcp is None:
            enable_elite_override = bool(self.params.get("vcp_elite_override_enabled", True))
            if not enable_elite_override:
                self._last_vcp_failure_reason = str(vcp_reason or "no_breakout")
                return None

            rs_percentile = self._resolve_rs_percentile(row)
            elite_rs_min = _as_percent_threshold(self.params.get("vcp_elite_rs_override_min", 95.0), 0.0)
            close_px = _as_float(row.get("close"), 0.0)
            high_52w = _first_finite(
                [
                    row.get("high_52w"),
                    row.get("high52w"),
                ],
                default=float("nan"),
            )
            near_high_52w = bool(math.isfinite(high_52w) and high_52w > 0 and close_px >= (high_52w * 0.90))

            if near_high_52w and rs_percentile >= elite_rs_min:
                used_elite_override = True
                vcp, vcp_reason = detect_vcp_breakout(
                    df,
                    i,
                    lookback=lookback,
                    extrema_order=max(1, extrema_order - 1),
                    min_contractions=1,
                    breakout_volume_mult=max(1.25, breakout_vol_req * 0.85),
                    damping_ratio=vcp_damping_ratio,
                    volume_dryup_mult=vcp_volume_dryup_mult,
                    breakout_buffer=max(0.0, breakout_buffer),
                    require_breakout_close=require_breakout_close,
                    require_breakout_volume=require_breakout_volume,
                    return_reason=True,
                )

            if vcp is None:
                self._last_vcp_failure_reason = str(vcp_reason or "no_breakout")
                return None

        trigger_px = float(vcp.pivot_price * (1.0 + max(0.0, breakout_buffer)))
        if use_setup_trigger:
            close_px_now = _as_float(row.get("close"), 0.0)
            high_px_now = _as_float(row.get("high"), 0.0)
            vol_now = _as_float(row.get("volume"), 0.0)
            vol_ma50_now = _as_float(row.get("vol_ma50"), float("nan"))
            setup_proximity_pct = max(0.0, float(self.params.get("vcp_setup_proximity_pct", 0.02) or 0.02))
            setup_high_prox_pct = max(0.0, float(self.params.get("vcp_setup_high_proximity_pct", 0.01) or 0.01))
            setup_vol_max_mult = max(0.1, float(self.params.get("vcp_setup_volume_max_mult", 1.30) or 1.30))

            if close_px_now > (trigger_px * (1.0 + _STOP_WIDTH_TOL)):
                self._last_vcp_failure_reason = "setup_already_broken"
                return None
            if close_px_now < (trigger_px * (1.0 - setup_proximity_pct)):
                self._last_vcp_failure_reason = "setup_too_far_below_pivot"
                return None
            if high_px_now < (trigger_px * (1.0 - setup_high_prox_pct)):
                self._last_vcp_failure_reason = "setup_high_not_near_pivot"
                return None
            if math.isfinite(vol_ma50_now) and vol_ma50_now > 0 and vol_now > (vol_ma50_now * setup_vol_max_mult):
                self._last_vcp_failure_reason = "setup_volume_not_dry"
                return None

        close_px = _as_float(row.get("close"), 0.0)
        vcp_close_extension_max_pct = float(
            self.params.get("vcp_close_extension_max_pct", 0.0) or 0.0
        )
        if (not use_setup_trigger) and trigger_px > 0.0 and vcp_close_extension_max_pct > 0.0:
            breakout_ext_pct = (close_px - trigger_px) / trigger_px
            if breakout_ext_pct > vcp_close_extension_max_pct:
                self._last_vcp_failure_reason = "breakout_close_too_extended"
                return None

        low_px = _as_float(row.get("low"), 0.0)
        max_stop_pct = float(self.params.get("max_stop_pct", 0.06) or 0.06)
        stop_floor = trigger_px * (1.0 - max_stop_pct)
        # Breakout bars can gap and hold above pivot, making day low > trigger.
        # In that case, use bounded %-risk stop rather than invalid stop>entry.
        if low_px > 0.0 and low_px < trigger_px:
            stop_px = max(low_px, stop_floor)
        else:
            stop_px = stop_floor

        if trigger_px <= 0 or stop_px <= 0 or stop_px >= trigger_px:
            return None

        strength = float(
            vcp.breakout_volume_multiple
            + (vcp.price_contraction_1 / 10.0)
            - (vcp.price_contraction_2 / 10.0)
        )
        if used_elite_override:
            strength += 0.15

        entry_timing = "next_day"
        signal_mode = "after_close"
        natr = _as_float(row.get("natr"), float("nan"))
        if vcp_entry_mode in {"same_day", "same_day_close", "close"}:
            entry_timing = "same_day_close"
            signal_mode = "close"
        elif vcp_entry_mode in {"same_day_open", "open"}:
            # Same-day open entries are not causally valid for close-confirmed VCP logic.
            entry_timing = "next_day"
            signal_mode = "after_close"
        elif vcp_entry_mode in {"adaptive", "auto"}:
            breakout_ext_pct = ((close_px - trigger_px) / trigger_px) * 100.0 if trigger_px > 0 else 0.0
            same_day_vol_mult = float(self.params.get("vcp_same_day_vol_mult", 2.5) or 2.5)
            same_day_ext_pct = float(self.params.get("vcp_same_day_extension_pct", 2.0) or 2.0)
            same_day_natr_max = float(self.params.get("vcp_same_day_natr_max_pct", 7.0) or 7.0)
            natr_ok = (not math.isfinite(natr)) or (natr <= same_day_natr_max)
            if (
                vcp.breakout_volume_multiple >= same_day_vol_mult
                and breakout_ext_pct >= same_day_ext_pct
                and natr_ok
            ):
                entry_timing = "same_day_close"
                signal_mode = "close"

        return {
            "trigger_price": trigger_px,
            "stop_price": stop_px,
            "stop_limit_pct": float(self.params.get("stop_limit_pct", 0.02)),
            "stop_loss_type": "low_or_pct",
            "entry_type": "vcp",
            "max_stop_pct": max_stop_pct,
            "entry_timing": entry_timing,
            "signal_mode": signal_mode,
            "signal_strength": strength,
        }

    def _check_setup_breakout_only(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        self._last_reject_reason = ""
        self._last_vcp_failure_reason = ""

        warmup = int(self.params.get("warmup_bars", 200) or 200)
        if i < warmup:
            return self._reject("warmup")

        row = df.iloc[i]
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return self._reject("close_invalid")

        # Gate 1: Trend template hierarchy.
        sma10 = _as_float(row.get("sma10"), 0.0)
        sma20 = _as_float(row.get("sma20"), 0.0)
        sma50 = _as_float(row.get("sma50"), 0.0)
        sma150 = _as_float(row.get("sma150"), 0.0)
        sma200 = _as_float(row.get("sma200"), 0.0)
        trend_template_mode = str(self.params.get("trend_template_mode", "strict") or "strict").lower()
        if trend_template_mode in {"classic", "minervini", "sepa"}:
            sma200_lookback = int(self.params.get("sma200_rising_lookback_bars", 20) or 20)
            sma200_prev = float("nan")
            if (i - sma200_lookback) >= 0:
                sma200_prev = _as_float(df.iloc[i - sma200_lookback].get("sma200"), float("nan"))
            sma200_rising = math.isfinite(sma200_prev) and sma200 > sma200_prev
            if not (close_px > sma50 > sma150 > sma200 and sma200_rising):
                return self._reject(
                    f"trend_gate classic close={close_px:.2f} sma50={sma50:.2f} "
                    f"sma150={sma150:.2f} sma200={sma200:.2f} "
                    f"sma200_prev={sma200_prev:.2f}"
                )
        else:
            if not (close_px > sma10 > sma20 > sma50 > sma150 > sma200):
                return self._reject(
                    f"trend_gate strict close={close_px:.2f} sma10={sma10:.2f} sma20={sma20:.2f} "
                    f"sma50={sma50:.2f} sma150={sma150:.2f} sma200={sma200:.2f}"
                )

        min_price = float(self.params.get("min_price", 2.0) or 2.0)
        if close_px < min_price:
            return self._reject(f"min_price close={close_px:.2f} < {min_price:.2f}")

        market_green = self._is_adaptive_market_green(row)

        min_avg_volume = float(self.params.get("min_avg_volume_30", 0.0) or 0.0)
        vol_ma30 = _as_float(row.get("vol_ma30"), float("nan"))
        if min_avg_volume > 0 and math.isfinite(vol_ma30) and vol_ma30 < min_avg_volume:
            return self._reject(f"liquidity vol_ma30={vol_ma30:.0f} < {min_avg_volume:.0f}")
        min_avg_dollar_volume_50 = float(self.params.get("min_avg_dollar_volume_50", 0.0) or 0.0)
        if market_green:
            adaptive_adv50_min = self.params.get("adaptive_green_min_avg_dollar_volume_50")
            if adaptive_adv50_min is not None:
                min_avg_dollar_volume_50 = min(
                    min_avg_dollar_volume_50,
                    float(adaptive_adv50_min or 0.0),
                )
        if min_avg_dollar_volume_50 > 0:
            vol_ma50 = _as_float(row.get("vol_ma50"), float("nan"))
            if (not math.isfinite(vol_ma50)) or vol_ma50 <= 0:
                return self._reject("liquidity adv50_missing")
            adv50 = close_px * vol_ma50
            if adv50 < min_avg_dollar_volume_50:
                return self._reject(
                    f"liquidity adv50={adv50:.0f} < {min_avg_dollar_volume_50:.0f}"
                )

        # Gate 2: Volatility floor (avoid sluggish, low-ADR names).
        adr_min = _as_percent_threshold(
            self.params.get(
                "adr_min_pct",
                self.params.get("adr_min", _DEFAULT_ADR_MIN_PCT),
            ),
            0.0,
        )
        if market_green:
            adaptive_adr_min = self.params.get("adaptive_green_adr_min_pct")
            if adaptive_adr_min is not None:
                adr_min = min(
                    adr_min,
                    _as_percent_threshold(adaptive_adr_min, 0.0),
                )
        if adr_min > 0:
            adr_pct = _as_float(row.get("adr_pct"), float("nan"))
            adr_pct_q = _as_float(row.get("adr_pct_q"), float("nan"))
            adr_values = [v for v in (adr_pct, adr_pct_q) if math.isfinite(v)]
            effective_adr = max(adr_values) if adr_values else 0.0
            if effective_adr < adr_min:
                return self._reject(f"adr_gate adr={effective_adr:.2f} < {adr_min:.2f}")

        # Gate 3: Require a meaningful prior thrust before base breakout.
        runup_min = _as_percent_threshold(
            self.params.get(
                "prior_runup_min_pct",
                self.params.get("runup_min_pct", _DEFAULT_PRIOR_RUNUP_MIN_PCT),
            ),
            0.0,
        )
        if market_green:
            adaptive_runup_min = self.params.get("adaptive_green_runup_min_pct")
            if adaptive_runup_min is not None:
                runup_min = min(
                    runup_min,
                    _as_percent_threshold(adaptive_runup_min, 0.0),
                )
        if runup_min > 0:
            ret_1m = _as_float(row.get("ret_1m"), float("nan"))
            ret_3m = _as_float(row.get("ret_3m"), float("nan"))
            runup_candidates = [v for v in (ret_1m, ret_3m) if math.isfinite(v)]
            prior_runup = max(runup_candidates) if runup_candidates else 0.0
            if prior_runup < runup_min:
                return self._reject(f"runup_gate runup={prior_runup:.2f} < {runup_min:.2f}")

        # Gate 4: Relative strength percentile gate.
        rs_percentile = self._resolve_rs_percentile(row)
        rs_gate_min = _as_percent_threshold(
            self.params.get(
                "rs_percentile_min",
                self.params.get("rs_gate_min", 85.0),
            ),
            0.0,
        )
        if market_green:
            adaptive_rs_min = self.params.get("adaptive_green_rs_min")
            if adaptive_rs_min is not None:
                rs_gate_min = min(
                    rs_gate_min,
                    _as_percent_threshold(adaptive_rs_min, 0.0),
                )
        if rs_percentile < rs_gate_min:
            return self._reject(f"rs_gate rs_percentile={rs_percentile:.2f} < {rs_gate_min:.2f}")

        # Gate 5: Fundamental growth gate with IPO/HTF override.
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

        growth_min = _as_percent_threshold(self.params.get("fundamental_growth_min_pct", 20.0), 0.0)
        # Edgar point-in-time fundamentals can contain denominator artifacts
        # (e.g., extreme negative YoY values around near-zero prior quarters).
        # Treat implausible outliers as unavailable and allow price-action override.
        min_growth_valid = float(self.params.get("fundamental_growth_min_valid_pct", -90.0) or -90.0)
        max_growth_valid = float(self.params.get("fundamental_growth_max_valid_pct", 300.0) or 300.0)
        eps_available = math.isfinite(eps_yoy) and (min_growth_valid <= eps_yoy <= max_growth_valid)
        sales_available = math.isfinite(sales_yoy) and (min_growth_valid <= sales_yoy <= max_growth_valid)
        eps_artifact_guard = float(self.params.get("fundamental_eps_artifact_guard_pct", 300.0) or 300.0)
        sales_confirm_floor = float(self.params.get("fundamental_eps_artifact_sales_confirm_pct", 10.0) or 10.0)
        if eps_available and eps_yoy > eps_artifact_guard:
            if (not sales_available) or (sales_yoy < sales_confirm_floor):
                eps_available = False
        htf_override = _as_percent_threshold(self.params.get("high_tight_flag_override_pct", 95.0), 0.0)
        price_action_pct = self._resolve_price_action_percentile(row, rs_percentile)
        strong_rs_override_min = _as_percent_threshold(self.params.get("fundamental_override_rs_min", 95.0), 0.0)
        strong_price_override = (
            price_action_pct >= htf_override and rs_percentile >= strong_rs_override_min
        )
        allow_price_override = bool(self.params.get("fundamental_override_enabled", True))

        if eps_available or sales_available:
            eps_ok = eps_available and eps_yoy >= growth_min
            sales_ok = sales_available and sales_yoy >= growth_min
            if not (eps_ok or sales_ok):
                if allow_price_override and strong_price_override:
                    pass
                else:
                    return self._reject(
                        f"fundamental_gate eps_yoy={eps_yoy:.2f} sales_yoy={sales_yoy:.2f} < {growth_min:.2f}"
                    )
        else:
            if allow_price_override and strong_price_override:
                pass
            else:
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
                vcp_failure = str(self._last_vcp_failure_reason or "").strip()
                details.append(f"vcp:{vcp_failure}" if vcp_failure else "vcp:no_breakout")
            if allow_ep and ep_candidate is None:
                details.append(ep_failure or "ep:no_signal")
            if details:
                return self._reject("archetype_none " + " | ".join(details))
            return self._reject("archetype_none")

        selected = max(candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
        selected.pop("signal_strength", None)
        return selected

    def _check_setup_multi_sleeve(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        self._last_reject_reason = ""
        self._last_vcp_failure_reason = ""

        warmup = int(self.params.get("warmup_bars", 200) or 200)
        if i < warmup:
            return self._reject("warmup")

        row = df.iloc[i]
        close_px = _as_float(row.get("close"), 0.0)
        if close_px <= 0:
            return self._reject("close_invalid")

        rs_percentile = self._resolve_rs_percentile(row)
        candidates: List[Dict[str, Any]] = []
        details: List[str] = []

        enabled = self.params.get("enabled_sleeves", ["breakout", "continuation", "recovery"])
        if isinstance(enabled, str):
            enabled_sleeves = {enabled.strip().lower()}
        elif isinstance(enabled, (list, tuple, set)):
            enabled_sleeves = {str(item).strip().lower() for item in enabled if str(item).strip()}
        else:
            enabled_sleeves = {"breakout", "continuation", "recovery"}
        if not enabled_sleeves:
            enabled_sleeves = {"breakout"}

        if "breakout" in enabled_sleeves:
            breakout_candidate = self._check_setup_breakout_only(df, i)
            if breakout_candidate is not None:
                breakout_candidate = dict(breakout_candidate)
                breakout_candidate.setdefault("sleeve", "breakout")
                candidates.append(breakout_candidate)
            else:
                details.append(str(self._last_reject_reason or "breakout_rejected"))

        if "continuation" in enabled_sleeves:
            continuation_candidate = evaluate_continuation(
                df,
                i,
                self.params,
                rs_percentile=rs_percentile,
            )
            if continuation_candidate is None and bool(
                self.params.get("continuation_breakout_enabled", False)
            ):
                continuation_candidate = evaluate_continuation_breakout(
                    df,
                    i,
                    self.params,
                    rs_percentile=rs_percentile,
                )
            if continuation_candidate is not None:
                candidates.append(continuation_candidate)
            else:
                details.append("continuation:no_signal")

        if "recovery" in enabled_sleeves:
            recovery_candidate = evaluate_recovery(
                df,
                i,
                self.params,
                rs_percentile=rs_percentile,
            )
            if recovery_candidate is not None:
                candidates.append(recovery_candidate)
            else:
                details.append("recovery:no_signal")

        if not candidates:
            return self._reject("archetype_none " + " | ".join(details))

        # Normalize breakout archetypes into the breakout sleeve and select winner.
        breakout_like = [c for c in candidates if str(c.get("sleeve", "")).lower() == "breakout"]
        non_breakout = [c for c in candidates if str(c.get("sleeve", "")).lower() != "breakout"]
        selected_breakout = None
        if breakout_like:
            vcp = next((c for c in breakout_like if str(c.get("entry_type", "")).lower() == "vcp"), None)
            ep = next((c for c in breakout_like if str(c.get("entry_type", "")).lower() == "ep"), None)
            selected_breakout = select_breakout_candidate(vcp, ep)
            if selected_breakout is None:
                selected_breakout = max(breakout_like, key=lambda c: float(c.get("signal_strength", 0.0)))

        merged_candidates: List[Dict[str, Any]] = []
        if selected_breakout is not None:
            merged_candidates.append(selected_breakout)
        merged_candidates.extend(non_breakout)
        selected = max(merged_candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
        selected = dict(selected)
        selected.pop("signal_strength", None)
        return selected

    def check_setup(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        multi_enabled = bool(self.params.get("multi_sleeve_enabled", False))
        if not multi_enabled:
            return self._check_setup_breakout_only(df, i)
        return self._check_setup_multi_sleeve(df, i)

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
            "stop_to_avg_cost": bool(self.params.get("pyramid_stop_to_avg_cost", False)),
        }
