from typing import Dict, List, Optional, Sequence, Tuple
import math
import numpy as np
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


_Pivot = Tuple[int, str, float, float, float]


def _collect_pivots(df_window: pd.DataFrame, pivot_span: int) -> List[_Pivot]:
    highs = pd.to_numeric(df_window.get("high"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    lows = pd.to_numeric(df_window.get("low"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vols = pd.to_numeric(df_window.get("volume"), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    vol_ma50 = pd.to_numeric(df_window.get("vol_ma50"), errors="coerce").to_numpy(dtype=np.float64, copy=False)

    n = len(df_window)
    if n < (pivot_span * 2 + 3):
        return []

    roll_window = (pivot_span * 2) + 1
    hi_roll = (
        pd.Series(highs)
        .rolling(roll_window, center=True, min_periods=roll_window)
        .max()
        .to_numpy(dtype=np.float64, copy=False)
    )
    lo_roll = (
        pd.Series(lows)
        .rolling(roll_window, center=True, min_periods=roll_window)
        .min()
        .to_numpy(dtype=np.float64, copy=False)
    )

    pivot_high = np.isfinite(highs) & np.isfinite(hi_roll) & (highs >= (hi_roll - 1e-9))
    pivot_low = np.isfinite(lows) & np.isfinite(lo_roll) & (lows <= (lo_roll + 1e-9))
    pivot_idx = np.flatnonzero(pivot_high | pivot_low)
    if pivot_idx.size == 0:
        return []

    pivots: List[_Pivot] = []
    for idx in pivot_idx.tolist():
        is_high = bool(pivot_high[idx])
        is_low = bool(pivot_low[idx])
        if is_high and is_low:
            # Flat bars can tag both; skip ambiguous pivots.
            continue
        if is_high:
            pivots.append((idx, "H", float(highs[idx]), float(vols[idx]), float(vol_ma50[idx])))
        elif is_low:
            pivots.append((idx, "L", float(lows[idx]), float(vols[idx]), float(vol_ma50[idx])))
    if len(pivots) < 3:
        return pivots

    # Compress consecutive same-type pivots into the dominant extreme.
    compressed: List[_Pivot] = [pivots[0]]
    for ev in pivots[1:]:
        prev = compressed[-1]
        if ev[1] == prev[1]:
            if ev[1] == "H" and ev[2] >= prev[2]:
                compressed[-1] = ev
            elif ev[1] == "L" and ev[2] <= prev[2]:
                compressed[-1] = ev
        else:
            compressed.append(ev)
    return compressed


def _extract_contractions(pivots: Sequence[_Pivot]) -> Tuple[List[float], List[Tuple[_Pivot, _Pivot]]]:
    contractions: List[float] = []
    pairs: List[Tuple[_Pivot, _Pivot]] = []
    for i in range(len(pivots) - 1):
        left = pivots[i]
        right = pivots[i + 1]
        if left[1] != "H" or right[1] != "L":
            continue
        hi_px = float(left[2])
        lo_px = float(right[2])
        if not (math.isfinite(hi_px) and math.isfinite(lo_px) and hi_px > 0 and lo_px > 0):
            continue
        contraction = ((hi_px - lo_px) / hi_px) * 100.0
        if contraction <= 0 or not math.isfinite(contraction):
            continue
        contractions.append(contraction)
        pairs.append((left, right))
    return contractions, pairs


def _pivot_volume_ok(pivots: Sequence[_Pivot]) -> bool:
    for _, _, _, vol, vol_ma50 in pivots:
        if not (math.isfinite(vol) and math.isfinite(vol_ma50)):
            continue
        if vol_ma50 > 0 and vol >= vol_ma50:
            return False
    return True


def _contraction_ok(df: pd.DataFrame, i: int, params: Dict) -> bool:
    lookback = int(params.get("vcp_lookback_bars", 80) or 80)
    pivot_span = int(params.get("vcp_pivot_span", 1) or 1)
    required = int(params.get("vcp_required_contractions", 2) or 2)
    max_base_depth = float(params.get("vcp_max_base_depth_pct", 35.0) or 35.0)
    max_last = float(params.get("vcp_last_contraction_max_pct", 10.0) or 10.0)
    damping_ratio = float(params.get("vcp_damping_ratio", 0.75) or 0.75)
    require_pivot_vol_dryup = bool(params.get("vcp_require_pivot_volume_dryup", True))

    start = max(0, i - lookback + 1)
    window = df.iloc[start : i + 1]
    if len(window) < max(20, (pivot_span * 2) + 5):
        return False

    pivots = _collect_pivots(window, pivot_span)
    if len(pivots) < (required * 2):
        return False

    contractions, pairs = _extract_contractions(pivots)
    if len(contractions) < required:
        return False

    recent = contractions[-required:]
    recent_pairs = pairs[-required:]

    if recent[0] > max_base_depth:
        return False
    if recent[-1] > max_last:
        return False

    # Damping oscillation requirement: C1 > C2 > C3 ...
    for prev, curr in zip(recent, recent[1:]):
        if curr >= prev:
            return False
        if damping_ratio > 0 and curr > (prev * damping_ratio):
            return False

    if require_pivot_vol_dryup:
        pivot_slice: List[_Pivot] = []
        for hi_pivot, lo_pivot in recent_pairs:
            pivot_slice.extend([hi_pivot, lo_pivot])
        if not _pivot_volume_ok(pivot_slice):
            return False
    return True


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

        # VCP state machine: swing pivots + damped contractions + pivot dry-up.
        if not _contraction_ok(df, i, self.params):
            return None

        self._gate_counts["vcp_pass"] += 1

        stop_buy_ref = self.params.get("stop_buy_ref", "high_20_prev")
        pivot = _as_float(row.get(stop_buy_ref), _as_float(row.get("high_20_prev"), 0.0))
        if pivot <= 0:
            return None

        # Require breakout day confirmation (close > pivot) + volume expansion
        vol = _as_float(row.get("volume"), 0.0)
        vol_ma50 = _as_float(row.get("vol_ma50"), vol)
        vol_mult = float(self.params.get("vol_mult", 2.0))
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
            "entry_type": "vcp",
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
