from __future__ import annotations

import operator
from typing import Any, Dict, Optional, Tuple

import numpy as np


_SCALAR_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _best_profit_target_multiple(exit_rules: Any) -> Optional[float]:
    best: Optional[float] = None
    for rule in exit_rules or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") != "profit_target":
            continue
        try:
            mult = float(rule.get("val", 1.0) or 1.0)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(mult) or mult <= 1.0:
            continue
        best = mult if best is None else min(best, mult)
    return best


def _generic_exit_decision(
    genome: Dict[str, Any],
    sd: Any,
    loc: int,
    entry_i: int,
    entry_price: float,
    initial_stop: float,
) -> Tuple[bool, float, Optional[float]]:
    """
    Engine-native exit logic for rule-based (GenericStrategy) genomes.

    Returns:
      - should_exit
      - effective_stop (including trail_activation)
      - target_px (when profit_target triggers), else None
    """
    if isinstance(genome, dict) and isinstance(genome.get("genome"), dict):
        genome = genome["genome"]

    np_isfinite = np.isfinite

    close_px = float(sd.close[loc])
    if not np_isfinite(close_px) or close_px <= 0:
        close_px = float(entry_price)

    high_px = float(sd.high[loc])
    if not np_isfinite(high_px):
        high_px = close_px

    low_px = float(sd.low[loc])
    if not np_isfinite(low_px):
        low_px = close_px

    days_held = int(loc) - int(entry_i)
    pnl_pct = ((close_px - entry_price) / entry_price) * 100.0 if entry_price else 0.0

    use_bb_exit = bool(genome.get("use_bb_exit", False))

    atr = float(sd.atr14[loc])
    if not np_isfinite(atr) or atr <= 0:
        atr = close_px * 0.02

    trail_mult_raw = genome.get("trail_atr", genome.get("stop_loss_atr", 3.0))
    try:
        trail_mult = float(trail_mult_raw or 3.0)
    except (TypeError, ValueError):
        trail_mult = 3.0

    try:
        act_raw = genome.get("trail_activation", 1.0) if isinstance(genome, dict) else 1.0
        trail_activation = float(act_raw or 1.0)
    except (TypeError, ValueError, AttributeError):
        trail_activation = 1.0
    if not np_isfinite(trail_activation) or trail_activation < 1.0:
        trail_activation = 1.0

    peak_high = high_px
    if 0 <= entry_i <= loc:
        try:
            peak_high_val = float(np.nanmax(sd.high[entry_i : loc + 1]))
        except Exception:
            peak_high_val = float("nan")
        if np_isfinite(peak_high_val):
            peak_high = peak_high_val

    trailing_active = (trail_activation <= 1.0) or (peak_high >= entry_price * trail_activation)
    trailing_stop = float("-inf")
    if trailing_active and atr > 0:
        trailing_stop = peak_high - (atr * trail_mult)

    effective_stop = float(initial_stop)
    if np_isfinite(trailing_stop):
        effective_stop = max(effective_stop, float(trailing_stop))

    breakeven_raw = genome.get("breakeven_pct") if isinstance(genome, dict) else None
    if breakeven_raw is not None and entry_price > 0:
        try:
            breakeven_val = float(breakeven_raw)
        except (TypeError, ValueError):
            breakeven_val = 0.0
        if breakeven_val > 0:
            breakeven_thresh = breakeven_val if breakeven_val > 1.0 else breakeven_val * 100.0
            if pnl_pct >= breakeven_thresh:
                effective_stop = max(effective_stop, float(entry_price))

    # 1. STOP LOSS CHECK
    if low_px < effective_stop:
        return True, effective_stop, None

    # 1b. Bollinger profit release (optional)
    if use_bb_exit:
        bb_upper = float(sd.bb_upper[loc])
        if np_isfinite(bb_upper) and high_px >= bb_upper:
            return True, effective_stop, None

    # 2. TIME-BASED EXITS
    try:
        time_limit = int(genome.get("time_stop", 45) or 45)
    except Exception:
        time_limit = 45
    time_limit = max(1, time_limit)

    # Progressive tightening in final 20% of hold period
    if days_held >= int(time_limit * 0.8):
        if pnl_pct < -5.0:
            return True, effective_stop, None
        if pnl_pct < 3.0:
            sma20 = float(sd.sma20[loc])
            if not np_isfinite(sma20):
                sma20 = close_px
            if close_px < sma20:
                return True, effective_stop, None

    # Final time limit
    if days_held >= time_limit:
        if pnl_pct < 0:
            return True, effective_stop, None
        if pnl_pct < 8.0:
            return True, effective_stop, None
        sma50 = float(sd.sma50[loc])
        if np_isfinite(sma50) and close_px < sma50:
            return True, effective_stop, None

    # 3. TREND BREAK PROTECTION
    sma50 = float(sd.sma50[loc])
    if np_isfinite(sma50) and close_px < sma50:
        if pnl_pct > 0:
            return True, effective_stop, None
        if days_held > 10:
            return True, effective_stop, None

    # 4. PROFIT TARGETS
    target_px: Optional[float] = None
    if not use_bb_exit:
        mult = _best_profit_target_multiple(genome.get("exit_rules"))
        if mult is not None:
            target_px = entry_price * mult
            if high_px >= target_px:
                return True, effective_stop, target_px

    # 5. OTHER RULE-BASED EXITS
    row = None
    for rule in genome.get("exit_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") == "profit_target":
            continue

        col = rule.get("col")
        op = rule.get("op")
        if not col or not op:
            continue

        op_fn = _SCALAR_OPS.get(str(op))
        if op_fn is None:
            continue

        if row is None:
            try:
                row = sd.df.iloc[loc]
            except Exception:
                break

        val_a = row.get(col, np.nan)
        if "val" in rule:
            val_b = rule.get("val", np.nan)
        elif "ref" in rule:
            val_b = row.get(rule.get("ref"), np.nan)
        else:
            continue

        try:
            a = float(val_a)
            b = float(val_b)
        except Exception:
            continue
        if not np_isfinite(a) or not np_isfinite(b):
            continue
        if op_fn(a, b):
            return True, effective_stop, None

    return False, effective_stop, None
