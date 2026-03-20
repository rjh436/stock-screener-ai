from __future__ import annotations

import json
import operator
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


_SCALAR_OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

_STOP_ICON = "\U0001F6D1"
_TRAIL_ICON = "\U0001F6E1\ufe0f"
_TIME_ICON = "\u23f1\ufe0f"
_TREND_ICON = "\U0001F4C9"
_TARGET_ICON = "\U0001F3AF"


def _unwrap_genome(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if isinstance(value, dict):
        for _ in range(3):
            nested = None
            for key in ("genome", "params", "strategy"):
                if isinstance(value.get(key), dict):
                    nested = value.get(key)
                    break
            if nested is None:
                break
            value = nested
    return value if isinstance(value, dict) else None


def _resolve_strategy_genome(row, strategies_map):
    genome = _unwrap_genome(
        row.get("genome")
        or row.get("Genome")
        or row.get("params")
        or row.get("Params")
        or row.get("strategy_genome")
    )
    if genome is None:
        strat_obj = row.get("strategy_obj")
        if strat_obj is not None:
            genome = _unwrap_genome(getattr(strat_obj, "params", None) or getattr(strat_obj, "genome", None))

    strat = genome
    strat_name = row.get("Strategy") or row.get("strategy_name") or row.get("strategy") or ""
    if strat is None and isinstance(strat_name, dict):
        strat = _unwrap_genome(strat_name)
    if strat is None:
        if isinstance(strat_name, str):
            strat_name = strat_name.split(" + ")[0].strip()
        else:
            strat_name = str(strat_name).split(" + ")[0].strip()
        strat = strategies_map.get(strat_name)
        if not strat and strat_name:
            def _normalize(name: str) -> str:
                return "".join(ch for ch in str(name).lower() if ch.isalnum())

            target = _normalize(strat_name)
            best_key = ""
            best_cfg = None
            for key, cfg in strategies_map.items():
                key_norm = _normalize(str(key))
                if not key_norm:
                    continue
                if key_norm in target or target in key_norm:
                    if len(key_norm) > len(best_key):
                        best_key = key_norm
                        best_cfg = cfg
            strat = best_cfg
    return strat


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


def _as_float(value, default=0.0):
    try:
        val = float(value)
    except Exception:
        return default
    if not np.isfinite(val):
        return default
    return val


def calculate_stop_price(entry_price, atr, multiplier):
    stop = _as_float(entry_price) - (_as_float(atr) * _as_float(multiplier))
    return round(stop, 2)


def rehydrate_exit_state(row, strategies_map, history_map=None):
    strat = _resolve_strategy_genome(row, strategies_map)
    if not strat:
        return {"valid": False, "reason": "missing_strategy"}

    symbol = str(row.get("Symbol") or row.get("symbol") or "").upper()
    df = None
    if history_map and symbol:
        df = history_map.get(symbol)
        if df is None:
            df = history_map.get(symbol.upper())
        if df is None:
            df = history_map.get(symbol.lower())
    if df is None or df.empty:
        return {"valid": False, "reason": "missing_history", "strategy": strat}

    df_cols = {str(col).lower() for col in df.columns}
    required_cols = {"close", "high", "low", "atr14", "sma50", "sma20", "sma10", "bb_upper"}
    if not required_cols.issubset(df_cols):
        from execution.engine import _compute_indicators

        df = _compute_indicators(df)
        if df is None or df.empty:
            return {"valid": False, "reason": "missing_indicators", "strategy": strat}
    else:
        if any(str(col).lower() != col for col in df.columns):
            df = df.copy()
            df.columns = df.columns.str.lower()

    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    loc = row.get("loc")
    if loc is None:
        loc = row.get("current_i") or row.get("bar_i")
    try:
        loc = int(loc) if loc is not None else len(df) - 1
    except Exception:
        loc = len(df) - 1

    if loc < 0 or loc >= len(df):
        return {"valid": False, "reason": "missing_bars", "strategy": strat}

    entry_price = round(_as_float(row.get("Entry Price") or row.get("entry_price")), 2)
    initial_stop = round(_as_float(row.get("Stop Loss") or row.get("stop_price")), 2)

    row_last = df.iloc[loc]
    close_px = round(_as_float(row_last.get("close"), default=entry_price), 2)
    if close_px <= 0:
        close_px = entry_price
    high_px = round(_as_float(row_last.get("high"), default=close_px), 2)
    low_px = round(_as_float(row_last.get("low"), default=close_px), 2)

    if entry_price <= 0:
        entry_price = close_px

    pnl_pct = ((close_px - entry_price) / entry_price) * 100.0 if entry_price else 0.0

    atr = _as_float(row_last.get("atr14"), default=close_px * 0.02)
    if atr <= 0:
        atr = close_px * 0.02

    sma50 = _as_float(row_last.get("sma50"), default=float("nan"))
    sma20 = _as_float(row_last.get("sma20"), default=float("nan"))
    bb_upper = _as_float(row_last.get("bb_upper"), default=float("nan"))

    entry_idx = loc
    entry_raw = row.get("Date") or row.get("date") or row.get("Entry Date")
    if entry_raw is not None:
        try:
            entry_ts = pd.to_datetime(entry_raw)
            if getattr(entry_ts, "tzinfo", None) is not None:
                entry_ts = entry_ts.tz_localize(None)
            entry_idx = int(df.index.searchsorted(entry_ts))
        except Exception:
            entry_idx = loc
    else:
        entry_i = row.get("entry_i")
        if isinstance(entry_i, (int, np.integer)) and 0 <= int(entry_i) <= loc:
            entry_idx = int(entry_i)
    entry_idx = max(0, min(entry_idx, loc))

    days_held = max(0, loc - entry_idx)

    try:
        peak_val = float(np.nanmax(df["high"].iloc[entry_idx : loc + 1]))
    except Exception:
        peak_val = float("nan")
    peak_high = round(peak_val, 2) if np.isfinite(peak_val) else high_px

    trail_mult_raw = strat.get("trail_atr")
    if trail_mult_raw is not None:
        try:
            trail_mult = float(trail_mult_raw)
        except (TypeError, ValueError):
            trail_mult_raw = None
    if trail_mult_raw is None:
        try:
            trail_mult = float(strat.get("stop_loss_atr", 3.0))
        except (TypeError, ValueError):
            trail_mult = 3.0

    try:
        act_raw = strat.get("trail_activation", 1.0)
        trail_activation = float(act_raw or 1.0)
    except (TypeError, ValueError, AttributeError):
        trail_activation = 1.0
    if not np.isfinite(trail_activation) or trail_activation < 1.0:
        trail_activation = 1.0

    activation_price = round(entry_price * trail_activation, 2) if entry_price > 0 else 0.0
    trailing_enabled = np.isfinite(trail_mult) and trail_mult > 0
    trailing_active = trailing_enabled and (
        (trail_activation <= 1.0) or (entry_price > 0 and peak_high >= activation_price)
    )
    trailing_stop = float("-inf")
    if trailing_active and atr > 0:
        trailing_stop = round(peak_high - (atr * trail_mult), 2)

    if initial_stop <= 0 and entry_price > 0:
        initial_stop = round(entry_price * 0.9, 2)

    effective_stop = initial_stop
    if np.isfinite(trailing_stop):
        effective_stop = max(effective_stop, float(trailing_stop))
    if np.isfinite(effective_stop):
        effective_stop = round(effective_stop, 2)

    breakeven_raw = strat.get("breakeven_pct")
    if breakeven_raw is not None and entry_price > 0:
        try:
            breakeven_val = float(breakeven_raw)
        except (TypeError, ValueError):
            breakeven_val = 0.0
        if breakeven_val > 0:
            breakeven_thresh = breakeven_val if breakeven_val > 1.0 else breakeven_val * 100.0
            if pnl_pct >= breakeven_thresh:
                effective_stop = max(effective_stop, entry_price)
                if np.isfinite(effective_stop):
                    effective_stop = round(effective_stop, 2)

    try:
        time_stop = int(strat.get("time_stop_days", strat.get("time_stop", 7)) or 7)
    except Exception:
        time_stop = 7
    time_stop = max(1, time_stop)
    days_left = time_stop - days_held
    time_compression = days_held >= int(time_stop * 0.8)

    stop_breached = bool(effective_stop > 0 and low_px < effective_stop)
    trend_threat = bool(np.isfinite(sma50) and sma50 > 0 and close_px <= sma50)

    use_bb_exit = bool(strat.get("use_bb_exit", False))
    target_px = None
    if not use_bb_exit:
        mult = _best_profit_target_multiple(strat.get("exit_rules"))
        if mult is not None and entry_price > 0:
            target_px = round(entry_price * mult, 2)

    return {
        "valid": True,
        "entry_price": entry_price,
        "close_px": close_px,
        "low_px": low_px,
        "high_px": high_px,
        "atr": atr,
        "sma20": sma20,
        "sma50": sma50,
        "entry_idx": entry_idx,
        "days_held": days_held,
        "days_left": days_left,
        "time_stop": time_stop,
        "time_compression": time_compression,
        "peak_high": peak_high,
        "trail_activation": trail_activation,
        "activation_price": activation_price,
        "trailing_active": trailing_active,
        "trailing_stop": trailing_stop,
        "effective_stop": effective_stop,
        "stop_breached": stop_breached,
        "trend_threat": trend_threat,
        "target_px": target_px,
        "use_bb_exit": use_bb_exit,
        "bb_upper": bb_upper,
        "pnl_pct": pnl_pct,
        "row_last": row_last,
    }


def calc_exit_plan(row, strategies_map, history_map=None):
    state = rehydrate_exit_state(row, strategies_map, history_map)
    if not state.get("valid"):
        return "Unknown"

    if state["stop_breached"]:
        return f"{_STOP_ICON} Stop Breached: ${state['effective_stop']:.2f}"

    if state["trailing_active"]:
        stop_px = state["effective_stop"]
        peak_high = state.get("peak_high")
        if np.isfinite(peak_high):
            return f"{_TRAIL_ICON} Trailing: ${stop_px:.2f} (Peak ${peak_high:.2f})"
        return f"{_TRAIL_ICON} Trailing: ${stop_px:.2f}"

    if state["time_compression"]:
        days_left = state.get("days_left")
        if days_left is None:
            return f"{_TIME_ICON} Time-Exit Looming"
        if days_left <= 0:
            return f"{_TIME_ICON} Time-Exit Due"
        return f"{_TIME_ICON} Time-Exit Looming ({days_left}d left)"

    if state["trend_threat"]:
        sma50 = state.get("sma50")
        if sma50 is not None and np.isfinite(sma50):
            return f"{_TREND_ICON} Trend Support (SMA50): ${sma50:.2f}"
        return f"{_TREND_ICON} Trend Support (SMA50): N/A"

    target_px = state.get("target_px")
    if target_px is not None and np.isfinite(target_px):
        return f"{_TARGET_ICON} Profit Target: ${target_px:.2f}"
    return f"{_TARGET_ICON} Profit Target: N/A"


def _exit_plan_style(val: str) -> str:
    if not isinstance(val, str):
        return ""
    if val.startswith(_STOP_ICON):
        return "background-color: #8b0000; color: #ffffff; font-weight: 600;"
    if val.startswith(_TRAIL_ICON):
        return "background-color: #f39c12; color: #000000; font-weight: 600;"
    if val.startswith(_TIME_ICON):
        return "background-color: #f1c232; color: #000000; font-weight: 600;"
    return ""


def _generic_exit_decision(
    strategy_obj: Any,
    sd: Any,
    loc: int,
    entry_loc: int,
    entry_px: float,
    current_stop: float,
    partial_taken: bool = False,
    initial_risk: Optional[float] = None,
) -> Tuple[bool, float, Optional[float]]:
    """
    Centralized exit logic for both backtesting and live execution.
    Returns: (should_exit, new_stop_price, target_price_hit)
    """
    if isinstance(strategy_obj, dict):
        genome = strategy_obj
    else:
        genome = getattr(strategy_obj, "params", getattr(strategy_obj, "genome", {})) or {}
    params = _unwrap_genome(genome) or (genome if isinstance(genome, dict) else {})

    close_px = float(sd.close[loc])
    high_px = float(sd.high[loc])
    low_px = float(sd.low[loc])

    # 1. STOP LOSS CHECK
    if low_px <= current_stop:
        return True, current_stop, None

    # 2. TRAILING STOP LOGIC (ATR)
    # FIX: Explicit check for None to allow 0.0 to disable trailing
    trail_raw = params.get("trail_atr")
    if trail_raw is not None:
        trail_mult = float(trail_raw)
    else:
        trail_mult = float(params.get("stop_loss_atr", 3.0))

    new_stop = current_stop

    if trail_mult > 0.001:
        # Trail Activation
        activation_mult = float(params.get("trail_activation", 1.0) or 1.0)
        atr_val = float(sd.atr14[loc])

        # Calculate theoretical trail stop based on High - Trail
        potential_stop = high_px - (atr_val * trail_mult)

        # Only raise stop, never lower
        if potential_stop > current_stop:
            # Check activation
            dist_from_entry = (high_px - entry_px)
            atr_at_entry = float(sd.atr14[entry_loc])
            if dist_from_entry >= (atr_at_entry * activation_mult):
                new_stop = potential_stop

    # 2b. NUCLEAR RESET TRAILING SCHEDULE
    # >20%: stop to breakeven | >40%: trail SMA50 | >100%: trail SMA10.
    profit_pct = (close_px - entry_px) / entry_px if entry_px > 0 else 0.0

    # Superperformance configs historically used breakeven_at_pct; honor both keys.
    be_threshold = _as_float(
        params.get("breakeven_profit_pct", params.get("breakeven_at_pct", 0.20)),
        0.20,
    )
    if be_threshold > 1.0:
        be_threshold /= 100.0
    if profit_pct >= max(0.0, be_threshold) and entry_px > new_stop:
        new_stop = entry_px

    sma50_threshold = _as_float(params.get("sma50_trail_profit_pct", 0.40), 0.40)
    sma10_threshold = _as_float(params.get("sma10_trail_profit_pct", 1.00), 1.00)
    if sma50_threshold > 1.0:
        sma50_threshold /= 100.0
    if sma10_threshold > 1.0:
        sma10_threshold /= 100.0

    trail_choice = None
    if profit_pct >= sma10_threshold:
        trail_choice = "sma10"
    elif profit_pct >= sma50_threshold:
        trail_choice = "sma50"

    dynamic_trail_active = False
    if trail_choice:
        try:
            ma_arr = getattr(sd, trail_choice)
            ma_val = float(ma_arr[loc])
        except Exception:
            ma_val = float("nan")
        if not np.isfinite(ma_val):
            row_df_local = getattr(sd, "df", None)
            if row_df_local is not None:
                try:
                    ma_val = float(row_df_local.iloc[loc].get(trail_choice, float("nan")))
                except Exception:
                    ma_val = float("nan")
        if np.isfinite(ma_val) and ma_val > new_stop:
            new_stop = ma_val
            dynamic_trail_active = True

    # 3. TIME STOP (Dead-Money Rule)
    dead_money_days = int(params.get("dead_money_days", params.get("time_stop_days", 5)) or 5)
    if dead_money_days < 0:
        dead_money_days = 0
    dead_money_profit_pct = _as_float(
        params.get("dead_money_profit_pct", params.get("time_stop_profit_pct", 0.01)),
        0.01,
    )
    if dead_money_profit_pct > 1.0:
        dead_money_profit_pct /= 100.0
    if dead_money_days > 0:
        days_held = loc - entry_loc
        if days_held >= dead_money_days and profit_pct < dead_money_profit_pct:
            return True, new_stop, None

    # 4. FIXED PROFIT TARGETS (Legacy opt-in only)
    target_px = None
    allow_profit_target_exit = bool(params.get("allow_profit_target_exit", False))
    if allow_profit_target_exit:
        profit_target = float(params.get("profit_target", 0.0) or 0.0)
        if profit_target > 0:
            target_mult = profit_target if profit_target >= 1.0 else (1.0 + profit_target)
            target_px = entry_px * target_mult
        if target_px is None:
            mult = _best_profit_target_multiple(params.get("exit_rules"))
            if mult is not None:
                target_px = entry_px * mult
        if target_px is not None and high_px >= target_px:
            return True, new_stop, target_px

    row_df = getattr(sd, "df", None)
    if row_df is not None:
        try:
            row_last = row_df.iloc[loc]
        except Exception:
            row_last = None
        
        # --- ZOMBIE RUNNER FIX ---
        # If partial profit taken, we switch to "Loose Hold" (SMA50)
        # Otherwise use the genome's configured exit (e.g., SMA10 or SMA20)
        active_exit_sma = (
            params.get("exit_ma")
            or params.get("exit_sma")
            or params.get("exit_sma_fast")
            or "sma20"
        )
        if partial_taken:
            active_exit_sma = (
                params.get("exit_ma_after_partial")
                or params.get("exit_sma_after_partial")
                or params.get("exit_sma_slow")
                or "sma50"
            )
        if not partial_taken and params.get("exit_ma_after_profit") and close_px > entry_px:
            active_exit_sma = params.get("exit_ma_after_profit")

        # If we are explicitly trailing via SMA, rely on the stop update instead of
        # immediate SMA cross exits (avoids premature exits on minor dips).
        # Also suppress SMA exits when dynamic parabolic trailing is active.
        if params.get("trail_ma") or dynamic_trail_active:
            active_exit_sma = None
            
        # Check SMA Exit Logic
        # We need to find the rule corresponding to the active SMA
        # This is a bit indirect because params['exit_rules'] is a list of dicts.
        # But we can also look up the SMA value directly if we know the string.
        
        # Simplified Check for SMA Breach:
        # If the 'active_exit_sma' column exists and Close < SMA, EXIT.
        if active_exit_sma in ["sma10", "sma20", "sma50"]:
            sma_val = float(sd.df.iloc[loc].get(active_exit_sma, 0))
            if 0 < sma_val and close_px < sma_val:
                return True, new_stop, None

        for rule in params.get("exit_rules", []) or []:
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
            if row_last is None:
                break
            val_a = row_last.get(col, np.nan)
            if "val" in rule:
                val_b = rule.get("val", np.nan)
            elif "ref" in rule:
                val_b = row_last.get(rule.get("ref"), np.nan)
            else:
                continue
            try:
                a = float(val_a)
                b = float(val_b)
            except Exception:
                continue
            if not np.isfinite(a) or not np.isfinite(b):
                continue
            if op_fn(a, b):
                return True, new_stop, None

    return False, new_stop, None
