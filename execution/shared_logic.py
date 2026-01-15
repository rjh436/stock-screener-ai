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
    required_cols = {"close", "high", "low", "atr14", "sma50", "sma20", "bb_upper"}
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
        time_stop = int(strat.get("time_stop", 45) or 45)
    except Exception:
        time_stop = 45
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

    symbol_key = "__ENGINE__"
    row = {
        "genome": genome,
        "entry_i": entry_i,
        "entry_price": entry_price,
        "stop_price": initial_stop,
        "symbol": symbol_key,
        "loc": loc,
    }
    history_map = {symbol_key: getattr(sd, "df", None)}
    state = rehydrate_exit_state(row, {}, history_map)
    if not state.get("valid"):
        return False, float(initial_stop), None

    np_isfinite = np.isfinite

    entry_px = float(state.get("entry_price", entry_price))
    close_px = float(state.get("close_px", entry_px))
    high_px = float(state.get("high_px", close_px))
    low_px = float(state.get("low_px", close_px))

    effective_stop = state.get("effective_stop")
    if effective_stop is None or not np_isfinite(effective_stop):
        effective_stop = float(initial_stop)
    else:
        effective_stop = float(effective_stop)

    target_px = state.get("target_px")
    target_px = float(target_px) if target_px is not None and np_isfinite(target_px) else None

    pnl_pct = state.get("pnl_pct")
    if pnl_pct is None:
        pnl_pct = ((close_px - entry_px) / entry_px) * 100.0 if entry_px else 0.0
    else:
        pnl_pct = float(pnl_pct)

    days_held = int(state.get("days_held", int(loc) - int(entry_i)))
    time_limit = int(state.get("time_stop", 45) or 45)
    time_limit = max(1, time_limit)

    # 1. STOP LOSS CHECK
    if state.get("stop_breached"):
        return True, effective_stop, None

    # 1b. Bollinger profit release (optional)
    use_bb_exit = bool(state.get("use_bb_exit", False))
    if use_bb_exit:
        bb_upper = state.get("bb_upper")
        if bb_upper is not None and np_isfinite(bb_upper) and high_px >= float(bb_upper):
            return True, effective_stop, None

    # 2. TIME-BASED EXITS
    if days_held >= int(time_limit * 0.8):
        if pnl_pct < -5.0:
            return True, effective_stop, None
        if pnl_pct < 3.0:
            sma20 = state.get("sma20", np.nan)
            if not np_isfinite(sma20):
                sma20 = close_px
            if close_px < float(sma20):
                return True, effective_stop, None

    # Final time limit
    if days_held >= time_limit:
        if pnl_pct < 0:
            return True, effective_stop, None
        if pnl_pct < 8.0:
            return True, effective_stop, None
        sma50 = state.get("sma50", np.nan)
        if np_isfinite(sma50) and close_px < float(sma50):
            return True, effective_stop, None

    # 3. TREND BREAK PROTECTION
    sma50 = state.get("sma50", np.nan)
    if np_isfinite(sma50) and close_px < float(sma50):
        if pnl_pct > 0:
            return True, effective_stop, None
        if days_held > 10:
            return True, effective_stop, None

    # SMA surfing override (profit-protect for runners)

    if pnl_pct > 0 and days_held > 3:

        sma10 = state.get("sma10")

        if sma10 is None:

            row_last = state.get("row_last")

            if row_last is not None:

                sma10 = row_last.get("sma10")

        if sma10 is not None and np_isfinite(sma10) and close_px < float(sma10):

            return True, effective_stop, None


    # 4. PROFIT TARGETS
    if target_px is not None and high_px >= target_px:
        return True, effective_stop, target_px

    # 5. OTHER RULE-BASED EXITS
    row_last = state.get("row_last")
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

        if row_last is None:
            try:
                row_last = sd.df.iloc[loc]
            except Exception:
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
        if not np_isfinite(a) or not np_isfinite(b):
            continue
        if op_fn(a, b):
            return True, effective_stop, None

    return False, effective_stop, None
