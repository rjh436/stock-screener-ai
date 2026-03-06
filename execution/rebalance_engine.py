from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from execution.portfolio_constraints import (
    enforce_industry_cap,
    enforce_position_cap,
    enforce_sector_cap,
    enforce_turnover_budget,
)


def _coerce_score_series(ranked_scores: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(ranked_scores, pd.DataFrame):
        if "score" in ranked_scores.columns:
            series = ranked_scores["score"]
        elif "momentum_rank" in ranked_scores.columns:
            series = ranked_scores["momentum_rank"]
        elif "combined_rank" in ranked_scores.columns:
            series = ranked_scores["combined_rank"]
        else:
            series = ranked_scores.iloc[:, 0]
    else:
        series = ranked_scores

    series = pd.to_numeric(series, errors="coerce")
    series.index = series.index.astype(str)
    return series.dropna().sort_values(ascending=False)


def apply_hold_buffer(
    existing_symbols: Iterable[str],
    ranked_scores: pd.Series | pd.DataFrame,
    target_count: int,
    hold_buffer_mult: float = 1.25,
) -> List[str]:
    if target_count <= 0:
        return []

    ranked = _coerce_score_series(ranked_scores)
    if ranked.empty:
        return []

    existing = {str(s) for s in (existing_symbols or [])}
    threshold = max(int(target_count), int(np.ceil(float(target_count) * float(max(1.0, hold_buffer_mult)))))
    top_buffer = list(ranked.head(threshold).index)
    return [sym for sym in top_buffer if sym in existing]


def select_target_portfolio(
    ranked_scores: pd.Series | pd.DataFrame,
    target_count: int,
    existing_symbols: Optional[Iterable[str]] = None,
    hold_buffer_mult: float = 1.25,
) -> List[str]:
    if target_count <= 0:
        return []

    ranked = _coerce_score_series(ranked_scores)
    if ranked.empty:
        return []

    top_target = list(ranked.head(int(target_count)).index)
    if not existing_symbols:
        return top_target

    keepers = apply_hold_buffer(existing_symbols, ranked, int(target_count), float(hold_buffer_mult))
    if not keepers:
        return top_target

    keep_set = set(keepers)
    selected: List[str] = []
    for sym in keepers:
        if sym not in selected:
            selected.append(sym)

    for sym in top_target:
        if sym in keep_set:
            continue
        selected.append(sym)
        if len(selected) >= int(target_count):
            break

    if len(selected) < int(target_count):
        for sym in ranked.index:
            if sym in selected:
                continue
            selected.append(sym)
            if len(selected) >= int(target_count):
                break

    return selected[: int(target_count)]


def _normalize_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    clean: Dict[str, float] = {}
    for raw_key, raw_val in dict(weights or {}).items():
        key = str(raw_key)
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val) or val <= 0:
            continue
        clean[key] = val
    total = float(sum(clean.values()))
    if total <= 0:
        return {}
    return {k: (v / total) for k, v in clean.items()}


def _sanitize_target_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """
    Accept explicit portfolio weights that may sum to <= 1.0 (cash residual allowed).
    If the sum exceeds 1.0, scale down proportionally.
    """
    clean: Dict[str, float] = {}
    for raw_key, raw_val in dict(weights or {}).items():
        key = str(raw_key)
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val) or val <= 0:
            continue
        clean[key] = val

    total = float(sum(clean.values()))
    if total <= 0:
        return {}
    if total > 1.0 + 1e-12:
        return {k: (v / total) for k, v in clean.items()}
    return clean


def _build_selected_target_weights(
    selected: Iterable[str],
    ranked_scores: Mapping[str, float] | pd.Series,
    *,
    conviction_weighted: bool,
    conviction_power: float = 1.0,
) -> Dict[str, float]:
    chosen = [str(s) for s in selected if str(s)]
    if not chosen:
        return {}

    n = len(chosen)
    if not conviction_weighted or n <= 1:
        equal = 1.0 / float(n)
        return {sym: equal for sym in chosen}

    ranked = pd.to_numeric(pd.Series(ranked_scores), errors="coerce")
    ranked = ranked.reindex(chosen).dropna()
    if ranked.empty:
        equal = 1.0 / float(n)
        return {sym: equal for sym in chosen}

    shifted = ranked - float(ranked.min()) + 1e-6
    try:
        power = float(conviction_power)
    except Exception:
        power = 1.0
    if not np.isfinite(power) or power <= 0:
        power = 1.0
    if abs(power - 1.0) > 1e-9:
        shifted = shifted.pow(power)

    total = float(shifted.sum())
    if not np.isfinite(total) or total <= 0:
        equal = 1.0 / float(n)
        return {sym: equal for sym in chosen}

    base = (1.0 - (len(ranked) / float(n))) / float(n) if len(ranked) < n else 0.0
    out: Dict[str, float] = {}
    for sym in chosen:
        if sym in shifted.index:
            out[sym] = float(shifted.loc[sym] / total)
        else:
            out[sym] = max(0.0, base)
    out = _normalize_weights(out)
    if out:
        return out

    equal = 1.0 / float(n)
    return {sym: equal for sym in chosen}


def _coerce_weight_row(frame: pd.DataFrame, dt: pd.Timestamp) -> Dict[str, float]:
    if frame is None or frame.empty:
        return {}
    hist = frame.loc[frame.index <= dt]
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    weights = pd.to_numeric(row, errors="coerce").dropna()
    return _sanitize_target_weights(weights.to_dict())


def _turnover(prev_weights: Mapping[str, float], next_weights: Mapping[str, float]) -> float:
    keys = set(prev_weights.keys()) | set(next_weights.keys())
    return 0.5 * sum(abs(float(next_weights.get(k, 0.0)) - float(prev_weights.get(k, 0.0))) for k in keys)


def _portfolio_state(
    positions: Mapping[str, int],
    cash: float,
    prices: Mapping[str, float],
) -> Dict[str, object]:
    equity = float(cash)
    notionals: Dict[str, float] = {}
    for sym, shares in dict(positions or {}).items():
        try:
            qty = int(shares)
        except Exception:
            continue
        if qty == 0:
            continue
        px = prices.get(sym)
        if px is None:
            continue
        try:
            px_val = float(px)
        except Exception:
            continue
        if not np.isfinite(px_val) or px_val <= 0:
            continue
        val = float(qty) * px_val
        if val <= 0:
            continue
        notionals[str(sym)] = val
        equity += val

    if equity <= 0:
        return {"equity": 0.0, "weights": {}}

    weights = {sym: (val / equity) for sym, val in notionals.items() if val > 0}
    total_w = float(sum(weights.values()))
    if total_w > 1.0 + 1e-12:
        weights = {k: (v / total_w) for k, v in weights.items()}
    return {"equity": float(equity), "weights": weights}


def generate_orders_from_target(
    current_positions: Mapping[str, int],
    target_weights: Mapping[str, float],
    prices: Mapping[str, float],
    cash: float,
    transaction_cost_bps: float = 0.0,
) -> Dict[str, object]:
    tx_rate = float(max(0.0, transaction_cost_bps)) / 10_000.0
    positions = {str(k): int(v) for k, v in dict(current_positions or {}).items() if int(v) != 0}
    target_w = _sanitize_target_weights(target_weights)

    px = {str(k): float(v) for k, v in dict(prices or {}).items() if np.isfinite(v) and float(v) > 0}

    current_value = 0.0
    for sym, shares in positions.items():
        price = px.get(sym)
        if price is None:
            continue
        current_value += float(shares) * price

    total_equity = float(cash) + current_value
    if total_equity <= 0:
        return {"orders": [], "positions": positions, "cash": float(cash), "traded_notional": 0.0}

    desired_shares: Dict[str, int] = {}
    for sym, wt in target_w.items():
        price = px.get(sym)
        if price is None or price <= 0:
            continue
        desired_shares[sym] = int(np.floor((wt * total_equity) / price))

    orders: List[Dict[str, object]] = []
    traded_notional = 0.0
    cash_now = float(cash)

    # Sells first to free buying power.
    for sym in sorted(set(positions.keys()) | set(desired_shares.keys())):
        curr = int(positions.get(sym, 0))
        target = int(desired_shares.get(sym, 0))
        delta = target - curr
        if delta >= 0:
            continue

        price = px.get(sym)
        if price is None or price <= 0:
            continue

        qty = min(curr, -delta)
        if qty <= 0:
            continue

        gross = qty * price
        fee = gross * tx_rate
        cash_now += (gross - fee)
        positions[sym] = curr - qty
        if positions[sym] <= 0:
            positions.pop(sym, None)

        traded_notional += gross
        orders.append({"symbol": sym, "side": "SELL", "shares": qty, "price": price, "fee": fee})

    # Buys after sells.
    buy_candidates = []
    for sym in set(positions.keys()) | set(desired_shares.keys()):
        curr = int(positions.get(sym, 0))
        target = int(desired_shares.get(sym, 0))
        delta = target - curr
        if delta > 0:
            buy_candidates.append((sym, delta))

    buy_candidates.sort(key=lambda item: item[1], reverse=True)
    for sym, delta in buy_candidates:
        price = px.get(sym)
        if price is None or price <= 0:
            continue
        max_affordable = int(np.floor(cash_now / (price * (1.0 + tx_rate))))
        qty = min(int(delta), max_affordable)
        if qty <= 0:
            continue

        gross = qty * price
        fee = gross * tx_rate
        cash_now -= (gross + fee)
        positions[sym] = int(positions.get(sym, 0)) + qty

        traded_notional += gross
        orders.append({"symbol": sym, "side": "BUY", "shares": qty, "price": price, "fee": fee})

    return {
        "orders": orders,
        "positions": positions,
        "cash": float(cash_now),
        "traded_notional": float(traded_notional),
    }


def run_periodic_rebalance(
    prices: pd.DataFrame,
    ranked_scores: pd.DataFrame,
    *,
    execution_prices: Optional[pd.DataFrame] = None,
    rebalance_freq: str = "M",
    target_count: int = 20,
    hold_buffer_mult: float = 1.25,
    position_cap: float = 1.0,
    sector_map: Optional[Mapping[str, str]] = None,
    sector_cap: float = 1.0,
    industry_map: Optional[Mapping[str, str]] = None,
    industry_cap: float = 1.0,
    turnover_budget: float = 1.0,
    transaction_cost_bps: float = 2.0,
    start_cash: float = 100000.0,
    target_weights_by_date: Optional[pd.DataFrame] = None,
    risk_scalar_by_date: Optional[pd.Series] = None,
    hard_stop_pct: Optional[float] = None,
    trend_ma_days: Optional[int] = None,
    time_stop_days: Optional[int] = None,
    execution_lag_days: int = 1,
    max_stale_price_days: Optional[int] = 5,
    conviction_weighted: bool = False,
    conviction_power: float = 1.0,
) -> Dict[str, object]:
    if prices is None or prices.empty:
        return {
            "final_value": float(start_cash),
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "equity_curve": [],
            "rebalance_log": [],
        }

    px = prices.copy().sort_index()
    scores = ranked_scores.copy().sort_index()
    px.index = pd.to_datetime(px.index)
    scores.index = pd.to_datetime(scores.index)
    exec_px = execution_prices.copy().sort_index() if execution_prices is not None else px
    exec_px.index = pd.to_datetime(exec_px.index)
    target_schedule = None
    if target_weights_by_date is not None:
        target_schedule = target_weights_by_date.copy().sort_index()
        target_schedule.index = pd.to_datetime(target_schedule.index)
    risk_scalars = None
    if risk_scalar_by_date is not None:
        risk_scalars = pd.to_numeric(risk_scalar_by_date, errors="coerce")
        if isinstance(risk_scalars, pd.Series):
            risk_scalars = risk_scalars.sort_index()
            risk_scalars.index = pd.to_datetime(risk_scalars.index)
    tx_rate = float(max(0.0, transaction_cost_bps)) / 10_000.0
    hard_stop = None
    if hard_stop_pct is not None:
        try:
            hs = float(hard_stop_pct)
            if np.isfinite(hs) and hs > 0:
                hard_stop = hs
        except Exception:
            hard_stop = None

    trend_ma: Optional[pd.DataFrame] = None
    if trend_ma_days is not None:
        try:
            tma = int(trend_ma_days)
            if tma > 1:
                trend_ma = px.rolling(window=tma, min_periods=max(20, tma // 2)).mean()
        except Exception:
            trend_ma = None

    max_hold_days: Optional[int] = None
    if time_stop_days is not None:
        try:
            td = int(time_stop_days)
            if td > 0:
                max_hold_days = td
        except Exception:
            max_hold_days = None
    stale_limit_days: Optional[int] = None
    if max_stale_price_days is not None:
        try:
            sd = int(max_stale_price_days)
            if sd > 0:
                stale_limit_days = sd
        except Exception:
            stale_limit_days = None

    # Restrict to dates with available prices; score snapshots are pulled on or before date.
    trade_dates = list(pd.Index(px.index).unique())
    if not trade_dates:
        return {
            "final_value": float(start_cash),
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "equity_curve": [],
            "rebalance_log": [],
        }

    freq = str(rebalance_freq or "M").upper()
    if freq not in {"D", "W", "M", "Q"}:
        raise ValueError(f"Unsupported rebalance_freq '{rebalance_freq}'. Expected 'D', 'W', 'M', or 'Q'.")

    if freq == "D":
        raw_signal_dates = list(trade_dates)
    else:
        period_freq = "W-FRI" if freq == "W" else freq
        raw_signal_dates = list(px.groupby(px.index.to_period(period_freq)).tail(1).index)
    lag_days = max(0, int(execution_lag_days))
    trade_dates_index = list(pd.Index(trade_dates))
    date_pos = {pd.Timestamp(d): i for i, d in enumerate(trade_dates_index)}
    rebalance_events: Dict[pd.Timestamp, pd.Timestamp] = {}
    for raw_dt in raw_signal_dates:
        signal_dt = pd.Timestamp(raw_dt)
        pos = date_pos.get(signal_dt)
        if pos is None:
            continue
        exec_pos = pos + lag_days
        if exec_pos >= len(trade_dates_index):
            continue
        exec_dt = pd.Timestamp(trade_dates_index[exec_pos])
        rebalance_events[exec_dt] = signal_dt

    cash = float(start_cash)
    positions: Dict[str, int] = {}
    last_prices: Dict[str, float] = {}
    position_meta: Dict[str, Dict[str, object]] = {}
    last_fresh_seen: Dict[str, pd.Timestamp] = {}

    equity_curve: List[Dict[str, object]] = []
    rebalance_log: List[Dict[str, object]] = []
    total_orders = 0
    exposure_profile: List[float] = []
    annual_turnover_target: Dict[str, float] = {}
    annual_turnover_realized: Dict[str, float] = {}
    same_day_signal_exec = 0

    for dt in trade_dates:
        dt = pd.Timestamp(dt)
        row_prices = pd.to_numeric(px.loc[dt], errors="coerce")
        row_exec_prices = pd.Series(dtype=float)
        if dt in exec_px.index:
            row_exec_prices = pd.to_numeric(exec_px.loc[dt], errors="coerce")
        tradable_prices = {
            str(sym): float(val)
            for sym, val in row_prices.items()
            if np.isfinite(val) and float(val) > 0
        }
        tradable_exec_prices = {
            str(sym): float(val)
            for sym, val in row_exec_prices.items()
            if np.isfinite(val) and float(val) > 0
        }
        if tradable_prices:
            last_prices.update(tradable_prices)
            for sym in tradable_prices.keys():
                last_fresh_seen[str(sym)] = dt

        # Daily stop logic for open positions before periodic rebalance.
        stop_exits: List[Dict[str, object]] = []
        if positions and (
            hard_stop is not None
            or trend_ma is not None
            or max_hold_days is not None
            or stale_limit_days is not None
        ):
            for sym, shares in list(positions.items()):
                fresh_px = row_prices.get(sym)
                fresh_valid = fresh_px is not None and np.isfinite(fresh_px) and float(fresh_px) > 0
                price = float(fresh_px) if fresh_valid else last_prices.get(sym)
                if price is None or not np.isfinite(price) or float(price) <= 0:
                    continue
                px_val = float(price)

                meta = position_meta.get(sym, {})
                entry_price = meta.get("entry_price")
                entry_dt = meta.get("entry_date")

                reason = None
                if hard_stop is not None and isinstance(entry_price, (int, float)) and np.isfinite(float(entry_price)):
                    stop_px = float(entry_price) * (1.0 - float(hard_stop))
                    if px_val <= stop_px:
                        reason = "hard_stop"

                if reason is None and trend_ma is not None and sym in trend_ma.columns:
                    ma_val = trend_ma.at[dt, sym]
                    if np.isfinite(ma_val) and px_val < float(ma_val):
                        reason = "trend_stop"

                if reason is None and max_hold_days is not None and isinstance(entry_dt, pd.Timestamp):
                    held_days = int((dt - entry_dt).days)
                    if held_days >= int(max_hold_days):
                        reason = "time_stop"
                stale_days = None
                if reason is None and stale_limit_days is not None and not fresh_valid:
                    last_seen = last_fresh_seen.get(str(sym))
                    if isinstance(last_seen, pd.Timestamp):
                        stale_days = int((dt - last_seen).days)
                        if stale_days >= int(stale_limit_days):
                            last_px = last_prices.get(sym)
                            if last_px is not None and np.isfinite(last_px) and float(last_px) > 0:
                                px_val = float(last_px)
                                reason = "stale_price_exit"

                if reason is None:
                    continue

                qty = int(shares)
                gross = float(qty) * px_val
                fee = gross * tx_rate
                cash += (gross - fee)
                positions.pop(sym, None)
                position_meta.pop(sym, None)
                total_orders += 1
                stop_exits.append(
                    {
                        "symbol": sym,
                        "side": "SELL",
                        "shares": qty,
                        "price": px_val,
                        "fee": fee,
                        "reason": reason,
                        "stale_days": stale_days,
                    }
                )

        if dt in rebalance_events:
            signal_dt = rebalance_events[dt]
            if signal_dt == dt:
                same_day_signal_exec += 1
            if target_schedule is not None:
                target_weights = _coerce_weight_row(target_schedule, signal_dt)
                selected = sorted(target_weights.keys())
            else:
                signal_prices = pd.to_numeric(px.loc[signal_dt], errors="coerce")
                score_hist = scores.loc[scores.index <= signal_dt]
                if not score_hist.empty:
                    score_row = score_hist.iloc[-1]
                    if isinstance(score_row, pd.Series):
                        ranked = pd.to_numeric(score_row, errors="coerce")
                    else:
                        ranked = pd.Series(dtype=float)
                    ranked = ranked.dropna()
                    ranked = ranked.loc[ranked.index.intersection(signal_prices.index)]
                    ranked = ranked.loc[pd.to_numeric(signal_prices[ranked.index], errors="coerce") > 0]

                    selected = select_target_portfolio(
                        ranked,
                        target_count=int(target_count),
                        existing_symbols=positions.keys(),
                        hold_buffer_mult=float(hold_buffer_mult),
                    )

                    if selected:
                        target_weights = _build_selected_target_weights(
                            selected,
                            ranked,
                            conviction_weighted=bool(conviction_weighted),
                            conviction_power=float(conviction_power),
                        )
                    else:
                        target_weights = {}
                else:
                    selected = []
                    target_weights = {}

            state = _portfolio_state(positions, cash, last_prices)
            actual_prev_weights = dict(state.get("weights") or {})
            pre_trade_equity = float(state.get("equity") or 0.0)

            target_weights = enforce_position_cap(target_weights, float(position_cap))
            if sector_map:
                target_weights = enforce_sector_cap(target_weights, sector_map, float(sector_cap))
            if industry_map:
                target_weights = enforce_industry_cap(target_weights, industry_map, float(industry_cap))

            if risk_scalars is not None and target_weights:
                hist = risk_scalars.loc[risk_scalars.index <= signal_dt]
                risk_scalar = float(hist.iloc[-1]) if not hist.empty else 1.0
                if not np.isfinite(risk_scalar):
                    risk_scalar = 1.0
                risk_scalar = float(np.clip(risk_scalar, 0.0, 1.0))
                if risk_scalar <= 0.0:
                    target_weights = {}
                elif risk_scalar < 1.0:
                    target_weights = {k: (v * risk_scalar) for k, v in target_weights.items()}

            target_weights = enforce_turnover_budget(actual_prev_weights, target_weights, float(turnover_budget))
            target_turnover_val = float(_turnover(actual_prev_weights, target_weights))

            execution_price_map = tradable_exec_prices if tradable_exec_prices else (tradable_prices if tradable_prices else last_prices)
            order_res = generate_orders_from_target(
                current_positions=positions,
                target_weights=target_weights,
                prices=execution_price_map,
                cash=cash,
                transaction_cost_bps=float(transaction_cost_bps),
            )
            prev_positions = dict(positions)
            positions = dict(order_res["positions"])
            cash = float(order_res["cash"])
            orders = list(order_res["orders"])
            total_orders += len(orders)
            traded_notional = float(order_res.get("traded_notional", 0.0) or 0.0)
            realized_turnover_val = (
                float(traded_notional / (2.0 * pre_trade_equity))
                if pre_trade_equity > 0.0 and traded_notional > 0.0
                else 0.0
            )

            for order in orders:
                sym = str(order.get("symbol", ""))
                side = str(order.get("side", "")).upper()
                if side == "BUY":
                    if int(prev_positions.get(sym, 0)) <= 0 and int(positions.get(sym, 0)) > 0:
                        price = order.get("price")
                        if isinstance(price, (int, float)) and np.isfinite(float(price)):
                            position_meta[sym] = {"entry_date": dt, "entry_price": float(price)}
                elif side == "SELL":
                    if int(positions.get(sym, 0)) <= 0:
                        position_meta.pop(sym, None)

            year_key = str(pd.Timestamp(dt).year)
            annual_turnover_target[year_key] = float(annual_turnover_target.get(year_key, 0.0) + target_turnover_val)
            annual_turnover_realized[year_key] = float(
                annual_turnover_realized.get(year_key, 0.0) + realized_turnover_val
            )
            rebalance_log.append(
                {
                    "date": dt.isoformat(),
                    "signal_date": signal_dt.isoformat(),
                    "selected": list(selected),
                    "weights": dict(target_weights),
                    "target_turnover": target_turnover_val,
                    "realized_turnover": realized_turnover_val,
                    "orders": orders,
                    "stop_exits": stop_exits,
                }
            )

        mtm = float(cash)
        gross = 0.0
        for sym, shares in positions.items():
            price = row_prices.get(sym)
            if price is None or not np.isfinite(price) or float(price) <= 0:
                price = last_prices.get(sym)
            if price is None:
                continue
            try:
                px_val = float(price)
            except Exception:
                continue
            if not np.isfinite(px_val) or px_val <= 0:
                continue
            notional = float(shares) * px_val
            gross += notional
            mtm += notional

        exposure = (gross / mtm) if mtm > 0 else 0.0
        exposure = float(max(0.0, exposure))
        exposure_profile.append(exposure)

        equity_curve.append({"Date": dt.isoformat(), "Equity": mtm})

    if not equity_curve:
        return {
            "final_value": float(start_cash),
            "cagr": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": int(total_orders),
            "equity_curve": [],
            "rebalance_log": rebalance_log,
        }

    eq = pd.Series([float(x["Equity"]) for x in equity_curve])
    peaks = eq.cummax()
    dd = ((eq - peaks) / peaks).min() if len(eq) else 0.0
    max_dd = abs(float(dd)) if np.isfinite(dd) else 0.0

    total_days = (pd.Timestamp(equity_curve[-1]["Date"]) - pd.Timestamp(equity_curve[0]["Date"])) .days
    years = max(0.1, float(total_days) / 365.25)
    final_value = float(eq.iloc[-1])
    cagr = ((final_value / float(start_cash)) ** (1.0 / years) - 1.0) if start_cash > 0 else 0.0
    max_gross = float(max(exposure_profile)) if exposure_profile else 0.0
    annual_turnover_target_pct = {k: float(v * 100.0) for k, v in annual_turnover_target.items()}
    annual_turnover_realized_pct = {k: float(v * 100.0) for k, v in annual_turnover_realized.items()}
    target_turnover_years = list(annual_turnover_target_pct.values())
    realized_turnover_years = list(annual_turnover_realized_pct.values())
    avg_turnover_target_pct = float(np.mean(target_turnover_years)) if target_turnover_years else 0.0
    avg_turnover_realized_pct = float(np.mean(realized_turnover_years)) if realized_turnover_years else 0.0

    return {
        "final_value": final_value,
        "cagr": float(cagr),
        "max_drawdown_pct": float(max_dd),
        "total_trades": int(total_orders),
        "equity_curve": equity_curve,
        "rebalance_log": rebalance_log,
        "annual_turnover_target_pct": annual_turnover_target_pct,
        "annual_turnover_realized_pct": annual_turnover_realized_pct,
        "annual_turnover_pct": annual_turnover_realized_pct,
        "avg_annual_turnover_target_pct": avg_turnover_target_pct,
        "avg_annual_turnover_realized_pct": avg_turnover_realized_pct,
        "avg_annual_turnover_pct": avg_turnover_realized_pct,
        "audit_report": {
            "same_day_open_entries": int(same_day_signal_exec),
            "max_gross_exposure_pct": max_gross,
            "exposure_profile": exposure_profile,
            "execution_lag_days": int(lag_days),
        },
    }
