#!/usr/bin/env python3
from collections import Counter

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import (
    _compute_indicators,
    MIN_BARS,
    MIN_ENTRY_SCORE,
    DEFAULT_SCORING_WEIGHTS,
    calculate_backtest_quality_score,
)
from simulation.paper_trader import (
    PaperTrader,
    MAX_POSITIONS,
    POSITION_FRACTION,
    SECTOR_CAP,
    MAX_RISK_PER_TRADE,
)


def _build_candidates(pt, data_dict, spy_df=None, vix_df=None):
    candidates = []
    pre_reject = {}

    for sym, df in data_dict.items():
        if df is None or df.empty:
            pre_reject[sym] = "No data"
            continue
        try:
            df_ind = _compute_indicators(df.copy(), spy_df=spy_df)
        except Exception as exc:
            pre_reject[sym] = f"Indicator error: {exc}"
            continue

        if vix_df is not None:
            df_ind["vix"] = vix_df["close"].reindex(df_ind.index).ffill().fillna(20.0)
        else:
            df_ind["vix"] = 20.0

        if len(df_ind) <= MIN_BARS:
            pre_reject[sym] = f"Insufficient bars {len(df_ind)} <= MIN_BARS {MIN_BARS}"
            continue

        signal_idx = len(df_ind) - 2
        if signal_idx < MIN_BARS:
            pre_reject[sym] = f"Signal idx {signal_idx} < MIN_BARS {MIN_BARS}"
            continue

        reason_for_sym = None
        for strat in pt.strategies:
            if not strat.entry(df_ind, signal_idx):
                reason_for_sym = "No entry signal"
                continue

            row_prev = df_ind.iloc[signal_idx]
            raw_score = calculate_backtest_quality_score(
                row_prev,
                strat.name,
                weights=DEFAULT_SCORING_WEIGHTS,
            )
            score = raw_score * 1.3 if "wealth" in strat.name.lower() else raw_score
            if score < MIN_ENTRY_SCORE:
                reason_for_sym = f"Score {score:.1f} < {MIN_ENTRY_SCORE:.1f}"
                continue

            signal_close = float(row_prev["close"])
            atr = float(row_prev.get("atr14", signal_close * 0.02))
            stop_mult = float(getattr(strat, "params", {}).get("stop_loss_atr", 3.0))
            stop_price = signal_close - (atr * stop_mult)

            candidates.append(
                {
                    "symbol": sym,
                    "price": signal_close,
                    "stop": stop_price,
                    "score": score,
                    "strategy": strat.name,
                    "spy_regime": int(row_prev.get("spy_regime", 0) or 0),
                }
            )

        if sym not in pre_reject and not any(c["symbol"] == sym for c in candidates):
            pre_reject[sym] = reason_for_sym or "No entry signal"

    return candidates, pre_reject


def _simulate_governor(pt, candidates):
    accepted = set()
    rejected = {}

    portfolio_syms = set(pt.portfolio.keys())
    pending_orders = list(pt.state.get("pending_orders", []))
    pending_syms = set(o.get("symbol") for o in pending_orders if o.get("symbol"))
    pending_committed = sum(o.get("committed_cash", 0.0) for o in pending_orders)
    available_cash = pt.state.get("cash", 0.0) - pending_committed
    sector_exposure = pt._current_sector_exposure()

    current_equity = pt.state.get("cash", 0.0) + sum(sector_exposure.values())

    sorted_candidates = sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
    for cand in sorted_candidates:
        sym = cand["symbol"]
        if sym in accepted:
            continue

        if len(portfolio_syms) + len(pending_syms) >= MAX_POSITIONS:
            rejected.setdefault(sym, f"All {MAX_POSITIONS} slots full")
            continue
        if sym in portfolio_syms:
            rejected.setdefault(sym, "Already held")
            continue
        if sym in pending_syms:
            rejected.setdefault(sym, "Already pending")
            continue

        trade_val = current_equity * POSITION_FRACTION
        if trade_val <= 0 or available_cash < trade_val:
            rejected.setdefault(sym, f"Insufficient cash for position ${trade_val:,.2f}")
            continue

        price = cand["price"]
        stop_price = cand["stop"]
        risk_per_share = price - stop_price
        if risk_per_share <= 0:
            rejected.setdefault(sym, "Invalid stop (stop >= price)")
            continue

        min_risk_distance = price * 0.01
        if risk_per_share < min_risk_distance:
            risk_pct = (risk_per_share / price) * 100 if price > 0 else 0.0
            rejected.setdefault(sym, f"Stop too tight ({risk_pct:.2f}% < 1.0%)")
            continue

        max_risk_distance = price * 0.20
        if risk_per_share > max_risk_distance:
            rejected.setdefault(sym, f"Stop too wide ({(risk_per_share / price) * 100:.2f}% > 20%)")
            continue

        risk_per_trade = current_equity * MAX_RISK_PER_TRADE
        shares = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
        if shares < 1:
            rejected.setdefault(sym, "Position too small (< 1 share)")
            continue

        max_shares_by_value = int((current_equity * POSITION_FRACTION) / price) if price > 0 else 0
        shares = min(shares, max_shares_by_value)
        if shares < 1:
            rejected.setdefault(sym, "Position too small after value cap")
            continue

        position_val = shares * price
        sec = pt._resolve_sector(sym)
        projected_exp = (
            (sector_exposure.get(sec, 0.0) + position_val) / current_equity
            if current_equity > 0
            else 1.0
        )
        if projected_exp > SECTOR_CAP:
            rejected.setdefault(
                sym,
                f"Sector cap {sec} {projected_exp*100:.0f}% > {SECTOR_CAP*100:.0f}%",
            )
            continue

        accepted.add(sym)
        pending_syms.add(sym)
        pending_orders.append({"symbol": sym, "committed_cash": position_val})
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + position_val
        available_cash -= position_val

    return accepted, rejected


def main():
    pt = PaperTrader()
    tickers = get_index_symbols("S&P 1500")
    print(f"Loaded {len(tickers)} tickers", flush=True)
    print(f"Using MIN_ENTRY_SCORE={MIN_ENTRY_SCORE}", flush=True)

    base_days = 400
    data_dict = fetch_data_pack(tickers, days=base_days)
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=base_days + 200) or {}
    spy_df = g_data.get("SPY")
    vix_df = g_data.get("$VIX")
    if vix_df is None:
        vix_df = g_data.get("VIX")

    candidates, pre_reject = _build_candidates(pt, data_dict, spy_df=spy_df, vix_df=vix_df)
    regime_map = {cand.get("symbol"): cand.get("spy_regime") for cand in candidates}
    accepted, governor_reject = _simulate_governor(pt, candidates)

    reason_counts = Counter()
    for sym in tickers:
        if sym in accepted:
            reason = "Accepted - Queued"
        elif sym in governor_reject:
            reason = f"Rejected - {governor_reject[sym]}"
        else:
            reason = f"Rejected - {pre_reject.get(sym, 'No candidate generated')}"
        reason_counts[reason] += 1
        display_reason = reason
        if sym in accepted:
            display_reason = f"{reason} | Regime: {regime_map.get(sym, 'n/a')}"
        print(f"{sym}: {display_reason}", flush=True)

    print("\nSummary (counts):", flush=True)
    for reason, count in reason_counts.most_common():
        print(f"{count:>5} | {reason}", flush=True)


if __name__ == "__main__":
    main()
