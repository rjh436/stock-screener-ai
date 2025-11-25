import math
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD, SMAIndicator

from strategies.base import BaseStrategy

MIN_BARS = 200


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator set used across screener/backtest/optimizer."""
    df = df.sort_index().copy()
    df.columns = df.columns.str.lower()

    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Averages
    for p in [5, 10, 20, 50, 100, 200]:
        df[f"sma{p}"] = df["close"].rolling(p).mean()
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    # Volatility (Bollinger + ATR)
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + (bb_std * 2)
    df["bb_lower"] = bb_mid - (bb_std * 2)
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / bb_mid).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr14_ma20"] = df["atr14"].rolling(20).mean()

    # Extremes (shifted to avoid look-ahead)
    df["highest20"] = df["high"].rolling(20).max()
    df["highest55"] = df["high"].rolling(55).max()
    df["lowest5"] = df["low"].rolling(5).min()

    df["highest20_1"] = df["highest20"].shift(1)
    df["highest55_1"] = df["highest55"].shift(1)
    df["lowest5_1"] = df["lowest5"].shift(1)

    # Oscillators
    def calc_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df["rsi2"] = calc_rsi(df["close"], 2)
    df["rsi3"] = calc_rsi(df["close"], 3)
    df["rsi14"] = calc_rsi(df["close"], 14)
    df["crsi"] = (df["rsi3"] + df["rsi2"] + (100 - df["rsi14"])) / 3

    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_hist"] = macd.macd_diff()

    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
    df["roc"] = ROCIndicator(df["close"], window=12).roc()
    df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    return df


def _empty_result(name: str, start_cash: float, params=None):
    return {
        "strategy": name,
        "final_value": start_cash,
        "total_trades": 0,
        "hit_rate": 0.0,
        "sharpe": 0.0,
        "cagr": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0,
        "avg_days_held": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "Score": 0.0,
        "params": params or {},
        "trades_list": [],
    }


def _safe_reindex_vix(vix_df: Optional[pd.DataFrame], target_index: pd.Index) -> pd.Series:
    if vix_df is None or "close" not in vix_df:
        return pd.Series(20.0, index=target_index)
    clean = vix_df.copy()
    if isinstance(clean.index, pd.DatetimeIndex) and clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    return clean["close"].reindex(target_index, method="ffill").fillna(20.0)


def run_backtest(
    strategy: BaseStrategy,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    symbols = list(data_dict.keys())
    if symbol_universe:
        symbols = [s for s in symbols if s in symbol_universe]

    enriched: Dict[str, pd.DataFrame] = {}
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        src_df = data_dict.get(sym)
        if src_df is None or src_df.empty:
            continue
        df_in = src_df.copy()
        if isinstance(df_in.index, pd.DatetimeIndex) and df_in.index.tz is not None:
            df_in.index = df_in.index.tz_localize(None)

        df = _compute_indicators(df_in)

        vix_series = _safe_reindex_vix(vix_df, df.index)
        df["vix"] = vix_series

        if start_date:
            start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
            df = df[df.index >= start_dt]

        if len(df) > MIN_BARS:
            enriched[sym] = df

    if not enriched:
        return _empty_result(strategy.name, start_cash, getattr(strategy, "params", {}))

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions: Dict[str, Dict] = {}
    equity_curve: List[float] = []
    trade_pnls: List[float] = []
    trade_returns: List[float] = []
    trade_durations: List[int] = []
    trades_list: List[Dict] = []

    pos_fraction = float(getattr(strategy, "params", {}).get("pos_size", 0.20))
    if pos_fraction <= 0 or pos_fraction > 1:
        pos_fraction = 0.20
    max_positions = max(1, int(1.0 / pos_fraction))

    for current_dt in all_dates:
        # Exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index:
                continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]

            if strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"]):
                row = df.iloc[i]
                stop_hit = row.get("low", np.inf) <= pos["stop_price"]
                exit_px = pos["stop_price"] if stop_hit else row.get("close", pos["entry_price"])
                if stop_hit and row.get("open", pos["stop_price"]) < pos["stop_price"]:
                    exit_px = row.get("open", pos["stop_price"])
                if exit_px > (pos["entry_price"] * 5.0):
                    exit_px = pos["entry_price"]

                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100

                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trade_returns.append(pct)
                trade_durations.append(i - pos["entry_i"])
                trades_list.append(
                    {
                        "Symbol": sym,
                        "Entry Date": str(pos["entry_dt"].date()),
                        "Exit Date": str(current_dt.date()),
                        "Entry": round(pos["entry_price"], 2),
                        "Exit": round(exit_px, 2),
                        "PnL": round(pnl, 2),
                        "Return %": round(pct, 2),
                    }
                )
                del positions[sym]

        # Equity snapshot
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        # Entries (use prior bar to avoid look-ahead)
        if len(positions) < max_positions and cash > 0:
            target_size = equity * pos_fraction
            for sym, df in enriched.items():
                if sym in positions or current_dt not in df.index:
                    continue
                i = df.index.get_loc(current_dt)
                if i == 0 or i - 1 < MIN_BARS:
                    continue

                signal_idx = i - 1
                entry_signal = strategy.entry(df, signal_idx)
                if not entry_signal:
                    continue

                px = float(df.iloc[i]["open"])
                if np.isnan(px) or px < 2.0:
                    continue

                stop = float(entry_signal["stop_price"])
                shares = int(target_size / px)
                cost = shares * px
                if shares > 0 and cash >= cost:
                    cash -= cost
                    positions[sym] = {
                        "shares": shares,
                        "entry_price": px,
                        "stop_price": stop,
                        "entry_i": i,
                        "entry_dt": current_dt,
                    }

    final_val = equity_curve[-1] if equity_curve else start_cash
    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    avg_profit = float(np.mean(trade_returns)) if trade_returns else 0.0

    gross_profit = sum(t for t in trade_pnls if t > 0)
    gross_loss = abs(sum(t for t in trade_pnls if t < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    avg_win = float(np.mean([t for t in trade_returns if t > 0])) if any(t > 0 for t in trade_returns) else 0.0
    avg_loss = abs(float(np.mean([t for t in trade_returns if t < 0]))) if any(t < 0 for t in trade_returns) else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    days_run = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 1 else 1
    years = max(days_run / 365.0, 1e-6)
    cagr = ((final_val / start_cash) ** (1.0 / years)) - 1.0 if final_val > 0 else 0.0

    eq = pd.Series(equity_curve)
    daily_ret = eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * math.sqrt(252)) if len(daily_ret) > 1 and daily_ret.std() > 0 else 0.0
    roll_max = eq.cummax()
    max_dd = (((eq - roll_max) / roll_max).min() * 100) if len(eq) > 0 else 0.0

    score = (cagr * 200) + (win_rate * 2) + (sharpe * 20) + (avg_profit * 50)

    return {
        "strategy": strategy.name,
        "final_value": final_val,
        "total_trades": trades,
        "hit_rate": win_rate,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown_pct": max_dd,
        "avg_profit_pct": avg_profit,
        "avg_days_held": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "Score": score,
        "trades_list": trades_list,
        "params": getattr(strategy, "params", {}),
    }


def run_compare(
    strategy_names: List[str],
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    use_parallel: bool = True,
    max_workers: int = 8,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
):
    import concurrent.futures
    import json
    import os

    from strategies.generic import GenericStrategy

    gen_strategies: Dict[str, Dict] = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json")), "r") as f:
            for g in json.load(f):
                gen_strategies[g["name"]] = g
    except Exception:
        pass

    strategies = [GenericStrategy(gen_strategies[name]) for name in strategy_names if name in gen_strategies]

    results = []
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat for strat in strategies}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Backtest failed for {futures[future].name}: {e}")
    else:
        for strat in strategies:
            try:
                results.append(run_backtest(strat, data_dict, symbol_universe, start_cash, start_date, global_data))
            except Exception as e:
                print(f"Backtest failed for {strat.name}: {e}")

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df.trade_logs = {r["strategy"]: r.get("trades_list", []) for r in results}
    return df.sort_values("Score", ascending=False)
