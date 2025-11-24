import math
import os
import json
import concurrent.futures
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator, MACD

MIN_BARS_FOR_WARMUP = 200


def _compute_indicators(df: pd.DataFrame, sym: str = "UNKNOWN") -> pd.DataFrame:
    """Compute indicators; print errors instead of failing silently."""
    df = df.sort_index().copy()
    df.columns = df.columns.str.lower()
    try:
        price = df["close"]
        high = df["high"]
        low = df["low"]
    except Exception as e:
        print(f"❌ Error calculating {sym}: missing price columns ({e})")
        return df

    try:
        for p in [5, 10, 20, 50, 100, 200]:
            df[f"sma{p}"] = price.rolling(p).mean()
    except Exception as e:
        print(f"❌ Error calculating {sym}: SMA ({e})")

    try:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
    except Exception as e:
        print(f"❌ Error calculating {sym}: vol_ma20 ({e})")

    try:
        df["highest20"] = high.rolling(20).max()
        df["highest55"] = high.rolling(55).max()
        df["lowest5"] = low.rolling(5).min()
        df["highest20_1"] = df["highest20"].shift(1)
        df["highest55_1"] = df["highest55"].shift(1)
        df["lowest5_1"] = df["lowest5"].shift(1)
    except Exception as e:
        print(f"❌ Error calculating {sym}: extremes ({e})")

    try:
        rolling_mean = price.rolling(20).mean()
        rolling_std = price.rolling(20).std()
        df["bb_upper"] = rolling_mean + (rolling_std * 2)
        df["bb_lower"] = rolling_mean - (rolling_std * 2)
        df["bb_mid"] = rolling_mean
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    except Exception as e:
        print(f"❌ Error calculating {sym}: Bollinger ({e})")

    try:
        tr = pd.concat(
            [high - low, (high - price.shift()).abs(), (low - price.shift()).abs()],
            axis=1,
        ).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr14_ma20"] = df["atr14"].rolling(20).mean()
    except Exception as e:
        print(f"❌ Error calculating {sym}: ATR ({e})")

    try:
        def calc_rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        df["rsi2"] = calc_rsi(price, 2)
        df["rsi3"] = calc_rsi(price, 3)
        df["rsi14"] = calc_rsi(price, 14)
    except Exception as e:
        print(f"❌ Error calculating {sym}: RSI ({e})")

    try:
        adx = ADXIndicator(high, low, price, fillna=True)
        df["adx"] = adx.adx()
        df["plus_di"] = adx.adx_pos()
        df["minus_di"] = adx.adx_neg()

        macd = MACD(price, fillna=True)
        df["macd"] = macd.macd()
        df["macd_hist"] = macd.macd_diff()
    except Exception as e:
        print(f"❌ Error calculating {sym}: trend ({e})")

    return df


def _empty_result(name: str, start_cash: float, params: Optional[Dict] = None) -> Dict:
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
    }


def run_backtest(
    strategy,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[str] = None,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    symbols = list(data_dict.keys())
    if symbol_universe:
        symbols = [s for s in symbols if s in symbol_universe]

    enriched: Dict[str, pd.DataFrame] = {}
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        src = data_dict.get(sym)
        if src is None or src.empty:
            continue
        try:
            df = _compute_indicators(src.copy(), sym)
            if vix_df is not None:
                df["vix"] = vix_df["close"].reindex(df.index, method="ffill").fillna(20.0)
            else:
                df["vix"] = 20.0

            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                df = df[df.index >= start_dt]

            if len(df) > MIN_BARS_FOR_WARMUP:
                enriched[sym] = df
        except Exception as e:
            print(f"❌ Error preparing {sym}: {e}")
            continue

    if not enriched:
        return _empty_result(getattr(strategy, "name", "Strategy"), start_cash, getattr(strategy, "params", None))

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions: Dict[str, Dict] = {}
    equity_curve: List[float] = []
    trade_pnls: List[float] = []
    trade_durations: List[int] = []
    trade_returns: List[float] = []
    pos_fraction = 0.20
    max_positions = int(1.0 / pos_fraction)

    for current_dt in all_dates:
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index:
                continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]
            try:
                should_exit = strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"])
            except Exception as e:
                print(f"❌ Exit error {sym}: {e}")
                should_exit = False

            if should_exit:
                row = df.iloc[i]
                exit_px = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]:
                    exit_px = row["open"]
                if exit_px > (pos["entry_price"] * 5.0):
                    exit_px = pos["entry_price"]

                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100

                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trade_returns.append(pct)
                trade_durations.append(i - pos["entry_i"])
                del positions[sym]

        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        if len(positions) < max_positions and cash > (equity * pos_fraction):
            for sym, df in enriched.items():
                if sym in positions or current_dt not in df.index:
                    continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS_FOR_WARMUP:
                    continue
                try:
                    entry = strategy.entry(df, i - 1)
                except Exception as e:
                    print(f"❌ Entry error {sym}: {e}")
                    continue

                if entry:
                    px = float(df.iloc[i]["open"])
                    stop = float(entry["stop_price"])
                    if px < 5.0:
                        continue

                    shares = int((equity * pos_fraction) / px)
                    if shares > 0 and cash >= (shares * px):
                        cash -= shares * px
                        positions[sym] = {
                            "shares": shares,
                            "entry_price": px,
                            "stop_price": stop,
                            "entry_i": i,
                        }
                if len(positions) >= max_positions:
                    break

    final_val = equity_curve[-1] if equity_curve else start_cash
    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins / trades * 100) if trades > 0 else 0.0
    avg_profit = float(np.mean(trade_returns)) if trade_returns else 0.0

    gross_profit = sum(t for t in trade_pnls if t > 0)
    gross_loss = abs(sum(t for t in trade_pnls if t < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    avg_win = np.mean([t for t in trade_returns if t > 0]) if any(t > 0 for t in trade_returns) else 0.0
    avg_loss = abs(np.mean([t for t in trade_returns if t < 0])) if any(t < 0 for t in trade_returns) else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    days_run = (all_dates[-1] - all_dates[0]).days if all_dates else 1
    cagr = (final_val / start_cash) ** (365.0 / days_run) - 1 if final_val > 0 else 0.0

    eq = pd.Series(equity_curve)
    ret = eq.pct_change().dropna()
    sharpe = (ret.mean() / ret.std() * math.sqrt(252)) if len(ret) > 2 and ret.std() > 0 else 0.0
    score = (cagr * 200) + (win_rate * 2) + (sharpe * 20) + (profit_factor * 5) + (payoff_ratio * 5)

    return {
        "strategy": getattr(strategy, "name", "Strategy"),
        "final_value": final_val,
        "total_trades": trades,
        "hit_rate": win_rate,
        "sharpe": sharpe,
        "cagr": cagr,
        "avg_profit_pct": avg_profit,
        "avg_days_held": float(np.mean(trade_durations)) if trade_durations else 0.0,
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "Score": score,
    }


def run_compare(
    strategy_names: List[str],
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[str] = None,
    use_parallel: bool = True,
    max_workers: int = 8,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    from strategies.generic import GenericStrategy

    gen_strategies: Dict[str, Dict] = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json")), "r") as f:
            for g in json.load(f):
                gen_strategies[g["name"]] = g
    except Exception as e:
        print(f"❌ Unable to load strategies: {e}")

    strategies = [GenericStrategy(gen_strategies[name]) for name in strategy_names if name in gen_strategies]
    results: List[Dict] = []

    if not strategies:
        return pd.DataFrame()

    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat
                for strat in strategies
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    strat = futures[future]
                    print(f"Backtest error for {getattr(strat, 'name', 'strategy')}: {e}")
    else:
        for strat in strategies:
            try:
                results.append(run_backtest(strat, data_dict, symbol_universe, start_cash, start_date, global_data))
            except Exception as e:
                print(f"Backtest error for {getattr(strat, 'name', 'strategy')}: {e}")

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score", ascending=False)
