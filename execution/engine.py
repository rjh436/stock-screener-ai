import math
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy

MIN_BARS_FOR_WARMUP = 200

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_mid'] = rolling_mean
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    high_low = df['high'] - df['low']
    tr = pd.concat([high_low, (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['atr14_ma20'] = df['atr14'].rolling(20).mean()
    df['highest20'] = df['high'].rolling(20).max()
    df['highest55'] = df['high'].rolling(55).max()
    df['lowest5'] = df['low'].rolling(5).min()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    gain2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs2 = gain2 / loss2.replace(0, np.nan)
    df['rsi2'] = 100 - (100 / (1 + rs2))

    adx = ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['plus_di'] = adx.adx_pos()
    df['minus_di'] = adx.adx_neg()
    
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_hist'] = macd.macd_diff()
    
    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    cci = CCIIndicator(df['high'], df['low'], df['close'])
    df['cci'] = cci.cci()
    
    roc = ROCIndicator(df['close'], window=12)
    df['roc'] = roc.roc()
    df['mom'] = df['close'].diff(10)

    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['avgvol50'] = df['volume'].rolling(50).mean()

    return df

def run_backtest(strategy, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
    symbols = list(data_dict.keys())
    if symbol_universe: symbols = [s for s in symbols if s in symbol_universe]
    if not symbols: return _empty_result(strategy.name, start_cash, strategy.params)

    enriched = {}
    spy_df = _compute_indicators(global_data["SPY"].copy()) if global_data and "SPY" in global_data and global_data["SPY"] is not None else None
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        df = data_dict[sym]
        if df is None or df.empty: continue
        try:
            df_local = _compute_indicators(df.copy())
            if spy_df is not None:
                spy_reindexed = spy_df["close"].reindex(df_local.index, method="ffill")
                df_local["rs_ratio"] = df_local["close"] / spy_reindexed
                df_local["rs_sma20"] = df_local["rs_ratio"].rolling(20).mean()
                df_local["rs_trend"] = (df_local["rs_ratio"] > df_local["rs_sma20"]).astype(int)
            else: df_local["rs_trend"] = 0
            
            if vix_df is not None:
                vix_reindexed = vix_df["close"].reindex(df_local.index, method="ffill")
                df_local["vix"] = vix_reindexed
            else: df_local["vix"] = 20.0

            if start_date: df_local = df_local[df_local.index >= datetime.combine(start_date, datetime.min.time())]
            if len(df_local) > MIN_BARS_FOR_WARMUP: enriched[sym] = df_local
        except: continue

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trade_returns = []
    trade_durations = []
    
    max_positions = 5
    pos_fraction = 0.250

    for current_dt in all_dates:
        # 1. Exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index: continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            row = df.iloc[i]
            pos = positions[sym]
            
            exit_signal = strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"])
            
            if exit_signal:
                exit_price = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]: exit_price = row["open"]

                # [CIRCUIT BREAKER]
                if exit_price > (pos["entry_price"] * 4.0): exit_price = pos["entry_price"]

                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
                
                cash += pos["shares"] * exit_price
                trade_pnls.append(pnl)
                trade_returns.append(pnl_pct)
                trade_durations.append(i - pos["entry_i"])
                del positions[sym]

        # 2. Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        # 3. Entries
        if len(positions) >= max_positions or cash < (equity * pos_fraction): continue
        for sym, df in enriched.items():
            if current_dt not in df.index or sym in positions: continue
            i = df.index.get_loc(current_dt)
            if i < MIN_BARS_FOR_WARMUP: continue
            
            entry_signal = strategy.entry(df, i-1)
            if entry_signal:
                price = float(df.iloc[i]["open"])
                stop = float(entry_signal["stop_price"])
                if price < 5.0: continue 
                
                shares = int((equity * pos_fraction) / price)
                if shares > 0 and cash >= (shares * price):
                    cash -= (shares * price)
                    positions[sym] = {"shares": shares, "entry_price": price, "stop_price": stop, "entry_i": i}
            if len(positions) >= max_positions: break

    final_value = equity_curve[-1] if equity_curve else start_cash
    total_trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    avg_profit_pct = (np.mean(trade_returns)) if trade_returns else 0.0
    avg_days_held = np.mean(trade_durations) if trade_durations else 0.0
    
    # Advanced Metrics
    gross_profit = sum(t for t in trade_pnls if t > 0)
    gross_loss = abs(sum(t for t in trade_pnls if t < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 99.0
    
    avg_win = np.mean([t for t in trade_returns if t > 0]) if any(t > 0 for t in trade_returns) else 0
    avg_loss = abs(np.mean([t for t in trade_returns if t < 0])) if any(t < 0 for t in trade_returns) else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 0 else 1
    cagr = (final_value / start_cash) ** (365.0/days) - 1 if final_value > 0 else 0.0
    
    eq_series = pd.Series(equity_curve)
    drawdown = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = drawdown.min() * 100
    
    rets = eq_series.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if not rets.empty and rets.std() > 0 else 0.0

    return {
        "strategy": strategy.name, "final_value": final_value, "total_trades": total_trades,
        "hit_rate": win_rate, "sharpe": sharpe, "cagr": cagr, "max_drawdown_pct": max_dd,
        "avg_profit_pct": avg_profit_pct, "avg_days_held": avg_days_held,
        "profit_factor": profit_factor, "payoff_ratio": payoff_ratio
    }

def _empty_result(name, start_cash, params=None):
    return {"strategy": name, "final_value": start_cash, "total_trades": 0, "hit_rate": 0, "sharpe": 0, "cagr": 0, "max_drawdown_pct": 0, "avg_profit_pct": 0, "avg_days_held": 0, "profit_factor": 0, "payoff_ratio": 0}
