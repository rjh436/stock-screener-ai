import math
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, date
from typing import Dict, List, Optional, Type
import numpy as np
import pandas as pd
import concurrent.futures
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
    
    # Volatility
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_mid'] = rolling_mean
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['atr14_ma20'] = df['atr14'].rolling(20).mean()
    
    # Extremes
    df['highest20'] = df['high'].rolling(20).max()
    df['highest55'] = df['high'].rolling(55).max()
    df['lowest5'] = df['low'].rolling(5).min()
    
    # Oscillators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # RSI 2
    gain2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs2 = gain2 / loss2.replace(0, np.nan)
    df['rsi2'] = 100 - (100 / (1 + rs2))

    # Trend Strength
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

def run_backtest(
    strategy: BaseStrategy,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    symbols = list(data_dict.keys())
    if symbol_universe:
        symbols = [s for s in symbols if s in symbol_universe]

    if not symbols:
        return _empty_result(strategy.name, start_cash, strategy.params)

    enriched = {}
    spy_df = None
    if global_data and "SPY" in global_data:
        spy_df = _compute_indicators(global_data["SPY"].copy()) if global_data["SPY"] is not None else None
    
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        df = data_dict[sym]
        if df is None or df.empty: continue
        df_local = df.copy()
        try:
            df_local = _compute_indicators(df_local)
            
            # Market Context injection
            if spy_df is not None:
                spy_reindexed = spy_df["close"].reindex(df_local.index, method="ffill")
                df_local["rs_ratio"] = df_local["close"] / spy_reindexed
                df_local["rs_sma20"] = df_local["rs_ratio"].rolling(20).mean()
                df_local["rs_trend"] = (df_local["rs_ratio"] > df_local["rs_sma20"]).astype(int)
            else:
                df_local["rs_trend"] = 0
            
            if vix_df is not None:
                vix_reindexed = vix_df["close"].reindex(df_local.index, method="ffill")
                df_local["vix"] = vix_reindexed
            else:
                df_local["vix"] = 20.0

            if start_date:
                from_dt = datetime.combine(start_date, datetime.min.time())
                df_local = df_local[df_local.index >= from_dt]
            
            if len(df_local) > MIN_BARS_FOR_WARMUP:
                enriched[sym] = df_local
        except:
            continue

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trade_durations = [] # Metric: Days Held
    
    max_positions = 5  # Aggressive Sizing (20%) requires max 5
    pos_fraction = 0.20 # Aggressive Sizing

    for current_dt in all_dates:
        # 1. Process Exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index: continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            row = df.iloc[i]
            pos = positions[sym]
            
            exit_signal = strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"])
            
            if exit_signal:
                exit_price = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                # Gap down check
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]:
                     exit_price = row["open"]

                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * exit_price
                trade_pnls.append(pnl)
                
                # Track Duration
                days_held = i - pos["entry_i"]
                trade_durations.append(days_held)
                
                del positions[sym]

        # 2. Mark Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        # 3. Process Entries
        if len(positions) >= max_positions or cash < (equity * pos_fraction): continue
        
        for sym, df in enriched.items():
            if current_dt not in df.index or sym in positions: continue
            i = df.index.get_loc(current_dt)
            if i < MIN_BARS_FOR_WARMUP: continue
            
            # Signal is from PREVIOUS day closing data
            entry_signal = strategy.entry(df, i-1)
            
            if entry_signal:
                price = float(df.iloc[i]["open"]) # Enter on Open
                stop = float(entry_signal["stop_price"])
                if price <= 0: continue
                
                shares = int((equity * pos_fraction) / price)
                if shares > 0 and cash >= (shares * price):
                    cash -= (shares * price)
                    positions[sym] = {
                        "shares": shares,
                        "entry_price": price,
                        "stop_price": stop,
                        "entry_i": i
                    }
            if len(positions) >= max_positions: break

    final_value = equity_curve[-1] if equity_curve else start_cash
    total_trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    avg_profit_pct = (np.mean(trade_pnls) / start_cash * 100) if trade_pnls else 0.0 # Approximate ROI per trade
    avg_days_held = np.mean(trade_durations) if trade_durations else 0.0

    # CAGR
    days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 0 else 1
    cagr = (final_value / start_cash) ** (365.0/days) - 1 if final_value > 0 else 0.0
    
    # Sharpe
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if not returns.empty and returns.std() > 0 else 0.0
    
    # Drawdown
    rolling_max = pd.Series(equity_curve).cummax()
    dd = (pd.Series(equity_curve) - rolling_max) / rolling_max
    max_dd = dd.min() * 100

    return {
        "strategy": strategy.name,
        "final_value": final_value,
        "total_trades": total_trades,
        "hit_rate": win_rate,
        "sharpe": sharpe,
        "cagr": cagr,
        "max_drawdown_pct": max_dd,
        "avg_profit_pct": avg_profit_pct, # Note: This is portfolio impact %, strictly speaking
        "avg_days_held": avg_days_held
    }

def _empty_result(name, start_cash, params=None):
    return {
        "strategy": name, "final_value": start_cash, "total_trades": 0,
        "hit_rate": 0.0, "sharpe": 0.0, "cagr": 0.0, "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0, "avg_days_held": 0.0, "params": params or {}
    }
