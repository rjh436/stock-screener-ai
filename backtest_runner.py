import math
from datetime import datetime, date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import concurrent.futures

# ============================================================
# M3 MAX OPTIMIZED BACKTESTER (Vectorized)
# ============================================================

MIN_BARS_FOR_WARMUP = 200

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    # Basic Indicators
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_mid'] = rolling_mean
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr14'] = true_range.rolling(14).mean()
    df['atr14_ma20'] = df['atr14'].rolling(20).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi14'] = 100 - (100 / (1 + rs))
    
    # RSI 2
    gain2 = (delta.where(delta > 0, 0)).rolling(window=2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(window=2).mean()
    rs2 = gain2 / loss2
    df['rsi2'] = 100 - (100 / (1 + rs2))
    
    # Volume MA
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    
    return df

def _precalculate_signals(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    VECTORIZED SIGNAL GENERATION.
    Calculates Buy/Sell booleans for the whole dataframe at once.
    """
    buy_signal = pd.Series(False, index=df.index)
    sell_signal = pd.Series(False, index=df.index)
    
    if strategy_name == "RHCTS":
        # Close > SMA50 AND RSI < 70 AND Reclaim EMA20
        c1 = df['close'] > df['sma50']
        c2 = df['rsi14'] < 70
        c3 = (df['close'] > df['ema20']) & (df['low'] < df['ema20'])
        buy_signal = c1 & c2 & c3
        sell_signal = df['close'] < df['ema20']

    elif strategy_name == "ConnorsRSI":
        buy_signal = (df['rsi2'] < 10) & (df['close'] > df['sma200'])
        sell_signal = df['close'] > df['sma5']
        
    elif strategy_name == "Raptor":
        buy_signal = (df['rsi2'] < 10) & (df['close'] > df['ema50']) & (df['close'] <= df['bb_lower'])
        sell_signal = df['close'] > df['bb_mid']

    # IMPORTANT: Shift signals by 1 because we trade on NEXT Open
    df['buy_signal'] = buy_signal.shift(1).fillna(False)
    df['sell_signal'] = sell_signal.shift(1).fillna(False)
    
    return df

def _backtest_strategy(strategy_name: str, data_dict: Dict[str, pd.DataFrame], symbol_universe=None, start_cash=100000.0, start_date=None, **kwargs):
    """
    Event Loop optimized for M3. Iterates ONLY days, but logic is pre-calculated.
    """
    cash = start_cash
    positions = {} 
    
    # 1. Pre-process all DataFrames (Vectorized)
    processed_data = {}
    all_dates = set()
    
    for sym, df in data_dict.items():
        if len(df) < 200: continue
        df = _compute_indicators(df.copy())
        df = _precalculate_signals(df, strategy_name)
        if start_date:
             df = df[df.index >= pd.to_datetime(start_date).tz_localize("UTC")]
        if not df.empty:
            processed_data[sym] = df
            all_dates.update(df.index)
        
    sorted_dates = sorted(list(all_dates))
    trade_log = []
    equity_curve = []
    
    # 2. Fast Event Loop
    for dt in sorted_dates:
        # A. Check Exits
        for sym in list(positions.keys()):
            df = processed_data[sym]
            if dt not in df.index: continue
            row = df.loc[dt]
            p = positions[sym]
            
            if row['low'] < p['stop'] or row['sell_signal']:
                exit_px = p['stop'] if row['low'] < p['stop'] else row['open']
                cash += p['shares'] * exit_px
                pnl_pct = (exit_px - p['entry_price']) / p['entry_price']
                trade_log.append(pnl_pct)
                del positions[sym]
        
        # B. Check Entries
        if cash > 0:
            for sym, df in processed_data.items():
                if dt not in df.index: continue
                if sym in positions: continue
                if len(positions) >= 10: break 
                
                row = df.loc[dt]
                if row['buy_signal']:
                     alloc = start_cash * 0.10
                     if cash < alloc: alloc = cash
                     px = row['open']
                     shares = int(alloc / px)
                     if shares > 0:
                         atr = row.get('atr14', px * 0.02)
                         positions[sym] = {
                             'shares': shares,
                             'entry_price': px,
                             'stop': px - (atr * 2.0)
                         }
                         cash -= (shares * px)
        
        # C. Mark Equity
        equity = cash
        for sym, p in positions.items():
            df = processed_data[sym]
            if dt in df.index:
                equity += p['shares'] * df.loc[dt]['close']
            else:
                equity += p['shares'] * p['entry_price']
        equity_curve.append(equity)

    # Metrics
    final_value = equity_curve[-1] if equity_curve else start_cash
    total_trades = len(trade_log)
    wins = len([t for t in trade_log if t > 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    eq_series = pd.Series(equity_curve)
    returns = eq_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if not returns.empty and returns.std() > 0 else 0
    
    return {
        "strategy": strategy_name,
        "final_value": final_value,
        "pnl_pct": (final_value - start_cash) / start_cash * 100,
        "hit_rate": win_rate,
        "total_trades": total_trades,
        "sharpe": sharpe,
        "Score": sharpe * 10 + (win_rate/10),
        "cagr": 0.0, # Placeholder
        "max_drawdown_pct": 0.0, # Placeholder
        "avg_profit_pct": np.mean(trade_log) * 100 if trade_log else 0.0
    }

def run_compare(strategy_names, data_dict, **kwargs):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for name in strategy_names:
            futures.append(executor.submit(_backtest_strategy, name, data_dict))
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except Exception as e: print(f"Strategy failed: {e}")
    return pd.DataFrame(results)