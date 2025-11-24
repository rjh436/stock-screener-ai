import math
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import concurrent.futures
from strategies.base import BaseStrategy

MIN_BARS_FOR_WARMUP = 200

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL indicators required for Sniper strategies."""
    df = df.sort_index().copy()
    
    # --- Moving Averages ---
    df['sma5'] = df['close'].rolling(5).mean()
    df['sma10'] = df['close'].rolling(10).mean()
    df['sma20'] = df['close'].rolling(20).mean()
    df['sma50'] = df['close'].rolling(50).mean()
    df['sma100'] = df['close'].rolling(100).mean()
    df['sma200'] = df['close'].rolling(200).mean()
    
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # --- Volatility (Bollinger & ATR) ---
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_mid'] = rolling_mean
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr10'] = tr.rolling(10).mean()
    df['atr14'] = tr.rolling(14).mean()
    df['atr20'] = tr.rolling(20).mean()
    df['atr14_ma20'] = df['atr14'].rolling(20).mean()
    df['atr_pct10'] = (df['atr10'] / df['close']) * 100
    df['atr_pct20'] = (df['atr20'] / df['close']) * 100

    # --- Extremes (Donchian) ---
    df['highest20'] = df['high'].rolling(20).max()
    df['highest55'] = df['high'].rolling(55).max()
    df['lowest5'] = df['low'].rolling(5).min()
    df['lowest20'] = df['low'].rolling(20).min()
    
    # --- Keltner (Approx) ---
    df['kc_mid'] = df['sma20']
    df['kc_upper'] = df['kc_mid'] + (df['atr14'] * 2)
    df['kc_lower'] = df['kc_mid'] - (df['atr14'] * 2)

    # --- Oscillators (RSI, Stoch, CCI) ---
    def calc_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df['rsi2'] = calc_rsi(df['close'], 2)
    df['rsi3'] = calc_rsi(df['close'], 3)
    df['rsi5'] = calc_rsi(df['close'], 5)
    df['rsi14'] = calc_rsi(df['close'], 14)
    
    # Connors RSI
    df['crsi'] = (df['rsi3'] + df['rsi2'] + (100 - df['rsi14'])) / 3 

    stoch = StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    cci = CCIIndicator(df['high'], df['low'], df['close'])
    df['cci'] = cci.cci()
    
    roc = ROCIndicator(df['close'], window=12)
    df['roc'] = roc.roc()
    df['mom'] = df['close'].diff(10)

    # --- Trend Strength ---
    adx = ADXIndicator(df['high'], df['low'], df['close'])
    df['adx'] = adx.adx()
    df['plus_di'] = adx.adx_pos()
    df['minus_di'] = adx.adx_neg()
    
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # --- Volume ---
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['avgvol50'] = df['volume'].rolling(50).mean()

    return df

def _empty_result(name, start_cash, params=None):
    return {
        "strategy": name,
        "final_value": start_cash,
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "total_trades": 0,
        "win_total": 0,
        "loss_total": 0,
        "hit_rate": 0.0,
        "sharpe": 0.0,
        "cagr": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0,
        "avg_days_held": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "Score": 0.0, # FIXED: Ensure Score exists for UI
        "params": params if params else {}
    }

def run_backtest(strategy, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
    symbols = list(data_dict.keys())
    if symbol_universe: symbols = [s for s in symbols if s in symbol_universe]
    if not symbols: return _empty_result(strategy.name, start_cash, strategy.params)

    enriched = {}
    spy_df = _compute_indicators(global_data["SPY"].copy()) if global_data and "SPY" in global_data else None
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        if data_dict[sym] is None or data_dict[sym].empty: continue
        try:
            df = _compute_indicators(data_dict[sym].copy())
            
            # Market Context
            if spy_df is not None:
                spy_reindexed = spy_df["close"].reindex(df.index, method="ffill")
                df["rs_ratio"] = df["close"] / spy_reindexed
                df["rs_sma20"] = df["rs_ratio"].rolling(20).mean()
                df["rs_trend"] = (df["rs_ratio"] > df["rs_sma20"]).astype(int)
            else: df["rs_trend"] = 0
            
            if vix_df is not None:
                df["vix"] = vix_df["close"].reindex(df.index, method="ffill")
            else: df["vix"] = 20.0

            if start_date: 
                df = df[df.index >= datetime.combine(start_date, datetime.min.time())]
            
            if len(df) > MIN_BARS_FOR_WARMUP: enriched[sym] = df
        except: continue

    if not enriched: return _empty_result(strategy.name, start_cash, strategy.params)

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trade_returns = []
    trade_durations = []
    
    max_positions = 5
    pos_fraction = 0.20

    for current_dt in all_dates:
        # Exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index: continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]
            
            if strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"]):
                row = df.iloc[i]
                exit_px = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]: exit_px = row["open"]
                
                # Circuit Breaker
                if exit_px > (pos["entry_price"] * 4.0): exit_px = pos["entry_price"]

                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pnl_pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100
                
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trade_returns.append(pnl_pct)
                trade_durations.append(i - pos["entry_i"])
                del positions[sym]

        # Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        # Entries
        if len(positions) >= max_positions or cash < (equity * pos_fraction): continue
        for sym, df in enriched.items():
            if current_dt not in df.index or sym in positions: continue
            i = df.index.get_loc(current_dt)
            if i < MIN_BARS_FOR_WARMUP: continue
            
            entry = strategy.entry(df, i-1)
            if entry:
                px = float(df.iloc[i]["open"])
                stop = float(entry["stop_price"])
                if px < 5.0: continue
                
                shares = int((equity * pos_fraction) / px)
                if shares > 0 and cash >= (shares * px):
                    cash -= (shares * px)
                    positions[sym] = {"shares": shares, "entry_price": px, "stop_price": stop, "entry_i": i}
            if len(positions) >= max_positions: break

    final_val = equity_curve[-1] if equity_curve else start_cash
    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins/trades*100) if trades > 0 else 0.0
    avg_profit = np.mean(trade_returns) if trade_returns else 0.0
    
    days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 0 else 1
    cagr = (final_val / start_cash) ** (365.0/days) - 1 if final_val > 0 else 0.0
    
    eq = pd.Series(equity_curve)
    sharpe = (eq.pct_change().dropna().mean() / eq.pct_change().dropna().std() * np.sqrt(252)) if len(eq) > 1 and eq.std() > 0 else 0.0
    
    # Score Calculation
    score = (cagr * 300) + (avg_profit * 100) + (min(sharpe, 3.0) * 50)

    return {
        "strategy": strategy.name, "final_value": final_val, "total_trades": trades,
        "hit_rate": win_rate, "sharpe": sharpe, "cagr": cagr, 
        "avg_profit_pct": avg_profit, "avg_days_held": np.mean(trade_durations) if trade_durations else 0.0,
        "Score": score, # FIXED: Ensuring Score is present
        "params": strategy.params
    }

def run_compare(strategy_names, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, use_parallel=True, max_workers=8, global_data=None):
    import json
    import os
    from strategies.generic import GenericStrategy

    gen_strategies = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json')), "r") as f:
            for g in json.load(f): gen_strategies[g["name"]] = g
    except: pass

    strategies = []
    for name in strategy_names:
        if name in gen_strategies:
            strategies.append(GenericStrategy(gen_strategies[name]))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat for strat in strategies}
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except Exception as e: print(f"Error {futures[future].name}: {e}")

    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score", ascending=False)
