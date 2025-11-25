import math
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, date
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from strategies.base import BaseStrategy

MIN_BARS = 200

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()
        
        # --- PRICE & VOLATILITY ---
        df['sma50'] = df['close'].rolling(50).mean()
        df['sma200'] = df['close'].rolling(200).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        df['atr14_ma20'] = df['atr14'].rolling(20).mean()

        # --- BREAKOUT LOGIC (SHIFTED) ---
        # Highest High of LAST 20 days (excluding today)
        # This is CRITICAL for backtesting "Close > Highest20"
        df['highest20'] = df['high'].rolling(20).max()
        df['highest20_1'] = df['highest20'].shift(1) 
        
        df['lowest5'] = df['low'].rolling(5).min()
        df['lowest5_1'] = df['lowest5'].shift(1)

        # --- OSCILLATORS ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(2).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi2'] = 100 - (100 / (1 + rs))
        
        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        return df
    except: return df

def _empty_result(name, start_cash, params=None):
    return {
        "strategy": name, "final_value": start_cash, "total_trades": 0,
        "hit_rate": 0.0, "sharpe": 0.0, "cagr": 0.0, "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0, "avg_days_held": 0.0, "profit_factor": 0.0,
        "payoff_ratio": 0.0, "Score": 0.0, "params": params or {}
    }

def run_backtest(strategy, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
    symbols = list(data_dict.keys())
    if symbol_universe: symbols = [s for s in symbols if s in symbol_universe]
    
    enriched = {}
    vix_df = global_data.get("VIX") if global_data else None

    for sym in symbols:
        if data_dict[sym] is None or data_dict[sym].empty: continue
        try:
            # PREPARE DATA
            df = _compute_indicators(data_dict[sym].copy())
            
            # INJECT VIX
            if vix_df is not None:
                df["vix"] = vix_df["close"].reindex(df.index, method="ffill").fillna(20.0)
            else: df["vix"] = 20.0
            
            if start_date: 
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]
            
            if len(df) > MIN_BARS: enriched[sym] = df
        except: continue

    if not enriched: return _empty_result(strategy.name, start_cash, strategy.params)

    # SIMULATION
    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions = {}
    trade_returns = []
    trade_durations = []
    pos_fraction = 0.20
    max_pos = 5

    for current_dt in all_dates:
        # 1. EXIT
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index: continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]
            
            if strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"]):
                row = df.iloc[i]
                # Sell at Close
                exit_px = row["close"]
                # Hard Stop Check
                if row["low"] < pos["stop_price"]: exit_px = pos["stop_price"]
                
                cash += pos["shares"] * exit_px
                
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100
                trade_returns.append(pct)
                trade_durations.append(i - pos["entry_i"])
                del positions[sym]

        # 2. ENTRY
        if len(positions) < max_pos:
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                
                # USE CURRENT INDEX `i`. 
                # Strategy `entry` logic must handle looking back (e.g. close > highest20_1)
                entry = strategy.entry(df, i)
                
                if entry:
                    px = float(df.iloc[i]["close"]) # Enter at Close for simplicity/accuracy
                    stop = float(entry["stop_price"])
                    cost = (cash * pos_fraction)
                    shares = int(cost / px)
                    if shares > 0:
                        cash -= (shares * px)
                        positions[sym] = {"shares": shares, "entry_price": px, "stop_price": stop, "entry_i": i}

    # METRICS
    trades = len(trade_returns)
    wins = len([t for t in trade_returns if t > 0])
    win_rate = (wins/trades*100) if trades > 0 else 0.0
    avg_profit = np.mean(trade_returns) if trade_returns else 0.0
    score = (avg_profit * 50) + (win_rate * 2)

    return {
        "strategy": strategy.name, "final_value": cash, "total_trades": trades,
        "hit_rate": win_rate, "sharpe": 0.0, "cagr": 0.0, 
        "avg_profit_pct": avg_profit, "avg_days_held": np.mean(trade_durations) if trade_durations else 0.0,
        "profit_factor": 0.0, "payoff_ratio": 0.0, "Score": score,
        "params": strategy.params
    }

def run_compare(strategy_names, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, use_parallel=True, max_workers=8, global_data=None):
    import json, os
    from strategies.generic import GenericStrategy
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gen_strategies = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json')), "r") as f:
            for g in json.load(f): gen_strategies[g["name"]] = g
    except: pass

    strategies = []
    for name in strategy_names:
        if name in gen_strategies: strategies.append(GenericStrategy(gen_strategies[name]))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat for strat in strategies}
        for future in as_completed(futures):
            try: results.append(future.result())
            except: pass

    if not results: return pd.DataFrame()
    return pd.DataFrame(results).sort_values("Score", ascending=False)
