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
    try:
        # 1. Clean & Sort
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()
        
        # 2. Force Timezone Naive (Crucial for Date Filtering)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # 3. Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            df[f'sma{p}'] = df['close'].rolling(p).mean()
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

        # 4. Volatility
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = rolling_mean + (rolling_std * 2)
        df['bb_lower'] = rolling_mean - (rolling_std * 2)
        df['bb_mid'] = rolling_mean
        
        mask = df['bb_mid'] != 0
        df['bb_width'] = 0.0
        df.loc[mask, 'bb_width'] = (df.loc[mask, 'bb_upper'] - df.loc[mask, 'bb_lower']) / df.loc[mask, 'bb_mid']
        
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        df['atr14_ma20'] = df['atr14'].rolling(20).mean()

        # 5. Extremes (Standard)
        df['highest20'] = df['high'].rolling(20).max()
        df['highest55'] = df['high'].rolling(55).max()
        df['lowest5'] = df['low'].rolling(5).min()
        
        # 6. Extremes (SHIFTED - The Fix for 0 Trades)
        # We shift by 1 so "highest20_1" is "Yesterday's 20-Day High"
        df['highest20_1'] = df['highest20'].shift(1)
        df['highest55_1'] = df['highest55'].shift(1)
        df['lowest5_1'] = df['lowest5'].shift(1)

        # 7. Oscillators
        def calc_rsi(series, period):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            return 100 - (100 / (1 + rs))

        df['rsi2'] = calc_rsi(df['close'], 2)
        df['rsi3'] = calc_rsi(df['close'], 3)
        df['rsi14'] = calc_rsi(df['close'], 14)
        df['crsi'] = (df['rsi3'] + df['rsi2'] + (100 - df['rsi14'])) / 3 

        adx = ADXIndicator(df['high'], df['low'], df['close'])
        df['adx'] = adx.adx()
        df['plus_di'] = adx.adx_pos()
        df['minus_di'] = adx.adx_neg()
        
        macd = MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_hist'] = macd.macd_diff()
        
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
        df['roc'] = ROCIndicator(df['close'], window=12).roc()
        df['stoch_k'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
        
        df['vol_ma20'] = df['volume'].rolling(20).mean()

        return df
    except Exception:
        return df

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

    # 1. PREPARE DATA
    for sym in symbols:
        if data_dict[sym] is None or data_dict[sym].empty: continue
        try:
            df_in = data_dict[sym].copy()
            # Timezone Normalize BEFORE indicators to avoid index mismatches
            if isinstance(df_in.index, pd.DatetimeIndex) and df_in.index.tz is not None:
                df_in.index = df_in.index.tz_localize(None)

            df = _compute_indicators(df_in)
            
            # Safe VIX Merge
            if vix_df is not None:
                vix_clean = vix_df.copy()
                if isinstance(vix_clean.index, pd.DatetimeIndex) and vix_clean.index.tz is not None:
                    vix_clean.index = vix_clean.index.tz_localize(None)
                df["vix"] = vix_clean["close"].reindex(df.index, method="ffill").fillna(20.0)
            else: 
                df["vix"] = 20.0
            
            # Date Filter (Safe because both are naive)
            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                df = df[df.index >= start_dt]
            
            if len(df) > MIN_BARS_FOR_WARMUP: enriched[sym] = df
        except: continue

    if not enriched: return _empty_result(strategy.name, start_cash, strategy.params)

    # 2. RUN SIMULATION
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
                
                if exit_px > (pos["entry_price"] * 5.0): exit_px = pos["entry_price"] # Circuit Breaker

                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100
                
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trade_returns.append(pct)
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
        if len(positions) < max_positions and cash > (equity * pos_fraction):
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS_FOR_WARMUP: continue
                
                # Entry
                entry = strategy.entry(df, i)
                if entry:
                    px = float(df.iloc[i]["open"])
                    stop = float(entry["stop_price"])
                    if px < 5.0: continue
                    
                    shares = int((equity * pos_fraction) / px)
                    if shares > 0 and cash >= (shares * px):
                        cash -= (shares * px)
                        positions[sym] = {"shares": shares, "entry_price": px, "stop_price": stop, "entry_i": i}
            if len(positions) >= max_positions: break

    # 3. METRICS
    final_val = equity_curve[-1] if equity_curve else start_cash
    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    win_rate = (wins/trades*100) if trades > 0 else 0.0
    avg_profit = np.mean(trade_returns) if trade_returns else 0.0
    
    gross_profit = sum(t for t in trade_pnls if t > 0)
    gross_loss = abs(sum(t for t in trade_pnls if t < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
    
    avg_win = np.mean([t for t in trade_returns if t > 0]) if any(t > 0 for t in trade_returns) else 0
    avg_loss = abs(np.mean([t for t in trade_returns if t < 0])) if any(t < 0 for t in trade_returns) else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 0 else 1
    cagr = (final_val / start_cash) ** (365.0/days) - 1 if final_val > 0 else 0.0
    
    eq = pd.Series(equity_curve)
    sharpe = (eq.pct_change().dropna().mean() / eq.pct_change().dropna().std() * math.sqrt(252)) if len(eq) > 1 and eq.std() > 0 else 0.0
    score = (cagr * 200) + (win_rate * 2) + (sharpe * 20) + (avg_profit * 50)

    return {
        "strategy": strategy.name, "final_value": final_val, "total_trades": trades,
        "hit_rate": win_rate, "sharpe": sharpe, "cagr": cagr, 
        "avg_profit_pct": avg_profit, "avg_days_held": np.mean(trade_durations) if trade_durations else 0.0,
        "profit_factor": profit_factor, "payoff_ratio": payoff_ratio, "Score": score,
        "params": strategy.params
    }

def run_compare(strategy_names, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, use_parallel=True, max_workers=8, global_data=None):
    import json, os, concurrent.futures
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
