import os


def fix_score_clamping():
    print("🔓 UNCLAMPING ENGINE SCORES (Restoring Deep Value Priority)...")

    engine_path = "execution/engine.py"

    # We rewrite the engine with the KNOWN Golden State logic (Unclamped)
    engine_code = """
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

MIN_BARS = 200

def _compute_indicators(df: pd.DataFrame, spy_df: pd.DataFrame = None) -> pd.DataFrame:
    try:
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()
        
        for p in [10, 20, 50, 200]:
            df[f'sma{p}'] = df['close'].rolling(p).mean()
            df[f'ema{p}'] = df['close'].ewm(span=p, adjust=False).mean()

        df['bb_upper'] = df['close'].rolling(20).mean() + (df['close'].rolling(20).std() * 2)
        df['bb_lower'] = df['close'].rolling(20).mean() - (df['close'].rolling(20).std() * 2)
        df['bb_mid'] = df['close'].rolling(20).mean()
        
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

        df['highest20'] = df['high'].rolling(20).max()
        df['highest20_1'] = df['highest20'].shift(1)
        df['highest55'] = df['high'].rolling(55).max()
        df['highest55_1'] = df['highest55'].shift(1)
        df['lowest5'] = df['low'].rolling(5).min()
        df['lowest5_1'] = df['lowest5'].shift(1)

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi14'] = 100 - (100 / (1 + rs))
        
        g2 = (delta.where(delta > 0, 0)).rolling(2).mean()
        l2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = g2 / l2.replace(0, np.nan)
        df['rsi2'] = 100 - (100 / (1 + rs2))

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

        if spy_df is not None and not spy_df.empty:
            spy_aligned = spy_df['close'].reindex(df.index).ffill().bfill()
            df['rs_ratio'] = df['close'] / spy_aligned
            df['rs_sma20'] = df['rs_ratio'].rolling(20).mean()
            df['rs_trend'] = df['rs_ratio'] - df['rs_sma20'] 
        else:
            df['rs_ratio'] = 1.0
            df['rs_trend'] = 0.0

        return df
    except: return df

def _empty_result(name, start_cash, params=None):
    return {
        "strategy": name, "final_value": start_cash, "total_trades": 0,
        "hit_rate": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "cagr": 0.0, "calmar": 0.0, "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0, "avg_days_held": 0.0, "exposure_pct": 0.0,
        "profit_factor": 0.0, "payoff_ratio": 0.0, "max_consecutive_losses": 0,
        "beta": 0.0, "avg_signals_per_day": 0.0, "Score": 0.0, 
        "params": params or {}, "trades_list": [], "equity_curve": []
    }

def calculate_backtest_quality_score(row, strategy_name):
    # GOLDEN STATE LOGIC (UNCLAMPED)
    score = 50.0
    
    # 1. Sniper Priority (The "Super Signal" Boost)
    cci = row.get("cci", 0)
    bb_width = row.get("bb_width", 0)
    if cci < 0 and bb_width > 0.1:
        score += 50
    
    # 2. Base Oversold (Shared)
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 3. Velocity Booster (Moderate)
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
        
    # 4. Bifurcated Trend (Machine Gun Only)
    if "MachineGun" in strategy_name:
        if row.get("close", 0) > row.get("sma200", 999999):
            score += 20
    
    # 5. Volume Support
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    # CRITICAL FIX: REMOVED MIN(100)
    return max(0.0, score) 

def run_backtest(strategy, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
    symbols = list(data_dict.keys())
    if symbol_universe: symbols = [s for s in symbols if s in symbol_universe]
    
    enriched = {}
    vix_df = global_data.get("VIX") if global_data else None
    spy_df = global_data.get("SPY") if global_data else None

    for sym in symbols:
        if data_dict[sym] is None or data_dict[sym].empty: continue
        try:
            df = _compute_indicators(data_dict[sym].copy(), spy_df=spy_df)
            if vix_df is not None:
                df["vix"] = vix_df["close"].reindex(df.index).ffill().fillna(20.0)
            else: df["vix"] = 20.0
            
            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]
            
            if len(df) > MIN_BARS: enriched[sym] = df
        except: continue

    if not enriched: return _empty_result(strategy.name, start_cash, strategy.params)

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    min_date = pd.Timestamp.now() - pd.Timedelta(days=365*20)
    all_dates = [d for d in all_dates if d >= min_date]
    if not all_dates: return _empty_result(strategy.name, start_cash, strategy.params)

    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trades_list = []
    
    days_invested = 0
    max_positions = 5
    raw_pos = strategy.params.get('pos_size', strategy.params.get('position_size', strategy.params.get('pos_fraction', 0.20)))
    try: pos_fraction = float(raw_pos)
    except: pos_fraction = 0.20
    pos_fraction = max(0.01, min(pos_fraction, 1.0))

    for current_dt in all_dates:
        if len(positions) > 0: days_invested += 1

        # 1. Exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index: continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]
            if i <= pos["entry_i"]: continue
            
            if strategy.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"]):
                row = df.iloc[i]
                exit_px = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]: exit_px = row["open"]
                if exit_px < pos["entry_price"] * 0.5: exit_px = pos["entry_price"] * 0.5 
                
                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100
                
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                
                trades_list.append({
                    "Symbol": sym, 
                    "Entry Date": str(df.index[pos["entry_i"]].date()), 
                    "Exit Date": str(current_dt.date()), 
                    "Entry": pos["entry_price"], 
                    "Exit": exit_px, 
                    "PnL": pnl, 
                    "Return%": pct
                })
                del positions[sym]

        # 2. Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        equity_curve.append({"Date": current_dt, "Equity": equity})

        # 3. Entries (Scan Close i-1, Buy Open i)
        if cash > 0:
            daily_candidates = []
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS + 1: continue
                
                if strategy.entry(df, i - 1):
                    row_prev = df.iloc[i-1]
                    row_curr = df.iloc[i]
                    
                    score = calculate_backtest_quality_score(row_prev, strategy.name)
                    open_px = float(row_curr["open"])
                    
                    stop_dist = float(row_prev["close"]) - strategy.entry(df, i-1)["stop_price"]
                    real_stop = open_px - stop_dist
                    
                    daily_candidates.append({
                        "sym": sym, "px": open_px, "stop": real_stop, 
                        "score": score, "rsi2": float(row_prev.get("rsi2", 50))
                    })
            
            if len(positions) < max_positions:
                daily_candidates.sort(key=lambda x: (x["score"], -x["rsi2"]), reverse=True)
                
                target_size = equity * pos_fraction
                for cand in daily_candidates:
                    if len(positions) >= max_positions or cash < 500: break
                    shares = int(target_size / cand["px"])
                    cost = shares * cand["px"]
                    if shares > 0 and cash >= cost:
                        cash -= cost
                        positions[cand["sym"]] = {
                            "shares": shares, "entry_price": cand["px"], 
                            "stop_price": cand["stop"], 
                            "entry_i": enriched[cand["sym"]].index.get_loc(current_dt)
                        }

    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    hit_rate = (wins/trades*100) if trades > 0 else 0.0
    avg_profit = (sum(trade_pnls)/start_cash)*100 / (trades or 1)
    
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty: eq_df = pd.DataFrame({"Equity": [start_cash]}, index=[pd.Timestamp.now()])
    
    days = (eq_df.index[-1] - eq_df.index[0]).days
    years = days / 365.25
    cagr = ((eq_df["Equity"].iloc[-1] / start_cash) ** (1.0 / (years if years > 0 else 1))) - 1.0
    
    rolling_max = eq_df["Equity"].cummax()
    drawdowns = (eq_df["Equity"] - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100

    return {
        "strategy": strategy.name, "final_value": equity, "total_trades": trades,
        "hit_rate": hit_rate, "cagr": cagr, "avg_profit_pct": avg_profit,
        "max_drawdown_pct": max_dd,
        "equity_curve": eq_df, 
        "trades_list": trades_list,
        "params": strategy.params,
        "Score": cagr * 1000
    }

def run_compare(strategy_names, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
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
            except: pass
    if not results: return pd.DataFrame()
    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['equity_curve', 'trades_list']} for r in results])
    return df.sort_values("Score", ascending=False)
"""
    with open(engine_path, "w") as f:
        f.write(engine_code)
    print("   ✅ Engine Unclamped. Granularity restored.")


if __name__ == "__main__":
    fix_score_clamping()
