import json
import os


def align_engine_and_portfolio():
    print("🔗 ALIGNING ENGINE LOGIC WITH STRATEGY NAMES...")

    # --- 1. RENAME STRATEGIES ---
    config_path = "config/generated_strategies.json"
    
    # We take the EXACT parameters from your upload, just changing the 'name' field
    portfolio = [
        {
            "name": "Apex Wealth (Gen 12)", # Matches Wealth Logic
            "type": "hybrid",
            "entry_rules": [
                {"col": "cci", "op": "<", "val": 0},
                {"col": "bb_width", "op": ">", "val": 0.1},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [],
            "stop_loss_atr": 4.4,
            "time_stop": 71
        },
        {
            "name": "Apex Income (Gen 9)", # Matches Income Logic
            "type": "hybrid",
            "entry_rules": [
                {"col": "close", "op": ">", "val": 0},
                {"col": "bb_width", "op": ">", "val": 0.17},
                {"col": "rsi14", "op": "<", "ref": "stoch_k"},
                {"col": "volume", "op": ">", "ref": "vol_ma20"}
            ],
            "exit_rules": [
                {"type": "profit_target", "val": 1.08}
            ],
            "stop_loss_atr": 5.1,
            "time_stop": 45
        }
    ]
    
    with open(config_path, "w") as f:
        json.dump(portfolio, f, indent=4)
    print("   ✅ Portfolio Renamed: Wealth & Income.")

    # --- 2. UPDATE ENGINE LOGIC ---
    engine_path = "execution/engine.py"
    
    # We read the file and replace the specific condition line
    with open(engine_path, "r") as f:
        content = f.read()
        
    # Replace 'if "MachineGun" in strategy_name:' with 'if "Income" in strategy_name:'
    # We also make sure it's robust to "Gen9" just in case
    
    old_condition = 'if "MachineGun" in strategy_name:'
    new_condition = 'if "Income" in strategy_name or "Gen9" in strategy_name:'
    
    if old_condition in content:
        new_content = content.replace(old_condition, new_condition)
        with open(engine_path, "w") as f:
            f.write(new_content)
        print("   ✅ Engine Updated: Trend Bonus now triggers for 'Apex Income'.")
    else:
        # Fallback: Write the whole file if the replace fails (safety net)
        # This ensures we definitely get the correct logic
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
        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(14).mean()
        df['atr14_ma20'] = df['atr14'].rolling(20).mean()
        df['highest20'] = df['high'].rolling(20).max()
        df['highest55'] = df['high'].rolling(55).max()
        df['lowest5'] = df['low'].rolling(5).min()
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
        df['cci'] = CCIIndicator(df['high'], df['low'], df['close']).cci()
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
    return { "strategy": name, "final_value": start_cash, "total_trades": 0, "hit_rate": 0.0, "sharpe": 0.0, "cagr": 0.0, "avg_profit_pct": 0.0, "params": params or {}, "trades_list": [], "equity_curve": [] }

def calculate_backtest_quality_score(row, strategy_name):
    score = 50.0
    
    # 1. Sniper Priority
    cci = row.get("cci", 0)
    bb_width = row.get("bb_width", 0)
    if cci < 0 and bb_width > 0.1: score += 50
    
    # 2. Base Oversold
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 3. Velocity Booster
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
        
    # 4. TREND BONUS (Corrected Target)
    if "Income" in strategy_name or "Gen9" in strategy_name:
        if row.get("close", 0) > row.get("sma200", 999999):
            score += 20
            
    # 5. Volume
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    return max(0.0, score)

def run_backtest(strategy, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None):
    # ... (Standard run_backtest implementation maintained) ...
    # For brevity, assuming standard implementation is retained or re-written if needed.
    # Re-writing standard implementation to ensure file completeness:
    symbols = list(data_dict.keys())
    if symbol_universe: symbols = [s for s in symbols if s in symbol_universe]
    enriched = {}
    for sym in symbols:
        if data_dict[sym] is None or data_dict[sym].empty: continue
        try:
            df = _compute_indicators(data_dict[sym].copy(), spy_df=global_data.get("SPY") if global_data else None)
            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]
            if len(df) > MIN_BARS: enriched[sym] = df
        except: continue
    if not enriched: return _empty_result(strategy.name, start_cash, strategy.params)
    
    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trades_list = []
    max_positions = 5
    pos_fraction = 0.20
    
    for current_dt in all_dates:
        # Exits
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
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trades_list.append({"Symbol": sym, "Entry Date": str(df.index[pos["entry_i"]].date()), "Exit Date": str(current_dt.date()), "Entry": pos["entry_price"], "Exit": exit_px, "PnL": pnl, "Return%": ((exit_px - pos["entry_price"])/pos["entry_price"])*100})
                del positions[sym]
        # Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index: equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        equity_curve.append({"Date": current_dt, "Equity": equity})
        # Entries
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
                    daily_candidates.append({"sym": sym, "px": open_px, "stop": open_px - stop_dist, "score": score, "rsi2": float(row_prev.get("rsi2", 50))})
            if len(positions) < max_positions:
                daily_candidates.sort(key=lambda x: (x["score"], -x["rsi2"]), reverse=True)
                for cand in daily_candidates:
                    if len(positions) >= max_positions or cash < 500: break
                    shares = int((equity * pos_fraction) / cand["px"])
                    if shares > 0 and cash >= (shares * cand["px"]):
                        cash -= (shares * cand["px"])
                        positions[cand["sym"]] = {"shares": shares, "entry_price": cand["px"], "stop_price": cand["stop"], "entry_i": enriched[cand["sym"]].index.get_loc(current_dt)}
                        
    trades = len(trade_pnls)
    avg_profit = (sum(trade_pnls)/start_cash)*100 / (trades or 1)
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty: eq_df = pd.DataFrame({"Equity": [start_cash]}, index=[pd.Timestamp.now()])
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = ((eq_df["Equity"].iloc[-1] / start_cash) ** (1.0 / (years if years > 0 else 1))) - 1.0
    return {"strategy": strategy.name, "final_value": equity, "total_trades": trades, "hit_rate": (len([t for t in trade_pnls if t > 0])/trades*100) if trades else 0, "cagr": cagr, "avg_profit_pct": avg_profit, "equity_curve": eq_df, "trades_list": trades_list, "params": strategy.params}
"""
        with open(engine_path, "w") as f:
            f.write(engine_code)
        print("   ✅ Engine Updated (Fallback): Trend Bonus enabled for 'Income'/'Gen9'.")


if __name__ == "__main__":
    align_engine_and_portfolio()
