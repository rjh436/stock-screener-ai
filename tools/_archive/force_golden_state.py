import json
import os

def force_golden_state():
    print("⚡ FORCE-RESTORING GOLDEN STATE (Manual Reconstruction)...")

    # --- 1. REWRITE STRATEGY CONFIGURATION ---
    # We hardcode the exact winning parameters from your CSV reports.
    config_path = "config/generated_strategies.json"
    
    strategies = [
        {
            "name": "Strategy_Apex_Gen12_Sniper",
            "type": "hybrid",
            "entry_rules": [
                {"col": "adx", "op": "<", "ref": "rsi14"},
                {"col": "volume", "op": ">", "val": 0},
                {"col": "sma50", "op": ">", "val": 0}
            ],
            "exit_rules": [], # NO Profit Target (Let it run for 18% gains)
            "stop_loss_atr": 4.4,
            "time_stop": 71 
        },
        {
            "name": "Strategy_Apex_Alpha_MachineGun",
            "type": "hybrid",
            "entry_rules": [
                {"col": "adx", "op": "<", "ref": "rsi14"},
                {"col": "volume", "op": ">", "val": 0},
                {"col": "sma50", "op": ">", "val": 0}
            ],
            "exit_rules": [
                {"type": "profit_target", "val": 1.06} # 6% Bank (Velocity)
            ],
            "stop_loss_atr": 4.3,
            "time_stop": 10
        }
    ]
    
    with open(config_path, "w") as f:
        json.dump(strategies, f, indent=4)
    print("   ✅ Config restored: Gen12 (Long) + MachineGun (Short).")

    # --- 2. REWRITE RANKING ENGINE (UNIFIED VELOCITY) ---
    # We restore the 'Soft Trend + Velocity' logic that worked for BOTH strategies.
    engine_path = "execution/engine.py"
    
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
        "params": params or {}, "trades_list": []
    }

def calculate_backtest_quality_score(row, strategy_name):
    # UNIFIED VELOCITY RANKING (The Logic that produced 49% CAGR)
    score = 50.0
    
    # 1. Base Oversold (Dip Buying)
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 2. Velocity Booster (Prioritize High Volatility)
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
        
    # 3. Smart Trend Filter (Soft Filter)
    # Give a bonus to uptrends, but DO NOT filter out downtrends.
    # Gen 12 needs to be able to buy Deep Value crashes (below SMA200).
    if row.get("close", 0) > row.get("sma200", 999999):
        score += 20
    
    # 4. Volume Support
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    return max(0.0, min(100.0, score))

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
    trade_returns = []
    trade_durations = []
    trades_list = []
    daily_signal_counts = []
    
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
                if exit_px > (pos["entry_price"] * 5.0): exit_px = pos["entry_price"]

                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100
                
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                trade_returns.append(pct)
                trade_durations.append(i - pos["entry_i"])
                
                trades_list.append({
                    "Symbol": sym, "Entry": pos["entry_price"], "Exit": exit_px, 
                    "PnL": pnl, "Return%": pct, "Date": str(current_dt.date())
                })
                del positions[sym]

        # 2. Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        equity_curve.append(equity)

        # 3. Entries (RANKED)
        if cash > 0:
            daily_candidates = []
            
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS: continue
                
                entry = strategy.entry(df, i)
                if entry:
                    row = df.iloc[i]
                    quality = calculate_backtest_quality_score(row, strategy.name)
                    daily_candidates.append({
                        "sym": sym, "entry": entry, "score": quality, 
                        "px": float(row["open"]), "rsi2": float(row.get("rsi2", 50))
                    })
            
            daily_signal_counts.append(len(daily_candidates))
            
            if len(positions) < max_positions:
                daily_candidates.sort(key=lambda x: x["score"], reverse=True)
                target_size = equity * pos_fraction
                for cand in daily_candidates:
                    if len(positions) >= max_positions or cash < 500: break
                    px = cand["px"]
                    if px < 5.0: continue
                    shares = int(target_size / px)
                    cost = shares * px
                    if shares > 0 and cash >= cost:
                        cash -= cost
                        positions[cand["sym"]] = {
                            "shares": shares, "entry_price": px, 
                            "stop_price": float(cand["entry"]["stop_price"]), 
                            "entry_i": enriched[cand["sym"]].index.get_loc(current_dt)
                        }

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
    
    sim_days = (all_dates[-1] - all_dates[0]).days if len(all_dates) > 1 else 1
    years = sim_days / 365.25 if sim_days > 0 else 0
    exposure_pct = (days_invested / len(all_dates) * 100.0) if len(all_dates) > 0 else 0.0
    avg_days_held = float(np.mean(trade_durations)) if trade_durations else 0.0
    avg_signals_per_day = float(np.mean(daily_signal_counts)) if daily_signal_counts else 0.0

    cagr = ((final_val / start_cash) ** (1.0 / years)) - 1.0 if (final_val > 0 and years > 0) else 0.0
    
    eq = pd.Series(equity_curve if equity_curve else [start_cash], index=all_dates)
    returns = eq.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * math.sqrt(252)) if not returns.empty and returns.std() > 0 else 0.0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (returns.mean() / downside_std * math.sqrt(252)) if (not returns.empty and downside_std > 0) else 0.0

    if not eq.empty and eq.max() > 0:
        rolling_max = eq.cummax()
        drawdowns = (eq - rolling_max) / rolling_max
        max_drawdown_pct = drawdowns.min() * 100.0
    else: max_drawdown_pct = 0.0
    
    calmar = (cagr / abs(max_drawdown_pct / 100.0)) if max_drawdown_pct < 0 else 0.0

    max_cons_losses = 0
    current_streak = 0
    for pnl in trade_pnls:
        if pnl < 0:
            current_streak += 1
            if current_streak > max_cons_losses: max_cons_losses = current_streak
        else: current_streak = 0

    beta = 0.0
    if global_data and "SPY" in global_data:
        try:
            spy_df = global_data["SPY"].copy()
            if not spy_df.empty:
                if spy_df.index.tz is not None: spy_df.index = spy_df.index.tz_localize(None)
                aligned_spy = spy_df["close"].reindex(eq.index).ffill().bfill()
                spy_returns = aligned_spy.pct_change().dropna()
                common = returns.index.intersection(spy_returns.index)
                if len(common) > 10:
                    strat_res = returns.loc[common]
                    mkt_res = spy_returns.loc[common]
                    cov = strat_res.cov(mkt_res)
                    var = mkt_res.var()
                    if var > 0: beta = cov / var
        except: pass

    score = (cagr * 200) + (win_rate * 2) + (sharpe * 20) + (avg_profit * 50)

    return {
        "strategy": strategy.name, "final_value": final_val, "total_trades": trades,
        "hit_rate": win_rate, "sharpe": sharpe, "sortino": sortino, "cagr": cagr,
        "calmar": calmar, "max_drawdown_pct": max_drawdown_pct, 
        "avg_profit_pct": avg_profit, "avg_days_held": avg_days_held, "exposure_pct": exposure_pct,
        "profit_factor": profit_factor, "payoff_ratio": payoff_ratio, "max_consecutive_losses": max_cons_losses,
        "beta": beta, "avg_signals_per_day": avg_signals_per_day,
        "Score": score, "trades_list": trades_list, "params": strategy.params
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
            except: pass
    if not results: return pd.DataFrame()
    df = pd.DataFrame(results)
    df.trade_logs = {r['strategy']: r.get('trades_list', []) for r in results}
    return df.sort_values("Score", ascending=False)
"""
    
    with open(engine_path, "w") as f:
        f.write(engine_code)
    print("   ✅ Engine restored to Unified Velocity Ranking.")

    print("\n🚀 GOLDEN STATE RESTORED.")
    print("   Please Run Backtest to confirm:")
    print("   - Gen 12 Sniper: ~49 Trades, ~18% Profit, ~49% CAGR")
    print("   - Machine Gun: ~400 Trades, ~1.5% Profit, ~43% CAGR")

if __name__ == "__main__":
    force_golden_state()
