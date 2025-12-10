
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
from strategies.strategy_loader import load_strategies

DEFAULT_SCORING_WEIGHTS = {
    # --- STRATEGY LAB WINNER (Robust Version) ---
    # Logic: Prioritize Strong Trends + Deep Oversold Panic

    "rsi_factor": 3.0,        # Was 2.0 (AI found 2.94 -> Rounded to 3.0)
    "atr_high_bonus": 12.0,   # Was 15.0 (AI found 12.26 -> Rounded to 12.0)
    "atr_med_bonus": 5.0,     # Unchanged (AI found 4.3 -> Kept baseline for stability)
    "vol_bonus": 11.0,        # Was 10.0 (AI found 10.7 -> Rounded to 11.0)
    "sniper_bonus": 50.0,     # Unchanged (AI found 49.9 -> Kept 50.0)
    "trend_bonus": 40.0,      # Was 20.0 (DOUBLED: Aggressive Trend Filter)
    "trend_penalty": -15.0    # Unchanged
}

MIN_BARS = 200

def get_sector(symbol: str) -> str:
    """
    Placeholder sector mapper. Extend with real classifications when available.
    """
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"

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
    return {
        "strategy": name, "final_value": start_cash, "total_trades": 0,
        "hit_rate": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "cagr": 0.0, "calmar": 0.0, "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0, "avg_days_held": 0.0, "exposure_pct": 0.0,
        "profit_factor": 0.0, "payoff_ratio": 0.0, "max_consecutive_losses": 0,
        "beta": 0.0, "avg_signals_per_day": 0.0, "Score": 0.0, 
        "params": params or {}, "trades_list": [], "equity_curve": []
    }

def calculate_backtest_quality_score(row, strategy_name, weights=None):
    w = DEFAULT_SCORING_WEIGHTS if weights is None else {**DEFAULT_SCORING_WEIGHTS, **weights}
    # --- GOLDEN STATE ENGINE v3.1 (Post-Clamp Scoring) ---
    # 1) Build a base score from shared signals, clamp to 100 to kill bankruptcy bias.
    base_score = 50.0

    rsi2 = row.get("rsi2", 50)
    base_score += (100 - rsi2) * w["rsi_factor"]  # Oversold depth

    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: base_score += w["atr_high_bonus"]
        elif atr_pct > 2.0: base_score += w["atr_med_bonus"]

    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: base_score += w["vol_bonus"]

    clamped_base = max(0.0, min(100.0, base_score))

    # 2) Apply strategy-specific adjustments AFTER the clamp to preserve bonuses/penalties.
    adjustment = 0.0

    wealth_tags = ["Wealth", "Gen12", "Gen 12", "Sniper"]
    if any(tag in strategy_name for tag in wealth_tags):
        cci = row.get("cci", 0)
        bb_width = row.get("bb_width", 0)
        if cci < 0 and bb_width > 0.17:
            adjustment += w["sniper_bonus"]

    income_tags = ["Gen9", "Gen 9", "Income", "Evolved"]
    if any(tag in strategy_name for tag in income_tags):
        sma200 = row.get("sma200", None)
        close_px = row.get("close", 0)
        if sma200 is not None and not pd.isna(sma200):
            if close_px > sma200:
                adjustment += w["trend_bonus"]  # Trend Bonus (Buy Safe Dip)
            else:
                adjustment += w["trend_penalty"]  # Downtrend Penalty (Avoid Falling Knife)

    final_score = clamped_base + adjustment
    return max(0.0, final_score)

def run_backtest(strategies, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, global_data=None, scoring_weights=None):
    import json, os

    # Backward compatibility: allow single strategy input
    if not isinstance(strategies, (list, tuple)):
        strategies = [strategies]
    strategies = [s for s in strategies if s is not None]
    if not strategies:
        return _empty_result("NoStrategy", start_cash, {})

    strategy_label = strategies[0].name if len(strategies) == 1 else "MultiStrategy"

    # Sector map loader
    sector_map = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/sectors.json')), "r") as f:
            sector_map = json.load(f)
    except:
        sector_map = {}

    def resolve_sector(sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in sector_map:
            return sector_map[sym_up]
        return get_sector(sym_up)

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

    if not enriched: return _empty_result(strategy_label, start_cash, strategies[0].params if strategies else {})

    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))
    min_date = pd.Timestamp.now() - pd.Timedelta(days=365*20)
    all_dates = [d for d in all_dates if d >= min_date]
    if not all_dates: return _empty_result(strategy_label, start_cash, strategies[0].params if strategies else {})

    cash = start_cash
    positions = {}
    equity_curve = []
    trade_pnls = []
    trades_list = []
    
    pos_fraction = 0.20
    max_positions = 5

    for current_dt in all_dates:
        # Step A: Update state
        sector_exposure = {}
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                val = pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else:
                val = pos["shares"] * pos["entry_price"]
            sec = resolve_sector(sym)
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + val

        total_equity = cash + sum(sector_exposure.values())

        # Step B: Collect candidates
        daily_candidates = []
        for strat in strategies:
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS + 1: continue
                
                entry_signal = strat.entry(df, i - 1)
                if not entry_signal:
                    continue

                row_prev = df.iloc[i-1]
                row_curr = df.iloc[i]
                raw_score = calculate_backtest_quality_score(row_prev, strat.name, weights=scoring_weights)
                score = raw_score * 1.3 if "wealth" in strat.name.lower() else raw_score
                open_px = float(row_curr["open"])

                signal_atr = float(row_prev.get("atr14", 0))
                current_atr = float(row_curr.get("atr14", signal_atr))
                effective_atr = max(signal_atr, current_atr)
                stop_atr_mult = float(getattr(strat, "params", {}).get("stop_loss_atr", 3.0))
                stop_width = effective_atr * stop_atr_mult
                real_stop = open_px - stop_width

                daily_candidates.append({
                    "sym": sym,
                    "px": open_px,
                    "stop": real_stop,
                    "score": score,
                    "strategy_name": strat.name,
                    "strategy_obj": strat
                })

        # Step C: Sort & Execute (Governor)
        daily_candidates.sort(key=lambda x: x["score"], reverse=True)
        for cand in daily_candidates:
            if cand["sym"] in positions:
                continue
            if len(positions) >= max_positions:
                break

            # Recalculate equity dynamically to shrink sizing as cash is used
            current_sector_equity = sum(sector_exposure.values())
            current_equity = cash + current_sector_equity
            trade_val = current_equity * pos_fraction

            cand_sec = resolve_sector(cand["sym"])
            proj_exp = (sector_exposure.get(cand_sec, 0.0) + trade_val) / current_equity if current_equity > 0 else 1.0
            if proj_exp > 0.60:
                continue

            shares = int(trade_val / cand["px"])
            cost = shares * cand["px"]
            if shares <= 0 or cash < cost:
                continue

            cash -= cost
            positions[cand["sym"]] = {
                "shares": shares,
                "entry_price": cand["px"],
                "stop_price": cand["stop"],
                "entry_i": enriched[cand["sym"]].index.get_loc(current_dt),
                "strategy_name": cand["strategy_name"],
                "strategy_obj": cand["strategy_obj"]
            }
            sector_exposure[cand_sec] = sector_exposure.get(cand_sec, 0.0) + cost

        # Step D: Process exits
        for sym in list(positions.keys()):
            if sym not in enriched or current_dt not in enriched[sym].index:
                continue
            df = enriched[sym]
            i = df.index.get_loc(current_dt)
            pos = positions[sym]
            if i <= pos["entry_i"]:
                continue

            active_strat = pos.get("strategy_obj")
            if active_strat and active_strat.exit(df, i, pos["entry_i"], pos["entry_price"], pos["stop_price"]):
                row = df.iloc[i]
                exit_px = pos["stop_price"] if row["low"] < pos["stop_price"] else row["close"]
                if row["low"] < pos["stop_price"] and row["open"] < pos["stop_price"]:
                    exit_px = row["open"]
                if exit_px < pos["entry_price"] * 0.5:
                    exit_px = pos["entry_price"] * 0.5

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
                    "Return%": pct,
                    "Strategy": pos.get("strategy_name", strategy_label)
                })
                del positions[sym]

        # Equity snapshot after exits
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else:
                equity += pos["shares"] * pos["entry_price"]
        equity_curve.append({"Date": current_dt, "Equity": equity})

    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    hit_rate = (wins/trades*100) if trades > 0 else 0.0
    if trades_list:
        avg_profit = sum(t.get("Return%", 0.0) for t in trades_list) / len(trades_list)
    else:
        avg_profit = 0.0
    
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty: eq_df = pd.DataFrame({"Equity": [start_cash]}, index=[pd.Timestamp.now()])
    
    days = (eq_df.index[-1] - eq_df.index[0]).days
    years = days / 365.25
    cagr = ((eq_df["Equity"].iloc[-1] / start_cash) ** (1.0 / (years if years > 0 else 1))) - 1.0
    
    rolling_max = eq_df["Equity"].cummax()
    drawdowns = (eq_df["Equity"] - rolling_max) / rolling_max
    max_dd = drawdowns.min() * 100

    return {
        "strategy": strategy_label,
        "final_value": equity,
        "total_trades": trades,
        "hit_rate": hit_rate,
        "cagr": cagr,
        "avg_profit_pct": avg_profit,
        "max_drawdown_pct": max_dd,
        "equity_curve": eq_df,
        "trades_list": trades_list,
        "params": strategies[0].params if strategies else {},
        "Score": cagr * 1000
    }

def run_compare(strategy_names, data_dict, symbol_universe=None, start_cash=100000.0, start_date=None, use_parallel=True, max_workers=8, global_data=None):
    import json, os, concurrent.futures

    gen_strategies = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json')), "r") as f:
            for g in json.load(f): gen_strategies[g["name"]] = g
    except: pass

    strategies = load_strategies([gen_strategies[name] for name in strategy_names if name in gen_strategies])

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat for strat in strategies}
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except: pass
    if not results: return pd.DataFrame()
    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['equity_curve', 'trades_list']} for r in results])
    return df.sort_values("Score", ascending=False)
