"""Upgrade engine outputs and Streamlit UI with charts and multi-period metrics."""

import os


def upgrade_full_ui():
    print("🚀 UPGRADING SYSTEM UI & ENGINE DATA FEED...")

    # --- 1. PATCH ENGINE (Expose Data) ---
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
        "params": params or {}, "trades_list": [], "equity_curve": [], "trades_df": pd.DataFrame()
    }

def calculate_backtest_quality_score(row, strategy_name):
    # GOLDEN STATE LOGIC (Unclamped, Bifurcated)
    score = 50.0
    
    # 1. Base
    rsi2 = row.get("rsi2", 50)
    score += (100 - rsi2) * 2.0 
    
    # 2. Velocity
    close_px = row.get("close", 1.0)
    if close_px > 0:
        atr_pct = (row.get("atr14", 0) / close_px) * 100
        if atr_pct > 3.0: score += 15
        elif atr_pct > 2.0: score += 5
        
    # 3. Trend (Machine Gun Only)
    if "MachineGun" in strategy_name:
        if row.get("close", 0) > row.get("sma200", 999999):
            score += 20
    
    # 4. Volume
    vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
    if vol_rel > 1.5: score += 10
    
    # 5. Sniper Priority
    # If CCI < 0 and BB Width > 0.1 (Gen 12 DNA), huge boost
    cci = row.get("cci", 0)
    bb = row.get("bb_width", 0)
    if cci < 0 and bb > 0.1:
        score += 50

    return max(0.0, score) # Unclamped

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
    trades_list = [] # Detailed log
    
    days_invested = 0
    max_positions = 5
    raw_pos = strategy.params.get('pos_size', strategy.params.get('position_size', strategy.params.get('pos_fraction', 0.20)))
    try: pos_fraction = float(raw_pos)
    except: pos_fraction = 0.20
    pos_fraction = max(0.01, min(pos_fraction, 1.0))

    for current_dt in all_dates:
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
                if exit_px < pos["entry_price"] * 0.5: exit_px = pos["entry_price"] * 0.5 # Max Gap Protection
                
                pnl = (exit_px - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * exit_px
                trade_pnls.append(pnl)
                
                trades_list.append({
                    "Symbol": sym, "Entry Date": str(df.index[pos["entry_i"]].date()), 
                    "Exit Date": str(current_dt.date()), "Entry": pos["entry_price"], 
                    "Exit": exit_px, "PnL": pnl, 
                    "Return%": ((exit_px - pos["entry_price"])/pos["entry_price"])*100
                })
                del positions[sym]

        # 2. Equity
        equity = cash
        for sym, pos in positions.items():
            if sym in enriched and current_dt in enriched[sym].index:
                equity += pos["shares"] * enriched[sym].loc[current_dt]["close"]
            else: equity += pos["shares"] * pos["entry_price"]
        
        # Store Equity Tuple (Date, Value) for easier DF creation later
        equity_curve.append({"Date": current_dt, "Equity": equity})

        # 3. Entries (Signal on Close i-1, Buy on Open i)
        if cash > 0:
            daily_candidates = []
            for sym, df in enriched.items():
                if current_dt not in df.index or sym in positions: continue
                i = df.index.get_loc(current_dt)
                if i < MIN_BARS + 1: continue
                
                # Check YESTERDAY'S Signal
                if strategy.entry(df, i - 1):
                    row_prev = df.iloc[i-1]
                    row_curr = df.iloc[i]
                    
                    score = calculate_backtest_quality_score(row_prev, strategy.name)
                    open_px = float(row_curr["open"])
                    
                    # Stop logic
                    stop_dist = float(row_prev["close"]) - strategy.entry(df, i-1)["stop_price"]
                    real_stop = open_px - stop_dist
                    
                    daily_candidates.append({
                        "sym": sym, "px": open_px, "stop": real_stop, 
                        "score": score, "rsi2": float(row_prev.get("rsi2", 50))
                    })
            
            if len(positions) < max_positions:
                # Tie Breaker: Score -> RSI2
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

    # Final Stats
    trades = len(trade_pnls)
    wins = len([t for t in trade_pnls if t > 0])
    hit_rate = (wins/trades*100) if trades > 0 else 0.0
    avg_profit = (sum(trade_pnls)/start_cash)*100 / (trades or 1) # Approx yield
    
    # Convert Equity Curve to Series
    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty: eq_df = pd.DataFrame({"Equity": [start_cash]}, index=[pd.Timestamp.now()])
    
    # CAGR Calc
    days = (eq_df.index[-1] - eq_df.index[0]).days
    years = days / 365.25
    cagr = ((eq_df["Equity"].iloc[-1] / start_cash) ** (1.0 / (years if years > 0 else 1))) - 1.0

    return {
        "strategy": strategy.name, "final_value": equity, "total_trades": trades,
        "hit_rate": hit_rate, "cagr": cagr, "avg_profit_pct": avg_profit,
        "equity_curve": eq_df, # RETURNS FULL DATAFRAME NOW
        "trades_list": trades_list,
        "trades_df": pd.DataFrame(trades_list),
        "params": strategy.params,
        "Score": cagr * 1000
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
            
    # Note: run_compare returns summary DF, but app will call run_backtest directly for detailed plots
    if not results: return pd.DataFrame()
    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ['equity_curve', 'trades_list', 'trades_df']} for r in results])
    return df.sort_values("Score", ascending=False)
"""
    with open(engine_path, "w") as f:
        f.write(engine_code)
    print("   ✅ Engine Patched: Equity Curve and Trade Lists now exposed.")

    # --- 2. BUILD THE UI (App.py) ---
    app_path = "app.py"
    app_code = """
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime
import plotly.express as px

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest, _compute_indicators
from strategies.generic import GenericStrategy
from simulation.paper_trader import PaperTrader

CONFIG_PATH = "config/generated_strategies.json"
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")

# --- HELPERS ---
def load_strategies():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return []

def format_rule(r):
    return f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}"

def calc_cagr(start_val, end_val, years):
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1.0 / years)) - 1.0

def run_scanner(strategies, universe):
    symbols = get_index_symbols(universe)
    data = fetch_data_pack(symbols, days=400)
    results = []
    for s_conf in strategies:
        strat = GenericStrategy(s_conf)
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            try:
                df_ind = _compute_indicators(df.copy())
                if df_ind.empty:
                    continue
                if strat.entry(df_ind, len(df_ind) - 1):
                    row = df_ind.iloc[-1]
                    atr = row.get("atr14", row["close"] * 0.02)
                    stop_mult = float(s_conf.get("stop_loss_atr", 3.0))
                    exits = s_conf.get("exit_rules", [])
                    target_txt = "OPEN (Run)"
                    if exits and exits[0].get("type") == "profit_target":
                        t_price = row["close"] * float(exits[0].get("val"))
                        target_txt = f"${t_price:.2f}"
                    results.append({
                        "Symbol": sym,
                        "Strategy": s_conf["name"],
                        "Price": row["close"],
                        "Stop Loss": row["close"] - (atr * stop_mult),
                        "Target": target_txt
                    })
            except:
                continue
    return pd.DataFrame(results)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    
    st.markdown("### 📘 Active Strategies")
    strategies = load_strategies()
    selected_strategies = []
    
    if strategies:
        for s in strategies:
            use = st.checkbox(s.get('name'), value=True)
            if use: selected_strategies.append(s)
            
            with st.expander("Details"):
                st.write(f"**Role:** {s.get('type')}")
                st.write(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.write(f"**Time:** {s.get('time_stop')} Days")
                target_txt = s.get('exit_rules')[0]['val'] if s.get('exit_rules') else 'NONE (Run)'
                st.write(f"**Target:** {target_txt}")
                st.write("**Entry Rules:**")
                for r in s.get('entry_rules', []):
                    st.code(format_rule(r))
    else:
        st.error("No strategies found!")

# --- LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500"], index=1)
        run_btn = st.button("RUN SCAN", type="primary")
        
    if run_btn:
        with st.spinner("Scanning market..."):
            df_res = run_scanner(selected_strategies, universe)
            st.session_state.scan_results = df_res

    if "scan_results" in st.session_state:
        df = st.session_state.scan_results
        if df is not None and not df.empty:
            dupes = df[df.duplicated(subset=['Symbol'], keep=False)]
            if not dupes.empty:
                st.success(f"🔥 SUPER SIGNALS: {len(dupes['Symbol'].unique())} High Conviction Trades")
                st.dataframe(dupes)
            
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "apex_scan.csv", "text/csv")
        elif df is not None:
            st.info("No signals found.")

# --- BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
    duration = st.selectbox("Backtest Duration", ["Max History (Full)", "5 Years", "2 Years"])
    if st.button("Run Backtest"):
        with st.spinner("Simulating strategies..."):
            symbols = get_index_symbols("S&P 500")
            data = fetch_data_pack(symbols, days=5000)
            
            tabs = st.tabs([s["name"] for s in selected_strategies])
            
            for i, s_conf in enumerate(selected_strategies):
                with tabs[i]:
                    res = run_backtest(GenericStrategy(s_conf), data)
                    
                    eq_curve = res["equity_curve"]
                    start_val = eq_curve.iloc[0]["Equity"]
                    curr_val = eq_curve.iloc[-1]["Equity"]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{((curr_val/start_val)-1)*100:.1f}%")
                    col2.metric("CAGR (Max)", f"{res['cagr']*100:.1f}%")
                    col3.metric("Trades", res['total_trades'])
                    
                    periods = [1, 3, 5, 10, 20]
                    cagr_data = []
                    today = eq_curve.index[-1]
                    
                    for y in periods:
                        target_date = today - pd.DateOffset(years=y)
                        if target_date in eq_curve.index:
                            past_val = eq_curve.loc[target_date]["Equity"]
                        elif not eq_curve[eq_curve.index <= target_date].empty:
                            past_val = eq_curve[eq_curve.index <= target_date].iloc[-1]["Equity"]
                        else:
                            past_val = None
                            
                        if past_val:
                            cagr = calc_cagr(past_val, curr_val, y)
                            cagr_data.append({"Period": f"{y} Year", "CAGR": f"{cagr*100:.1f}%"})
                        else:
                            cagr_data.append({"Period": f"{y} Year", "CAGR": "N/A"})
                            
                    st.table(pd.DataFrame(cagr_data))
                    
                    st.line_chart(eq_curve)
                    
                    trades_df = pd.DataFrame(res["trades_list"])
                    if not trades_df.empty:
                        csv_t = trades_df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"📥 Download Trades ({s_conf['name']})", csv_t, f"trades_{s_conf['name']}.csv", "text/csv")

# --- SIMULATOR ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader")
    trader = PaperTrader()
    
    col1, col2 = st.columns(2)
    col1.metric("Cash", f"${trader.state['cash']:,.2f}")
    col2.metric("Equity", f"${trader.state['equity']:,.2f}")
    
    if st.button("🔄 Run Daily Cycle"):
        state, fills = trader.update_valuations()
        for f in fills: st.success(f)
        
        strategies = load_strategies()
        df_scan = run_scanner(strategies, "S&P 1500")
        if not df_scan.empty:
            cands = df_scan.to_dict('records')
            clean_cands = [{"Symbol": c["Symbol"], "Strategy": c["Strategy"], "Stop": c["Stop Loss"], "Raw_Score": 100} for c in cands]
            logs = trader.execute_entries(clean_cands)
            for l in logs: st.write(l)
            
    st.subheader("Pending Orders (Next Open)")
    st.dataframe(pd.DataFrame(trader.state.get("pending_orders", [])))
    
    st.subheader("Holdings")
    st.dataframe(pd.DataFrame.from_dict(trader.state["positions"], orient='index'))
"""
    
    with open(app_path, "w") as f:
        f.write(app_code)
    print("   ✅ App Upgraded: Charts, Multi-Period CAGRs, and Full Control.")

if __name__ == "__main__":
    upgrade_full_ui()
