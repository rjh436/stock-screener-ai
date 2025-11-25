import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(__file__))
from data.schwab_client import sd
from data.indices import get_index_symbols
from data.loader import fetch_single_symbol, fetch_data_pack
from execution.engine import run_backtest, run_compare, _compute_indicators
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper", layout="wide", page_icon="🎯")
st.sidebar.title("🎯 Apex Sniper")

# --- API Status ---
with st.sidebar.expander("🔌 API Status", expanded=False):
    if st.button("Test Connection"):
        res = sd.health_check("VOO")
        if res.get("ok"): st.success("Online")
        else: st.error("Offline")

mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])

@st.cache_data(ttl=3600)
def get_global_data(days=400):
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    
    # FIXED: Explicit check to avoid DataFrame Ambiguity Error
    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
        
    return {"SPY": g_data.get("SPY"), "VIX": vix}

# --- 1. Live Screener ---
if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    
    col1, col2 = st.columns(2)
    with col1:
        # Default: S&P 1500
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500", "S&P 100"], index=1)
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        
        # Default: ALL Strategies
        all_names = list(strategies.keys())
        selected_names = st.multiselect("Strategies", all_names, default=all_names)

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        # Fetch Global Data First
        global_data = get_global_data(days=1260)
        
        progress = st.progress(0, text="Starting Scan...")
        status_text = st.empty()
        all_hits = []
        
        for i, sym in enumerate(symbols):
            if i % 5 == 0: 
                progress.progress(i / len(symbols))
                status_text.text(f"Scanning {sym}...")
            
            # FORCE FRESH: Ensures we get today's latest bar
            df = fetch_single_symbol(sym, days=1260, force_fresh=True)
            if df is None: continue 
            
            try:
                df = _compute_indicators(df)
                
                # Inject Context
                if global_data["VIX"] is not None:
                    df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill").fillna(20.0)
                else: df["vix"] = 20.0
                
                last_idx = len(df) - 1
                for strat_name in selected_names:
                    strat = GenericStrategy(strategies[strat_name])
                    signal = strat.entry(df, last_idx)
                    if signal:
                        row = df.iloc[-1]
                        target = row["close"] * 1.10
                        for r in strategies[strat_name].get("exit_rules", []):
                            if r.get("type") == "profit_target": target = row["close"] * r["val"]
                        
                        all_hits.append({
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": f"${row['close']:.2f}",
                            "Stop": f"${signal['stop_price']:.2f}",
                            "Target": f"${target:.2f}",
                            "Risk": f"{(1 - signal['stop_price']/row['close'])*100:.1f}%"
                        })
            except: pass

        progress.progress(100)
        status_text.text("Complete!")
        
        if all_hits:
            st.success(f"Found {len(all_hits)} Trade Setups!")
            st.dataframe(pd.DataFrame(all_hits))
        else:
            st.warning("No setups found.")

# --- 2. Backtest ---
elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    
    col1, col2, col3 = st.columns(3)
    with col1: bt_universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=2)
    with col2:
        gen_strategies = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strategies = [s["name"] for s in json.load(f)]
        except: pass
        bt_strategies = st.multiselect("Strategies", gen_strategies, default=gen_strategies)
    with col3: timeframe = st.selectbox("Timeframe", ["1 Year", "5 Years", "Max"], index=1)

    if st.button("Run Backtest"):
        days_map = {"1 Year": 365, "5 Years": 1260, "Max": 10000}
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_map.get(timeframe, 1260))).date()
        
        with st.status("Loading Data...") as status:
            symbols = get_index_symbols(bt_universe)
            # Backtest = Cached Data (Fast)
            data_map = fetch_data_pack(symbols, days=days_map.get(timeframe, 1260))
            global_data = get_global_data(days=1260)
            
            status.update(label=f"Backtesting {len(data_map)} symbols...", state="running")
            results = run_compare(bt_strategies, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_data)
            status.update(label="Complete!", state="complete")
        
        if not results.empty:
            st.dataframe(results.style.format({
                "hit_rate": "{:.1f}%", "avg_profit_pct": "{:.2f}%", "cagr": "{:.1f}%", 
                "Score": "{:.1f}", "profit_factor": "{:.2f}", "payoff_ratio": "{:.2f}",
                "avg_days_held": "{:.1f}d"
            }))
        else:
            st.error("No trades found.")
