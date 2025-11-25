import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(__file__))
from data.indices import get_index_symbols
from data.loader import fetch_single_symbol, fetch_data_pack
from execution.engine import run_backtest, run_compare, _compute_indicators
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper", layout="wide")
st.sidebar.title("🎯 Apex Sniper")

mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])

@st.cache_data(ttl=3600)
def get_global_data(days=400):
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX") or g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}

if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    col1, col2 = st.columns(2)
    with col1: universe = st.selectbox("Universe", ["S&P 1500", "S&P 500", "S&P 100"])
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        selected_names = st.multiselect("Strategies", list(strategies.keys()), default=list(strategies.keys()))

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        progress = st.progress(0)
        hits = []
        for i, sym in enumerate(symbols):
            if i % 10 == 0: progress.progress(i / len(symbols))
            df = fetch_single_symbol(sym, days=1260, force_fresh=True)
            if df is None: continue 
            try:
                df = _compute_indicators(df)
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
                        hits.append({"Strategy": strat_name, "Symbol": sym, "Price": f"${row['close']:.2f}", "Stop": f"${signal['stop_price']:.2f}", "Target": f"${target:.2f}"})
            except: pass
        progress.progress(100)
        if hits: st.dataframe(pd.DataFrame(hits))
        else: st.warning("No setups found.")

elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    col1, col2, col3 = st.columns(3)
    with col1: bt_univ = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"])
    with col2:
        gen_strats = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strats = [s["name"] for s in json.load(f)]
        except: pass
        bt_strats = st.multiselect("Strategies", gen_strats, default=gen_strats)
    with col3: tf = st.selectbox("Timeframe", ["1 Year", "5 Years", "Max"])

    if st.button("Run Backtest"):
        d_map = {"1 Year": 365, "5 Years": 1260, "Max": 5000}
        start_date = (datetime.now(timezone.utc) - timedelta(days=d_map.get(tf, 1260))).date()
        
        with st.status("Loading Data...") as status:
            symbols = get_index_symbols(bt_univ)
            data_map = fetch_data_pack(symbols, days=d_map.get(tf, 1260))
            global_data = get_global_data(days=1260)
            status.update(label="Running Simulation...", state="running")
            res = run_compare(bt_strats, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_data)
            status.update(label="Complete!", state="complete")
        
        if not res.empty:
            st.dataframe(res.style.format({
                "hit_rate": "{:.1f}%", "avg_profit_pct": "{:.2f}%", "cagr": "{:.1f}%", 
                "Score": "{:.1f}", "profit_factor": "{:.2f}", "payoff_ratio": "{:.2f}",
                "avg_days_held": "{:.1f}d"
            }))
            
            # NEW: Trade Inspector
            st.subheader("🧐 Trade Inspector (Top Strategy)")
            if hasattr(res, "trade_logs"):
                top_strat = res.iloc[0]["strategy"]
                logs = res.trade_logs.get(top_strat, [])
                if logs:
                    st.write(f"Showing recent trades for: **{top_strat}**")
                    st.dataframe(pd.DataFrame(logs).tail(50))
                else:
                    st.warning("No trades recorded for top strategy.")
        else: st.error("No trades found.")
