import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(__file__))
from data.schwab_client import sd
from data.cache_manager import DataCache
from data.indices import get_index_symbols
from execution.engine import run_backtest, run_compare, _compute_indicators
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper", layout="wide")
st.sidebar.title("🎯 Apex Sniper")
mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])

@st.cache_data(ttl=3600)
def get_global_data(days=400):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50)
    g_data = {}
    for sym in ["SPY", "$VIX", "VIX"]:
        try:
            df = DataCache.get_cached_data(sym)
            if df is None:
                candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
                if candles:
                    df = pd.DataFrame(candles)
                    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                    df = df.set_index("datetime").sort_index()
                    DataCache.save_to_cache(sym, df)
            if df is not None and not df.empty: g_data[sym] = df
        except: pass
    vix = g_data.get("$VIX") if "$VIX" in g_data else g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}

if mode == "Live Screener":
    st.header("🚀 Live Screener")
    col1, col2 = st.columns(2)
    with col1: universe = st.selectbox("Universe", ["S&P 500", "S&P 1500", "S&P 100"])
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        selected = st.multiselect("Strategies", list(strategies.keys()), default=list(strategies.keys())[:1])

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        progress = st.progress(0)
        hits = []
        
        for i, sym in enumerate(symbols):
            if i % 10 == 0: progress.progress(i / len(symbols))
            df = DataCache.get_cached_data(sym)
            if df is None or len(df) < 200: continue
            
            try:
                df = _compute_indicators(df)
                # Inject Context
                if global_data["VIX"] is not None:
                    df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill").fillna(20.0)
                else: df["vix"] = 20.0
                
                last_idx = len(df) - 1
                for s_name in selected:
                    strat = GenericStrategy(strategies[s_name])
                    signal = strat.entry(df, last_idx)
                    if signal:
                        hits.append({"Symbol": sym, "Strategy": s_name, "Price": df.iloc[-1]["close"]})
            except: pass
        
        progress.progress(100)
        if hits: st.dataframe(pd.DataFrame(hits))
        else: st.warning("No setups found.")

elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    bt_universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"])
    
    if st.button("Run Backtest"):
        symbols = get_index_symbols(bt_universe)
        data_map = {}
        global_data = get_global_data(days=1260)
        
        with st.status("Loading Data...") as status:
            for sym in symbols:
                df = DataCache.get_cached_data(sym)
                if df is not None and len(df) > 200: data_map[sym] = df
            
            # Pass ALL generated strategies
            strat_names = []
            with open("config/generated_strategies.json", "r") as f:
                strat_names = [s["name"] for s in json.load(f)]

            results = run_compare(strat_names, data_map, symbols, start_cash=100000.0, global_data=global_data)
            status.update(label="Done!", state="complete")
        
        if not results.empty:
            st.dataframe(results[["strategy", "cagr", "hit_rate", "Score", "total_trades", "avg_days_held"]])
        else:
            st.error("No results.")