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

st.set_page_config(page_title="Apex Sniper Screener", layout="wide", page_icon="🎯")
st.sidebar.title("🎯 Apex Sniper")
mode = st.sidebar.radio("Mode", ["Live Screener", "Strategy Lab"])

# --- Helper: Robust Data Fetching ---
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

# --- 1. Live Screener ---
if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox("1. Select Universe", ["S&P 500", "S&P 1500", "S&P 100", "Nasdaq 100"])
    with col2:
        # Load Strategies
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strats = json.load(f)
                for s in gen_strats: strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        
        selected_names = st.multiselect("2. Select Strategies (or leave empty for ALL)", list(strategies.keys()))
        if not selected_names: selected_names = list(strategies.keys())

    if st.button("Run Scan"):
        st.info(f"Scanning {len(selected_names)} strategies against {universe}...")
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        
        progress = st.progress(0)
        all_hits = []
        
        for i, sym in enumerate(symbols):
            if i % 10 == 0: progress.progress(i / len(symbols))
            
            # Optimized Cache Fetch
            df = DataCache.get_cached_data(sym)
            if df is None or len(df) < 200: continue # Skip uncached in UI to be fast
            
            try:
                df = _compute_indicators(df)
                # Inject Context
                if global_data["SPY"] is not None:
                    spy_re = global_data["SPY"]["close"].reindex(df.index, method="ffill")
                    df["rs_ratio"] = df["close"] / spy_re
                    df["rs_sma20"] = df["rs_ratio"].rolling(20).mean()
                    df["rs_trend"] = (df["rs_ratio"] > df["rs_sma20"]).astype(int)
                else: df["rs_trend"] = 0
                
                if global_data["VIX"] is not None:
                    df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill")
                else: df["vix"] = 20.0

                last_idx = len(df) - 1
                
                # Check ALL selected strategies
                for strat_name in selected_names:
                    strat_config = strategies[strat_name]
                    strat_obj = GenericStrategy(strat_config)
                    signal = strat_obj.entry(df, last_idx)
                    
                    if signal:
                        row = df.iloc[-1]
                        all_hits.append({
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": row["close"],
                            "Stop Loss": signal["stop_price"],
                            "ATR": row["atr14"],
                            "VIX": row["vix"]
                        })
            except: pass

        progress.progress(100)
        
        if all_hits:
            st.success(f"Found {len(all_hits)} Trade Setups!")
            st.dataframe(pd.DataFrame(all_hits))
        else:
            st.warning("No setups found.")

# --- 2. Strategy Lab (Backtester) ---
elif mode == "Strategy Lab":
    st.header("🧪 Strategy Lab")
    
    col1, col2 = st.columns(2)
    with col1:
        # FIXED: Universe Selector for Backtest
        bt_universe = st.selectbox("1. Backtest Universe", ["S&P 100", "S&P 500", "S&P 1500"])
    with col2:
        gen_strategies = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strategies = [s["name"] for s in json.load(f)]
        except: pass
        bt_strategies = st.multiselect("2. Select Strategies", gen_strategies, default=gen_strategies[:3] if gen_strategies else None)

    if st.button("Run Backtest"):
        symbols = get_index_symbols(bt_universe)
        data_map = {}
        global_data = get_global_data(days=1260)
        
        with st.status("Loading Data from Cache...") as status:
            for sym in symbols:
                df = DataCache.get_cached_data(sym)
                if df is not None and len(df) > 200:
                    data_map[sym] = df
            status.update(label=f"Running Backtest on {len(data_map)} symbols...", state="running")
            
            # RUN ENGINE
            results_df = run_compare(bt_strategies, data_map, symbols, start_cash=100000.0, global_data=global_data)
            status.update(label="Complete!", state="complete")
        
        if not results_df.empty:
            # Format for readability
            st.dataframe(results_df.style.format({
                "hit_rate": "{:.1f}%",
                "avg_profit_pct": "{:.2f}%",
                "cagr": "{:.1f}%",
                "Score": "{:.1f}",
                "profit_factor": "{:.2f}",
                "payoff_ratio": "{:.2f}",
                "avg_days_held": "{:.1f}d"
            }))
        else:
            st.error("Backtest returned no results.")