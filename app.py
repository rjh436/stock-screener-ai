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
    st.header("🚀 Live Market Screener")
    col1, col2 = st.columns(2)
    with col1: universe = st.selectbox("Universe", ["S&P 1500", "S&P 500", "S&P 100"], index=0)
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        
        all_strat_names = list(strategies.keys())
        use_all = st.checkbox("Select All Strategies", value=True)
        if use_all:
            selected_names = all_strat_names
        else:
            selected_names = st.multiselect("Strategies", all_strat_names, default=all_strat_names[:1])
        
        # Debug Info
        st.write(f"Loaded {len(strategies)} strategies.")
        if st.checkbox("Show Debug Info"):
            st.write("Strategies:", strategies)

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        progress = st.progress(0)
        all_hits = []
        
        for i, sym in enumerate(symbols):
            if i % 10 == 0: progress.progress(i / len(symbols))
            df = DataCache.get_cached_data(sym)
            if df is None or len(df) < 200: continue 
            
            # try:
            df = _compute_indicators(df)
            if global_data["VIX"] is not None:
                df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill").fillna(20.0)
            else: 
                df["vix"] = 20.0
                if i < 5: st.write(f"⚠️ No VIX data for {sym}, using default 20.0")

            last_idx = len(df) - 1
            for strat_name in selected_names:
                strat = GenericStrategy(strategies[strat_name])
                try:
                    signal = strat.entry(df, last_idx)
                    if signal:
                        row = df.iloc[-1]
                        target = row["close"] * 1.10
                        # Check for dynamic target
                        for rule in strategies[strat_name].get("exit_rules", []):
                            if rule.get("type") == "profit_target": target = row["close"] * rule["val"]
                        
                        all_hits.append({
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": f"${row['close']:.2f}",
                            "Stop": f"${signal['stop_price']:.2f}",
                            "Target": f"${target:.2f}",
                            "Risk": f"{(1 - signal['stop_price']/row['close'])*100:.1f}%"
                        })
                    # else:
                    #     if i < 3: st.write(f"No signal for {sym} with {strat_name}")
                except Exception as e:
                    st.error(f"Strategy Error {sym}: {e}")
            # except Exception as e: st.error(f"Error processing {sym}: {e}")

        progress.progress(100)
        if all_hits: st.dataframe(pd.DataFrame(all_hits))
        else: st.warning("No setups found.")

elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    
    # Initialize session state for timeframe
    if "bt_timeframe" not in st.session_state: st.session_state.bt_timeframe = "5 Years"

    col1, col2 = st.columns(2)
    with col1: bt_universe = st.selectbox("Universe", ["S&P 1500", "S&P 500", "S&P 100"], index=0)
    with col2:
        gen_strategies = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strategies = [s["name"] for s in json.load(f)]
        except: pass
        
        use_all_bt = st.checkbox("Select All Strategies (Backtest)", value=True)
        if use_all_bt:
            bt_strategies = gen_strategies
        else:
            bt_strategies = st.multiselect("Strategies", gen_strategies, default=gen_strategies[:3] if gen_strategies else None)

    st.write("Timeframe:")
    tf_cols = st.columns(6)
    timeframes = ["1 Year", "5 Years", "10 Years", "20 Years", "Max"]
    for i, tf in enumerate(timeframes):
        if tf_cols[i].button(tf, use_container_width=True, type="primary" if st.session_state.bt_timeframe == tf else "secondary"):
            st.session_state.bt_timeframe = tf
    
    timeframe = st.session_state.bt_timeframe

    if st.button("Run Backtest", type="primary"):
        days_map = {"1 Year": 365, "5 Years": 1260, "10 Years": 2520, "20 Years": 5040, "Max": 10000}
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_map.get(timeframe, 1260))).date()
        symbols = get_index_symbols(bt_universe)
        data_map = {}
        global_data = get_global_data(days=1260)
        
        with st.status("Running Backtest...") as status:
            for sym in symbols:
                df = DataCache.get_cached_data(sym)
                if df is not None and len(df) > 200: data_map[sym] = df
            
            results = run_compare(bt_strategies, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_data)
            status.update(label="Complete!", state="complete")
        
        if not results.empty:
            st.dataframe(results.style.format({
                "hit_rate": "{:.1f}%", "avg_profit_pct": "{:.2f}%", "cagr": "{:.1f}%", 
                "Score": "{:.1f}", "profit_factor": "{:.2f}", "payoff_ratio": "{:.2f}"
            }))
        else: st.error("No trades found.")