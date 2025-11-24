import streamlit as st
import pandas as pd
import sys
import os
import json
import asyncio
from datetime import datetime, timedelta, timezone

# Path setup
sys.path.append(os.path.dirname(__file__))

from data.schwab_client import sd
from data.cache_manager import DataCache
from data.indices import get_index_symbols
from execution.engine import run_backtest, run_compare, _compute_indicators

# Page Config
st.set_page_config(page_title="Apex Sniper Screener", layout="wide", page_icon="🎯")

# --- Sidebar ---
st.sidebar.title("🎯 Apex Sniper")
mode = st.sidebar.radio("Mode", ["Live Screener", "Strategy Lab", "Portfolio Manager"])

# --- Helper: Robust Data Fetching ---
@st.cache_data(ttl=3600)
def get_global_data(days=400):
    """Fetch SPY and VIX for regime filters."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days + 50)
    
    g_data = {}
    for sym in ["SPY", "$VIX", "VIX"]:
        try:
            # Try Cache
            df = DataCache.get_cached_data(sym)
            if df is None:
                candles = sd.price_daily(sym, start_datetime=start, end_datetime=end)
                if candles:
                    df = pd.DataFrame(candles)
                    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                    df = df.set_index("datetime").sort_index()
                    DataCache.save_to_cache(sym, df)
            
            if df is not None and not df.empty:
                g_data[sym] = df
        except: pass
        
    # Normalize VIX key
    vix = g_data.get("$VIX") if "$VIX" in g_data else g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}

# --- 1. Live Screener ---
if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    
    universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500", "Nasdaq 100"])
    
    # Load Strategies
    strategies = {}
    try:
        with open("config/generated_strategies.json", "r") as f:
            gen_strats = json.load(f)
            for s in gen_strats: strategies[s["name"]] = s
    except: st.error("No Generated Strategies Found!")

    selected_strat_name = st.selectbox("Strategy", list(strategies.keys()))
    
    if st.button("Run Scan"):
        st.info(f"Scanning {universe} with {selected_strat_name}...")
        
        # 1. Get Symbols
        symbols = get_index_symbols(universe)
        st.write(f"Loaded {len(symbols)} symbols.")
        
        # 2. Get Global Data
        global_data = get_global_data(days=1260)
        if global_data["VIX"] is None: st.warning("VIX Data Missing! Volatility filters may fail.")
        
        # 3. Scan Loop (Progress Bar)
        progress = st.progress(0)
        hits = []
        
        # Load Strategy Object
        from strategies.generic import GenericStrategy
        strat_config = strategies[selected_strat_name]
        strategy_obj = GenericStrategy(strat_config)

        for i, sym in enumerate(symbols):
            # Update Progress
            if i % 10 == 0: progress.progress(i / len(symbols))
            
            # Fetch Data (Cache first)
            df = DataCache.get_cached_data(sym)
            if df is None:
                # If missing, skip to save time in UI (Optimizer does the heavy lifting)
                # Or fetch strictly recent data
                try:
                    candles = sd.price_daily(sym, start_datetime=datetime.now(timezone.utc)-timedelta(days=400), end_datetime=datetime.now(timezone.utc))
                    if candles:
                        df = pd.DataFrame(candles)
                        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
                        df = df.set_index("datetime").sort_index()
                        DataCache.save_to_cache(sym, df)
                except: pass
            
            if df is None or len(df) < 200: continue
            
            # Compute Indicators using CENTRAL ENGINE
            try:
                df = _compute_indicators(df)
                
                # Inject Global Context
                if global_data["SPY"] is not None:
                    spy_reindexed = global_data["SPY"]["close"].reindex(df.index, method="ffill")
                    df["rs_ratio"] = df["close"] / spy_reindexed
                    df["rs_sma20"] = df["rs_ratio"].rolling(20).mean()
                    df["rs_trend"] = (df["rs_ratio"] > df["rs_sma20"]).astype(int)
                else: df["rs_trend"] = 0
                
                if global_data["VIX"] is not None:
                    df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill")
                else: df["vix"] = 20.0

                # Check Signal (Last Bar)
                last_idx = len(df) - 1
                signal = strategy_obj.entry(df, last_idx)
                
                if signal:
                    row = df.iloc[-1]
                    hits.append({
                        "Symbol": sym,
                        "Price": row["close"],
                        "Stop Loss": signal["stop_price"],
                        "Volatility (ATR)": row["atr14"],
                        "RSI(2)": row["rsi2"],
                        "VIX": row["vix"]
                    })
            except Exception as e:
                # Silent fail for individual symbols
                pass

        progress.progress(100)
        
        if hits:
            st.success(f"Found {len(hits)} Setups!")
            st.dataframe(pd.DataFrame(hits))
        else:
            st.warning("No setups found matching criteria today.")

# --- 2. Strategy Lab ---
elif mode == "Strategy Lab":
    st.header("🧪 Strategy Lab")
    
    # Load Strategies
    gen_strategies = []
    try:
        with open("config/generated_strategies.json", "r") as f:
            gen_strategies = [s["name"] for s in json.load(f)]
    except: pass
    
    all_strats = gen_strategies
    selected = st.multiselect("Select Strategies to Backtest", all_strats, default=all_strats[:3] if all_strats else None)
    
    if st.button("Run Backtest (S&P 100 Sample)"):
        symbols = get_index_symbols("S&P 100")
        
        # Fetch Data Helper
        data_map = {}
        global_data = get_global_data(days=1260)
        
        with st.spinner("Fetching Data..."):
            for sym in symbols:
                df = DataCache.get_cached_data(sym)
                if df is not None and len(df) > 200:
                    data_map[sym] = df
        
        st.write(f"Running Backtest on {len(data_map)} symbols...")
        
        # RUN ENGINE
        results_df = run_compare(selected, data_map, symbols, start_cash=100000.0, global_data=global_data)
        
        if not results_df.empty and "Score" in results_df.columns:
            st.dataframe(results_df.style.format({
                "hit_rate": "{:.1f}%",
                "avg_profit_pct": "{:.2f}%",
                "cagr": "{:.1f}%",
                "Score": "{:.1f}",
                "profit_factor": "{:.2f}",
                "payoff_ratio": "{:.2f}",
                "avg_days_held": "{:.1f}d"
            }))
        elif not results_df.empty:
            st.warning("Results generated but 'Score' column missing.")
            st.dataframe(results_df)
        else:
            st.error("Backtest returned no results.")