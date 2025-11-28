import streamlit as st
import pandas as pd
import sys
import os
import json
import math
from datetime import datetime, timedelta, timezone

# Ensure project root is in path
sys.path.append(os.path.dirname(__file__))

from data.schwab_client import sd
from data.indices import get_index_symbols
from data.loader import fetch_single_symbol, fetch_data_pack
from execution.engine import _compute_indicators, run_compare
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper", layout="wide", page_icon="🎯")
st.sidebar.title("🎯 Apex Sniper")

# --- API Status ---
with st.sidebar.expander("📡 API Status", expanded=False):
    if st.button("Test Connection"):
        res = sd.health_check("VOO")
        if res.get("ok"): st.success("Online")
        else: st.error("Offline")

mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])

@st.cache_data(ttl=3600)
def get_global_data(days=400):
    """Fetch SPY and VIX once for context."""
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}


def calculate_quality_score(row, strategy_name):
    """
    Quantifies the 'Strength' of a setup on a 0-100 scale.
    Prioritizes Relative Strength (Leaders) and Deep Pullbacks.
    """
    score = 50.0  # Base Score

    # 1. Relative Strength (The most important filter)
    # Reward stocks outperforming SPY (rs_trend > 0)
    rs_trend = row.get("rs_trend", 0)
    score += (rs_trend * 100.0)  # Boost score for leaders

    # 2. Market Cap / Volatility Weighting
    # Slight preference for smoother trends (ADX)
    adx = row.get("adx", 20)
    if adx > 25: score += 5
    if adx > 40: score += 5

    # 3. Strategy Specific Boosts
    if "Sniper" in strategy_name:
        # For Sniper: We want EXTREME fear. Higher VIX is better (handled in filter),
        # but locally, we want deep oversold.
        rsi2 = row.get("rsi2", 50)
        if rsi2 < 5: score += 20
        elif rsi2 < 10: score += 10

    elif "MachineGun" in strategy_name:
        # For Machine Gun: We want "Buy the Dip in a Leader"
        # Reward lower RSI2 (better entry price)
        rsi2 = row.get("rsi2", 50)
        score += (50 - rsi2) * 0.5  # Add points for being more oversold

        # Reward Volume Surges (Ignition)
        vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
        if vol_rel > 1.5: score += 10

    return max(0.0, min(100.0, score))


# --- 1. Live Screener ---
if mode == "Live Screener":
    st.header("🔭 Live Market Screener")

    col1, col2, col3 = st.columns(3)
    with col1:
        universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=1)
    with col2:
        # Load Strategies from JSON
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")

        all_names = list(strategies.keys())
        selected_names = st.multiselect("Strategies", all_names, default=all_names)

    with col3:
        # The Filter to solve the "500 results" problem
        top_n = st.slider("Show Top N Results", min_value=5, max_value=100, value=20)

    if st.button("Run Scan", type="primary"):
        symbols = get_index_symbols(universe)
        # Fetch global context (SPY/VIX) for Relative Strength math
        global_data = get_global_data(days=1260)
        spy_df = global_data.get("SPY")
        vix_df = global_data.get("VIX")

        progress = st.progress(0, text="Starting Scan...")
        status_text = st.empty()
        all_hits = []

        for i, sym in enumerate(symbols):
            if i % 10 == 0:
                progress.progress(i / len(symbols))
                status_text.text(f"Scanning {sym}...")

            # Fetch & Compute
            df = fetch_single_symbol(sym, days=400)  # Shorter history for speed
            if df is None or len(df) < 200: continue

            try:
                # Use Shared Engine Math (Includes RS_Trend, Beta, etc.)
                df = _compute_indicators(df, spy_df=spy_df)

                # Attach VIX
                if vix_df is not None:
                    # Quick reindex for just this symbol's dates
                    current_vix = vix_df["close"].reindex(df.index).ffill().fillna(20.0)
                    df["vix"] = current_vix
                else:
                    df["vix"] = 20.0

                # Check Last Candle (Today)
                # Note: We check 'iloc[-1]' for Live Signals.
                signal_idx = len(df) - 1
                price_row = df.iloc[signal_idx]

                for strat_name in selected_names:
                    strat = GenericStrategy(strategies[strat_name])

                    # Check Entry
                    signal = strat.entry(df, signal_idx)

                    if signal:
                        # Calculate Quality Score
                        quality = calculate_quality_score(price_row, strat_name)

                        # Calculate Targets based on Strategy Type
                        stop = signal.get("stop_price", price_row["close"] * 0.95)
                        risk_pct = (price_row["close"] - stop) / price_row["close"]
                        target = price_row["close"] * (1 + (risk_pct * 2.0))  # 2R Target Default

                        all_hits.append({
                            "Score": round(quality, 1),
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": price_row['close'],
                            "Stop": stop,
                            "Target": target,
                            "Risk %": round(risk_pct * 100, 2),
                            "RS_Trend": round(price_row.get('rs_trend', 0), 4),
                            "RSI2": round(price_row.get('rsi2', 50), 1)
                        })
            except Exception as e:
                # print(f"Error {sym}: {e}")
                pass

        progress.progress(100)
        status_text.text("Scan Complete!")

        if all_hits:
            # Sort by Quality Score (Highest First)
            df_results = pd.DataFrame(all_hits)
            df_results = df_results.sort_values(by="Score", ascending=False)

            # Filter Top N
            top_results = df_results.head(top_n)

            st.success(f"Found {len(all_hits)} total setups. Showing Top {len(top_results)}.")

            # Formatting for display
            st.dataframe(
                top_results.style.format({
                    "Price": "${:.2f}",
                    "Stop": "${:.2f}",
                    "Target": "${:.2f}",
                    "Score": "{:.1f}"
                }).background_gradient(subset=["Score"], cmap="Greens")
            )

            # Download Button
            csv = top_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Top Setups CSV", csv, "apex_top_setups.csv", "text/csv")

        else:
            st.warning("No setups found matching current criteria.")

# --- 2. Backtest ---
elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    col1, col2, col3 = st.columns(3)
    with col1: bt_universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=1)
    with col2:
        gen_strategies = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strategies = [s["name"] for s in json.load(f)]
        except: pass
        bt_strategies = st.multiselect("Strategies", gen_strategies, default=gen_strategies)
    with col3: timeframe = st.selectbox("Timeframe", ["1 Year", "5 Years"], index=1)

    if st.button("Run Backtest"):
        days_map = {"1 Year": 365, "5 Years": 1260}

        # Calculate Start Date for Filter
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_map.get(timeframe, 1260))).date()

        with st.status("Loading Data...") as status:
            symbols = get_index_symbols(bt_universe)
            data_map = fetch_data_pack(symbols, days=days_map.get(timeframe, 1260))

            # Fetch Globals
            g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days_map.get(timeframe, 1260))

            # FIXED: Handle DataFrame truth value ambiguity
            vix = g_data.get("$VIX")
            if vix is None:
                vix = g_data.get("VIX")

            global_context = {"SPY": g_data.get("SPY"), "VIX": vix}

            status.update(label=f"Backtesting {len(data_map)} symbols...", state="running")

            # Run using SHARED ENGINE
            results = run_compare(bt_strategies, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_context)
            status.update(label="Complete!", state="complete")

        if not results.empty:
            # Display metrics including the new Risk Metrics
            st.dataframe(results.style.format({
                "hit_rate": "{:.1f}%", "avg_profit_pct": "{:.2f}%", "cagr": "{:.1%}",
                "Score": "{:.1f}", "beta": "{:.2f}", "sortino": "{:.2f}",
                "exposure_pct": "{:.1f}%", "calmar": "{:.2f}"
            }))
        else:
            st.error("No trades generated.")
