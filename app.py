import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from data.cache_manager import DataCache
from data.indices import get_index_symbols
from data.schwab_client import sd
from execution.engine import _compute_indicators, run_compare
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper Screener", layout="wide", page_icon="🎯")
st.sidebar.title("🎯 Apex Sniper")
mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])


@st.cache_data(ttl=3600)
def get_global_data(days: int = 1260):
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
            if df is not None and not df.empty:
                g_data[sym] = df
        except Exception:
            continue
    vix = g_data.get("$VIX") if "$VIX" in g_data else g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}


def load_strategies():
    try:
        with open("config/generated_strategies.json", "r") as f:
            data = json.load(f)
        return {s["name"]: s for s in data}
    except Exception:
        return {}


if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox("Universe", ["S&P 1500", "S&P 500", "S&P 100"], index=0)
    with col2:
        strategies = load_strategies()
        all_names = list(strategies.keys())
        if not all_names:
            st.error("No strategies found. Please generate strategies.")
        use_all = st.checkbox("Select All Strategies", value=True)
        selected_names = all_names if use_all else st.multiselect("Strategies", all_names, default=all_names[:1])
        st.caption(f"Loaded {len(all_names)} strategies.")

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        all_hits = []
        progress = st.progress(0)

        for i, sym in enumerate(symbols):
            if i % 10 == 0:
                progress.progress(i / len(symbols))
            df = DataCache.get_cached_data(sym)
            if df is None or len(df) < 200:
                continue

            df = _compute_indicators(df)
            if global_data["VIX"] is not None:
                df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill").fillna(20.0)
            else:
                df["vix"] = 20.0

            last_idx = len(df) - 1
            for strat_name in selected_names:
                genome = strategies.get(strat_name)
                if not genome:
                    continue
                strat = GenericStrategy(genome)
                try:
                    signal = strat.entry(df, last_idx)
                    if not signal:
                        continue
                    row = df.iloc[last_idx]
                    target = row["close"] * 1.10
                    for rule in genome.get("exit_rules", []):
                        if rule.get("type") == "profit_target":
                            target = row["close"] * rule.get("val", 1.10)
                    rr_denom = row["close"] - signal["stop_price"]
                    rr = (target - row["close"]) / rr_denom if rr_denom > 0 else 0.0
                    all_hits.append(
                        {
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": f"${row['close']:.2f}",
                            "Stop": f"${signal['stop_price']:.2f}",
                            "Target": f"${target:.2f}",
                            "Risk": f"{(1 - signal['stop_price'] / row['close']) * 100:.1f}%",
                            "Risk/Reward": f"{rr:.2f}x",
                        }
                    )
                except Exception as e:
                    st.error(f"Strategy Error {sym}: {e}")

        progress.progress(1.0)
        if all_hits:
            st.dataframe(pd.DataFrame(all_hits))
        else:
            st.warning("No setups found.")

elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    col1, col2 = st.columns(2)
    with col1:
        bt_universe = st.selectbox("Universe", ["S&P 1500", "S&P 500", "S&P 100"], index=0)
    with col2:
        strategies = load_strategies()
        all_names = list(strategies.keys())
        bt_selection = st.multiselect("Strategies", all_names, default=all_names, help="Defaults to all Sniper strategies.")

    timeframe = st.selectbox("Timeframe", ["5 Years", "1 Year", "10 Years"], index=0)

    if st.button("Run Backtest", type="primary"):
        days_map = {"1 Year": 365, "5 Years": 1260, "10 Years": 2520}
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_map.get(timeframe, 1260))).date()
        symbols = get_index_symbols(bt_universe)
        data_map = {}
        global_data = get_global_data(days=1260)

        with st.status("Running Backtest..."):
            for sym in symbols:
                df = DataCache.get_cached_data(sym)
                if df is not None and len(df) > 200:
                    data_map[sym] = df

            results = run_compare(bt_selection, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_data)

        if not results.empty:
            st.dataframe(
                results.style.format(
                    {
                        "hit_rate": "{:.1f}%",
                        "avg_profit_pct": "{:.2f}%",
                        "cagr": "{:.1f}%",
                        "Score": "{:.1f}",
                        "profit_factor": "{:.2f}",
                        "payoff_ratio": "{:.2f}",
                    }
                )
            )
        else:
            st.error("No trades found. Check data availability and strategy rules.")
