import streamlit as st
import pandas as pd
import sys, os, json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(__file__))
from data.indices import get_index_symbols
from data.loader import fetch_data_pack  # NEW LOADER
from execution.engine import run_compare, _compute_indicators
from strategies.generic import GenericStrategy

st.set_page_config(page_title="Apex Sniper", layout="wide")
st.sidebar.title("🎯 Apex Sniper")
mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest"])


@st.cache_data(ttl=3600)
def load_ui_data(universe, days=400):
    symbols = get_index_symbols(universe)
    data_map = fetch_data_pack(symbols, days=days)
    global_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = global_data.get("$VIX") or global_data.get("VIX")
    return symbols, data_map, {"SPY": global_data.get("SPY"), "VIX": vix}


if mode == "Live Screener":
    st.header("🚀 Live Market Screener")
    col1, col2 = st.columns(2)
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500", "S&P 100"])
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f):
                    strategies[s["name"]] = s
        except:
            st.error("No Strategies Found!")
        selected = st.multiselect("Strategies", list(strategies.keys()), default=list(strategies.keys())[:1])

    if st.button("Run Scan"):
        symbols, data_map, global_data = load_ui_data(universe, days=1260)
        progress = st.progress(0)
        hits = []

        for i, sym in enumerate(symbols):
            if i % 10 == 0:
                progress.progress(i / len(symbols))
            df = data_map.get(sym)
            if df is None:
                continue

            try:
                df = _compute_indicators(df)
                if global_data["VIX"] is not None:
                    df["vix"] = global_data["VIX"]["close"].reindex(df.index, method="ffill").fillna(20.0)
                else:
                    df["vix"] = 20.0

                last_idx = len(df) - 1
                for s_name in selected:
                    strat = GenericStrategy(strategies[s_name])
                    signal = strat.entry(df, last_idx)
                    if signal:
                        row = df.iloc[-1]
                        tgt = row["close"] * 1.10
                        for r in strategies[s_name].get("exit_rules", []):
                            if r.get("type") == "profit_target":
                                tgt = row["close"] * r["val"]

                        hits.append(
                            {
                                "Strategy": s_name,
                                "Symbol": sym,
                                "Price": f"${row['close']:.2f}",
                                "Stop": f"${signal['stop_price']:.2f}",
                                "Target": f"${tgt:.2f}",
                            }
                        )
            except Exception:
                continue

        progress.progress(100)
        if hits:
            st.dataframe(pd.DataFrame(hits))
        else:
            st.warning("No setups found.")

elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        bt_univ = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"])
    with col2:
        gen_strats = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strats = [s["name"] for s in json.load(f)]
        except:
            pass
        bt_strats = st.multiselect("Strategies", gen_strats, default=gen_strats[:3] if gen_strats else None)
    with col3:
        tf = st.selectbox("Timeframe", ["1 Year", "5 Years", "Max"])

    if st.button("Run Backtest"):
        d_map = {"1 Year": 365, "5 Years": 1260, "Max": 5000}
        days = d_map.get(tf, 1260)

        with st.status("Loading Data...") as status:
            symbols, data_map, global_data = load_ui_data(bt_univ, days=days)
            status.update(label=f"Backtesting {len(data_map)} symbols...", state="running")
            res = run_compare(bt_strats, data_map, symbols, start_cash=100000.0, start_date=None, global_data=global_data)
            status.update(label="Complete!", state="complete")

        if not res.empty:
            st.dataframe(
                res.style.format(
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
            st.error("No trades found.")
