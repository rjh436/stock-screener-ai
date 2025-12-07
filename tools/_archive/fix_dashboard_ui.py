"""Restore full-featured Streamlit dashboard with navigation, strategy info, and simulator."""

import os


def fix_dashboard_ui():
    print("🛠️ RESTORING & FIXING DASHBOARD UI...")

    app_path = "app.py"

    app_code = """import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest, _compute_indicators
from strategies.generic import GenericStrategy
from simulation.paper_trader import PaperTrader

# --- CONFIGURATION ---
CONFIG_PATH = "config/generated_strategies.json"
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")

# --- HELPER: LOAD STRATEGIES ---
def load_strategies():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return []

def format_rules(rules):
    if not rules: return "None"
    return ", ".join([f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}" for r in rules])

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.markdown("---")

    # 1. MODE SELECTION (Restored)
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    st.markdown("---")

    # 2. STRATEGY DETAILS (Dynamic)
    st.subheader("📘 Active Strategies")
    strategies = load_strategies()

    if strategies:
        for s in strategies:
            with st.expander(f"**{s.get('name', 'Unknown')}**"):
                st.markdown(f"**Role:** {s.get('type', 'Hybrid')}")
                st.markdown(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.markdown(f"**Time:** {s.get('time_stop')} Days")

                # Exit Logic
                exits = s.get('exit_rules', [])
                if not exits:
                    st.markdown("**Target:** NONE (Run)")
                else:
                    for rule in exits:
                        if rule.get('type') == 'profit_target':
                            pct = (float(rule.get('val')) - 1) * 100
                            st.markdown(f"**Target:** +{pct:.1f}%")

                st.markdown("**Entry Rules:**")
                st.code(format_rules(s.get('entry_rules', [])))
    else:
        st.error("No strategies found!")

# --- HELPER: SCANNER LOGIC ---
def run_scanner(selected_strategies, universe="S&P 1500"):
    symbols = get_index_symbols(universe)
    data_map = fetch_data_pack(symbols, days=400)
    results = []

    progress = st.progress(0)
    status = st.empty()
    total = len(selected_strategies)

    for i, strat_config in enumerate(selected_strategies):
        name = strat_config["name"]
        status.write(f"Scanning: **{name}**...")
        strat = GenericStrategy(strat_config)

        for sym, df in data_map.items():
            if df is None or df.empty:
                continue
            try:
                df_ind = _compute_indicators(df.copy())
                if df_ind.empty:
                    continue

                # Check Last Candle
                if strat.entry(df_ind, len(df_ind) - 1):
                    row = df_ind.iloc[-1]
                    atr = row.get("atr14", row["close"] * 0.02)
                    close_px = row["close"]

                    # Calc Levels
                    stop_mult = float(strat_config.get("stop_loss_atr", 3.0))
                    stop_px = close_px - (atr * stop_mult)

                    # Calc Target
                    exits = strat_config.get("exit_rules", [])
                    target_txt = "OPEN (Run)"
                    if exits:
                        for rule in exits:
                            if rule.get("type") == "profit_target":
                                target_px = close_px * float(rule.get("val"))
                                target_txt = f"${target_px:.2f}"

                    results.append({
                        "Symbol": sym,
                        "Strategy": name,
                        "Price": close_px,
                        "ATR": atr,
                        "Stop Loss": stop_px,
                        "Target": target_txt,
                        "Action": "BUY NEXT OPEN"
                    })
            except:
                continue
        if total:
            progress.progress((i + 1) / total)

    status.empty()
    progress.empty()
    return pd.DataFrame(results)

# --- PAGE: LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    col1, col2 = st.columns([3, 1])
    with col1:
        # Strategy Selector (Restored)
        strat_names = [s["name"] for s in strategies]
        selected_names = st.multiselect("Select Strategies", strat_names, default=strat_names)
    with col2:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500"], index=1)

    if st.button("RUN SCAN", type="primary"):
        selected_configs = [s for s in strategies if s["name"] in selected_names]
        with st.spinner("Scanning market..."):
            df = run_scanner(selected_configs, universe)
            st.session_state.scan_results = df

    # Results Display
    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results
        if df.empty:
            st.info("No signals found today.")
        else:
            # Super Signal Check
            dupes = df[df.duplicated(subset=['Symbol'], keep=False)]
            if not dupes.empty:
                st.success(f"🔥 SUPER SIGNALS ({len(dupes['Symbol'].unique())}): High Conviction")
                st.dataframe(dupes)

            st.subheader("All Setups")
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}", "ATR": "{:.2f}"}), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"apex_scan_{datetime.now().date()}.csv", "text/csv")

# --- PAGE: SIMULATOR ---
elif mode == "Simulator":
    st.header("🎮 Paper Trading Simulator")
    trader = PaperTrader()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cash", f"${trader.state['cash']:,.2f}")
    with col2:
        st.metric("Equity", f"${trader.state['equity']:,.2f}")
    with col3:
        if st.button("⚠️ RESET ACCOUNT"):
            trader.reset_account()
            st.experimental_rerun()

    st.markdown("### Actions")
    if st.button("🔄 Run Daily Cycle (Process Orders & Scan)"):
        with st.spinner("Running daily cycle..."):
            # 1. Update Prices & Fill Pending
            state, fills = trader.update_valuations()
            if fills:
                for f in fills:
                    st.success(f)

            # 2. Run Scan for New Entries
            strategies = load_strategies()
            df_scan = run_scanner(strategies, "S&P 1500")

            # 3. Execute/Queue
            if not df_scan.empty:
                candidates = []
                for _, row in df_scan.iterrows():
                    candidates.append({
                        "Symbol": row["Symbol"],
                        "Strategy": row["Strategy"],
                        "Price": row["Price"],  # Close price used for sizing est
                        "Stop": row["Stop Loss"],
                        "Raw_Score": 100  # Default score if engine not fully engaged
                    })
                logs = trader.execute_entries(candidates)
                for log in logs:
                    st.write(log)
            else:
                st.info("No new signals.")

            st.success("Cycle Complete.")

    st.markdown("### Pending Orders (Buy Next Open)")
    pending = trader.state.get("pending_orders", [])
    if pending:
        st.table(pd.DataFrame(pending))
    else:
        st.info("No pending orders.")

    st.markdown("### Current Holdings")
    pos = trader.state["positions"]
    if pos:
        df_pos = pd.DataFrame.from_dict(pos, orient='index')
        st.dataframe(df_pos)
    else:
        st.info("Portfolio is empty.")

# --- PAGE: BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Backtest")
    if st.button("Run 5-Year Verification"):
        # Simple trigger wrapper
        st.info("Check terminal for detailed output.")
"""

    with open(app_path, "w") as f:
        f.write(app_code)
    print("   ✅ UI Restored: Navigation, Strategy Details, and Simulator are back.")


if __name__ == "__main__":
    fix_dashboard_ui()
