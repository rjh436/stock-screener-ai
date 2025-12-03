"""Fix Backtest tab persistence so results survive button re-render."""

import os


def fix_app_persistence():
    print("💾 UPGRADING APP PERSISTENCE (Fixing Download Reset)...")
    
    app_path = "app.py"
    
    app_code = """
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime
import plotly.express as px

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import run_backtest, _compute_indicators
from strategies.generic import GenericStrategy
from simulation.paper_trader import PaperTrader

CONFIG_PATH = "config/generated_strategies.json"
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")

# --- HELPERS ---
def load_strategies():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return []

def format_rule(r):
    return f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}"

def calc_cagr(start_val, end_val, years):
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1.0 / years)) - 1.0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    
    st.markdown("### 📘 Active Strategies")
    strategies = load_strategies()
    selected_strategies = []
    
    if strategies:
        for s in strategies:
            # Strategy Selector
            use = st.checkbox(s.get('name'), value=True)
            if use: selected_strategies.append(s)
            
            with st.expander("Details"):
                st.write(f"**Role:** {s.get('type')}")
                st.write(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.write(f"**Time:** {s.get('time_stop')} Days")
                
                # Fix Exit Rules Display
                exits = s.get('exit_rules', [])
                target_display = "NONE (Run)"
                if exits:
                    for r in exits:
                        if r.get('type') == 'profit_target':
                            target_display = f"{r.get('val')}x"
                st.write(f"**Target:** {target_display}")
                
                st.write("**Entry Rules:**")
                for r in s.get('entry_rules', []):
                    st.code(format_rule(r))
    else:
        st.error("No strategies found!")

# --- LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500"], index=1)
        run_btn = st.button("RUN SCAN", type="primary")
        
    if run_btn:
        with st.spinner("Scanning market..."):
            symbols = get_index_symbols(universe)
            data = fetch_data_pack(symbols, days=400)
            results = []
            
            for s_conf in selected_strategies:
                strat = GenericStrategy(s_conf)
                for sym, df in data.items():
                    if df is None or df.empty: continue
                    try:
                        df_ind = _compute_indicators(df.copy())
                        if df_ind.empty: continue
                        # Check signal
                        if strat.entry(df_ind, len(df_ind)-1):
                            row = df_ind.iloc[-1]
                            atr = row.get("atr14", row["close"]*0.02)
                            stop_mult = float(s_conf.get("stop_loss_atr", 3.0))
                            
                            # Calc Target
                            exits = s_conf.get("exit_rules", [])
                            target_txt = "OPEN (Run)"
                            if exits:
                                for rule in exits:
                                    if rule.get("type") == "profit_target":
                                        t_price = row["close"] * float(rule.get("val"))
                                        target_txt = f"${t_price:.2f}"

                            results.append({
                                "Symbol": sym,
                                "Strategy": s_conf["name"],
                                "Price": row["close"],
                                "Stop Loss": row["close"] - (atr * stop_mult),
                                "Target": target_txt
                            })
                    except: continue
            
            df_res = pd.DataFrame(results)
            st.session_state.scan_results = df_res

    if "scan_results" in st.session_state and st.session_state.scan_results is not None:
        df = st.session_state.scan_results
        if df.empty:
            st.info("No signals found today.")
        else:
            # Super Signal Logic
            dupes = df[df.duplicated(subset=['Symbol'], keep=False)]
            if not dupes.empty:
                st.success(f"🔥 SUPER SIGNALS: {len(dupes['Symbol'].unique())} High Conviction Trades")
                st.dataframe(dupes)
            
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "apex_scan.csv", "text/csv")

# --- BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
    # Initialize persistence
    if "backtest_data" not in st.session_state:
        st.session_state.backtest_data = {}

    col1, col2 = st.columns([1, 4])
    with col1:
        duration = st.selectbox("Backtest Duration", ["Max History (Full)", "5 Years", "2 Years"])
    with col2:
        if st.button("Run Backtest", type="primary"):
            with st.spinner("Simulating strategies..."):
                symbols = get_index_symbols("S&P 500")
                days_map = {"Max History (Full)": 5000, "5 Years": 1260, "2 Years": 504}
                data = fetch_data_pack(symbols, days=days_map.get(duration, 5000))
                
                # Run and Store results in Session State
                results_map = {}
                for s_conf in selected_strategies:
                    res = run_backtest(GenericStrategy(s_conf), data)
                    results_map[s_conf["name"]] = res
                
                st.session_state.backtest_data = results_map

    # Render from Session State (Persistent)
    if st.session_state.backtest_data:
        tabs = st.tabs(list(st.session_state.backtest_data.keys()))
        
        for i, (strat_name, res) in enumerate(st.session_state.backtest_data.items()):
            with tabs[i]:
                # 1. Metrics
                eq_curve = res["equity_curve"]
                if not eq_curve.empty:
                    start_val = eq_curve.iloc[0]["Equity"]
                    curr_val = eq_curve.iloc[-1]["Equity"]
                    total_ret = ((curr_val/start_val)-1)*100
                else:
                    total_ret = 0.0

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{total_ret:.1f}%")
                col2.metric("CAGR", f"{res['cagr']*100:.1f}%")
                col3.metric("Avg Profit", f"{res['avg_profit_pct']:.2f}%")
                col4.metric("Trades", res['total_trades'])
                
                # 2. Multi-Period CAGR Table
                periods = [1, 3, 5, 10]
                cagr_data = []
                if not eq_curve.empty:
                    today = eq_curve.index[-1]
                    for y in periods:
                        target_date = today - pd.DateOffset(years=y)
                        # Find closest date
                        idx_loc = eq_curve.index.get_indexer([target_date], method='nearest')[0]
                        if idx_loc >= 0 and idx_loc < len(eq_curve):
                            past_date = eq_curve.index[idx_loc]
                            # Only calc if date is reasonably close (within 1 month)
                            if (target_date - past_date).days < 30:
                                past_val = eq_curve.iloc[idx_loc]["Equity"]
                                cagr = calc_cagr(past_val, curr_val, y)
                                cagr_data.append({"Period": f"{y} Year", "CAGR": f"{cagr*100:.1f}%"})
                            else:
                                cagr_data.append({"Period": f"{y} Year", "CAGR": "N/A"})
                
                if cagr_data:
                    st.table(pd.DataFrame(cagr_data))
                
                # 3. Chart
                if not eq_curve.empty:
                    st.line_chart(eq_curve["Equity"])
                
                # 4. Downloads
                trades_df = pd.DataFrame(res["trades_list"])
                if not trades_df.empty:
                    csv_t = trades_df.to_csv(index=False).encode('utf-8')
                    # Unique key for each button to prevent conflicts
                    st.download_button(
                        label=f"📥 Download Trade Log ({strat_name})", 
                        data=csv_t, 
                        file_name=f"trades_{strat_name}.csv", 
                        mime="text/csv",
                        key=f"dl_{strat_name}"
                    )

# --- SIMULATOR ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader")
    trader = PaperTrader()
    
    col1, col2 = st.columns(2)
    col1.metric("Cash", f"${trader.state['cash']:,.2f}")
    col2.metric("Equity", f"${trader.state['equity']:,.2f}")
    
    if st.button("🔄 Run Daily Cycle"):
        with st.spinner("Processing..."):
            state, fills = trader.update_valuations()
            for f in fills: st.success(f)
            
            # Reuse scanner logic for signal generation
            # (Simplified for brevity, real app duplicates logic or calls shared func)
            # For now we assume manual scan or consistent logic
            st.info("Valuations updated. Use Live Screener to find new setups.")
            
    st.subheader("Pending Orders (Next Open)")
    pending = trader.state.get("pending_orders", [])
    if pending:
        st.table(pd.DataFrame(pending))
    else:
        st.info("No pending orders.")

    st.subheader("Holdings")
    pos = trader.state["positions"]
    if pos:
        df_pos = pd.DataFrame.from_dict(pos, orient='index')
        st.dataframe(df_pos)
    else:
        st.info("Portfolio is empty.")
"""

    with open(app_path, "w") as f:
        f.write(app_code)
    print("   ✅ App Fixed: Persistent Backtest Results & Downloads Enabled.")


if __name__ == "__main__":
    fix_app_persistence()
