
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

def run_scanner(strategies, universe):
    symbols = get_index_symbols(universe)
    data = fetch_data_pack(symbols, days=400)
    results = []
    for s_conf in strategies:
        strat = GenericStrategy(s_conf)
        for sym, df in data.items():
            if df is None or df.empty:
                continue
            try:
                df_ind = _compute_indicators(df.copy())
                if df_ind.empty:
                    continue
                if strat.entry(df_ind, len(df_ind) - 1):
                    row = df_ind.iloc[-1]
                    atr = row.get("atr14", row["close"] * 0.02)
                    stop_mult = float(s_conf.get("stop_loss_atr", 3.0))
                    exits = s_conf.get("exit_rules", [])
                    target_txt = "OPEN (Run)"
                    if exits and exits[0].get("type") == "profit_target":
                        t_price = row["close"] * float(exits[0].get("val"))
                        target_txt = f"${t_price:.2f}"
                    results.append({
                        "Symbol": sym,
                        "Strategy": s_conf["name"],
                        "Price": row["close"],
                        "Stop Loss": row["close"] - (atr * stop_mult),
                        "Target": target_txt
                    })
            except:
                continue
    return pd.DataFrame(results)

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
            use = st.checkbox(s.get('name'), value=True)
            if use: selected_strategies.append(s)
            
            with st.expander("Details"):
                st.write(f"**Role:** {s.get('type')}")
                st.write(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.write(f"**Time:** {s.get('time_stop')} Days")
                target_txt = s.get('exit_rules')[0]['val'] if s.get('exit_rules') else 'NONE (Run)'
                st.write(f"**Target:** {target_txt}")
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
            df_res = run_scanner(selected_strategies, universe)
            st.session_state.scan_results = df_res

    if "scan_results" in st.session_state:
        df = st.session_state.scan_results
        if df is not None and not df.empty:
            dupes = df[df.duplicated(subset=['Symbol'], keep=False)]
            if not dupes.empty:
                st.success(f"🔥 SUPER SIGNALS: {len(dupes['Symbol'].unique())} High Conviction Trades")
                st.dataframe(dupes)
            
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "apex_scan.csv", "text/csv")
        elif df is not None:
            st.info("No signals found.")

# --- BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
    duration = st.selectbox("Backtest Duration", ["Max History (Full)", "5 Years", "2 Years"])
    if st.button("Run Backtest"):
        with st.spinner("Simulating strategies..."):
            symbols = get_index_symbols("S&P 500")
            data = fetch_data_pack(symbols, days=5000)
            
            tabs = st.tabs([s["name"] for s in selected_strategies])
            
            for i, s_conf in enumerate(selected_strategies):
                with tabs[i]:
                    res = run_backtest(GenericStrategy(s_conf), data)
                    
                    eq_curve = res["equity_curve"]
                    start_val = eq_curve.iloc[0]["Equity"]
                    curr_val = eq_curve.iloc[-1]["Equity"]
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Return", f"{((curr_val/start_val)-1)*100:.1f}%")
                    col2.metric("CAGR (Max)", f"{res['cagr']*100:.1f}%")
                    col3.metric("Trades", res['total_trades'])
                    
                    periods = [1, 3, 5, 10, 20]
                    cagr_data = []
                    today = eq_curve.index[-1]
                    
                    for y in periods:
                        target_date = today - pd.DateOffset(years=y)
                        if target_date in eq_curve.index:
                            past_val = eq_curve.loc[target_date]["Equity"]
                        elif not eq_curve[eq_curve.index <= target_date].empty:
                            past_val = eq_curve[eq_curve.index <= target_date].iloc[-1]["Equity"]
                        else:
                            past_val = None
                            
                        if past_val:
                            cagr = calc_cagr(past_val, curr_val, y)
                            cagr_data.append({"Period": f"{y} Year", "CAGR": f"{cagr*100:.1f}%"})
                        else:
                            cagr_data.append({"Period": f"{y} Year", "CAGR": "N/A"})
                            
                    st.table(pd.DataFrame(cagr_data))
                    
                    st.line_chart(eq_curve)
                    
                    trades_df = pd.DataFrame(res["trades_list"])
                    if not trades_df.empty:
                        csv_t = trades_df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"📥 Download Trades ({s_conf['name']})", csv_t, f"trades_{s_conf['name']}.csv", "text/csv")

# --- SIMULATOR ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader")
    trader = PaperTrader()
    
    col1, col2 = st.columns(2)
    col1.metric("Cash", f"${trader.state['cash']:,.2f}")
    col2.metric("Equity", f"${trader.state['equity']:,.2f}")
    
    if st.button("🔄 Run Daily Cycle"):
        state, fills = trader.update_valuations()
        for f in fills: st.success(f)
        
        strategies = load_strategies()
        df_scan = run_scanner(strategies, "S&P 1500")
        if not df_scan.empty:
            cands = df_scan.to_dict('records')
            clean_cands = [{"Symbol": c["Symbol"], "Strategy": c["Strategy"], "Stop": c["Stop Loss"], "Raw_Score": 100} for c in cands]
            logs = trader.execute_entries(clean_cands)
            for l in logs: st.write(l)
            
    st.subheader("Pending Orders (Next Open)")
    st.dataframe(pd.DataFrame(trader.state.get("pending_orders", [])))
    
    st.subheader("Holdings")
    st.dataframe(pd.DataFrame.from_dict(trader.state["positions"], orient='index'))
