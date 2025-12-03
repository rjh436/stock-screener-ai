
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta
import plotly.express as px

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
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

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}'

def calc_exit_plan(row, strategies_map):
    strat_name = row.get('Strategy', '').split(" + ")[0] 
    entry_price = row.get('Entry Price', 0)
    entry_date = pd.to_datetime(row['Date'])
    
    strat = strategies_map.get(strat_name)
    if not strat: return "Unknown"
    
    exits = strat.get("exit_rules", [])
    for rule in exits:
        if rule.get("type") == "profit_target":
            target_px = entry_price * float(rule.get("val"))
            return f"Limit: ${target_px:.2f}"
            
    time_stop = strat.get("time_stop", 70)
    try:
        sell_date = entry_date + timedelta(days=time_stop)
        days_left = (sell_date.date() - datetime.now().date()).days
        if days_left < 0: return "Time Limit (Sell)"
        return f"Hold ({days_left}d left)"
    except:
        return f"Hold {time_stop}d"

def calc_cagr(start_val, end_val, years):
    if years <= 0: return 0.0
    return ((end_val / start_val) ** (1.0 / years)) - 1.0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Super Signal Lab", "Simulator"])
    
    st.markdown("### 📘 Active Strategies")
    strategies_list = load_strategies()
    strategies_map = {s['name']: s for s in strategies_list}
    
    selected_strategies = []
    if strategies_list:
        for s in strategies_list:
            use = st.checkbox(s.get('name'), value=True)
            if use: selected_strategies.append(s)
            with st.expander(f"Details: {s.get('name')[:15]}..."):
                st.write(f"**Role:** {s.get('type')}")
                st.write(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.write(f"**Time:** {s.get('time_stop')} Days")
                target = s.get('exit_rules')[0]['val'] if s.get('exit_rules') else 'NONE'
                st.write(f"**Target:** {target}")
                st.code(format_rule(s.get('entry_rules', [])[0]))
    else:
        st.error("No strategies found!")

# --- 1. LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    if "scan_results" not in st.session_state: st.session_state.scan_results = None

    col1, col2 = st.columns([1, 4])
    with col1:
        # Default to S&P 1500
        universe = st.selectbox("Universe", ["S&P 500", "S&P 100", "S&P 1500"], index=2)
        run_btn = st.button("RUN SCAN", type="primary")
        
    if run_btn:
        with st.spinner(f"Scanning {universe}..."):
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
                        if strat.entry(df_ind, len(df_ind)-1):
                            row = df_ind.iloc[-1]
                            atr = row.get("atr14", row["close"]*0.02)
                            stop_mult = float(s_conf.get("stop_loss_atr", 3.0))
                            exits = s_conf.get("exit_rules", [])
                            target_txt = "OPEN (Run)"
                            if exits and exits[0].get("type") == "profit_target":
                                t_price = row["close"] * float(exits[0].get("val"))
                                target_txt = f"${t_price:.2f}"

                            results.append({
                                "Symbol": sym, "Strategy": s_conf["name"],
                                "Price": row["close"], "Stop Loss": row["close"] - (atr * stop_mult),
                                "Target": target_txt
                            })
                    except: continue
            st.session_state.scan_results = pd.DataFrame(results)

    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results
        if df.empty:
            st.info("No signals found today.")
        else:
            dupes = df[df.duplicated(subset=['Symbol'], keep=False)]
            if not dupes.empty:
                st.success(f"🔥 SUPER SIGNALS: {len(dupes['Symbol'].unique())}")
                st.dataframe(dupes)
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"apex_scan_{datetime.now().date()}.csv", "text/csv")

# --- 2. BACKTEST (FIXED PERSISTENCE) ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
    # Initialize Session State
    if "backtest_results" not in st.session_state: st.session_state.backtest_results = {}
    if "bt_duration" not in st.session_state: st.session_state.bt_duration = "5 Years"
    
    # Settings
    col_uni, col_dur = st.columns([1, 3])
    with col_uni:
        bt_universe = st.selectbox("Universe", ["S&P 500", "S&P 100", "S&P 1500"], index=2)
    
    with col_dur:
        st.write("Duration:")
        c1, c2, c3, c4, c5 = st.columns(5)
        dur_map = {"1 Year": 252, "5 Years": 1260, "10 Years": 2520, "20 Years": 5040, "Max": 10000}
        
        for label in dur_map:
            if c1.button(label) if label=="1 Year" else c2.button(label) if label=="5 Years" else c3.button(label) if label=="10 Years" else c4.button(label) if label=="20 Years" else c5.button(label):
                st.session_state.bt_duration = label

    st.info(f"Settings: **{bt_universe}** for **{st.session_state.bt_duration}**")

    if st.button("🚀 RUN BACKTEST", type="primary"):
        with st.spinner("Simulating..."):
            days = dur_map.get(st.session_state.bt_duration, 1260)
            symbols = get_index_symbols(bt_universe)
            data = fetch_data_pack(symbols, days=days + 200)
            
            results_map = {}
            for s_conf in selected_strategies:
                res = run_backtest(GenericStrategy(s_conf), data)
                results_map[s_conf["name"]] = res
            
            st.session_state.backtest_results = results_map

    # DISPLAY RESULTS (Persistent)
    if st.session_state.backtest_results:
        tabs = st.tabs(list(st.session_state.backtest_results.keys()))
        for i, (name, res) in enumerate(st.session_state.backtest_results.items()):
            with tabs[i]:
                col1, col2, col3 = st.columns(3)
                col1.metric("CAGR", f"{res['cagr']:.1%}")
                col2.metric("Win Rate", f"{res['hit_rate']:.1f}%")
                col3.metric("Avg Profit", f"{res['avg_profit_pct']:.2f}%")
                
                st.line_chart(res["equity_curve"])
                
                # Multi-Period
                eq = res["equity_curve"]
                curr = eq.iloc[-1]["Equity"]
                periods = [1, 3, 5, 10]
                cagr_data = []
                today = eq.index[-1]
                for y in periods:
                    t_date = today - pd.DateOffset(years=y)
                    if not eq[eq.index <= t_date].empty:
                        past = eq[eq.index <= t_date].iloc[-1]["Equity"]
                        val = calc_cagr(past, curr, y)
                        cagr_data.append({"Period": f"{y}Y", "CAGR": f"{val:.1%}"})
                if cagr_data: st.table(pd.DataFrame(cagr_data))

                # Trade Log & Download
                trades_df = pd.DataFrame(res["trades_list"])
                if not trades_df.empty:
                        csv_t = trades_df.to_csv(index=False).encode('utf-8')
                        st.download_button(f"📥 Download Trades", csv_t, f"trades_{name}.csv", "text/csv", key=f"dl_{name}")

# --- 3. SUPER SIGNAL LAB ---
elif mode == "Super Signal Lab":
    st.header("🔥 Super Signal Backtest")
    st.markdown("""
    **Theory:** Trades that trigger BOTH *Gen 12* (Wealth) and *Gen 9* (Income).
    **Execution:** Managed as *Gen 12* (Let it Run).
    """)
    
    if "super_signal_res" not in st.session_state: st.session_state.super_signal_res = None

    if st.button("🧪 Test Super Signals (S&P 1500)"):
        with st.spinner("Calculating Intersection..."):
            super_conf = {
                "name": "SUPER_SIGNAL_V1", "type": "hybrid",
                "entry_rules": [{"col": "cci", "op": "<", "val": 0}, {"col": "bb_width", "op": ">", "val": 0.17}],
                "exit_rules": [], "stop_loss_atr": 4.4, "time_stop": 71
            }
            symbols = get_index_symbols("S&P 1500")
            data = fetch_data_pack(symbols, days=1260 + 200)
            st.session_state.super_signal_res = run_backtest(GenericStrategy(super_conf), data)
            
    if st.session_state.super_signal_res:
        res = st.session_state.super_signal_res
        st.success(f"CAGR: {res['cagr']:.1%} | Win Rate: {res['hit_rate']:.1f}% | Avg Profit: {res['avg_profit_pct']:.1f}%")
        st.line_chart(res["equity_curve"])
        
        trades_df = pd.DataFrame(res["trades_list"])
        if not trades_df.empty:
            st.dataframe(trades_df)
            csv = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Super Signal Trades", csv, "super_signals.csv", "text/csv")

# --- 4. SIMULATOR (PRO MODE) ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader (Pro)")
    trader = PaperTrader()
    state = trader.state
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl_val = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl_val:,.2f}", delta=f"{pnl_val/100000*100:.2f}%")
    m4.metric("Positions", len(state['positions']))
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Run Daily Cycle", type="primary"):
            with st.spinner("Processing..."):
                trader.update_valuations() 
                trader.process_exits(strategies_map)
            st.success("Cycle Complete.")
            st.rerun()
    with c2:
        if st.button("📡 Refresh Prices"):
            with st.spinner("Fetching quotes..."):
                trader.update_valuations() 
            st.success("Prices Updated.")
            st.rerun()
    with c3:
        if st.button("⚠️ Reset Account"):
            trader.reset_account()
            st.rerun()

    st.subheader("📂 Active Holdings")
    col_widths = [1.2, 1.0, 1.2, 1.2, 1.2, 1.5, 2.5, 1.2]
    h_cols = st.columns(col_widths)
    headers = ["Symbol", "Qty", "Entry", "Current", "Stop Loss", "PnL", "Exit Plan", "Action"]
    for col, h in zip(h_cols, headers):
        col.markdown(f"**{h}**")
    st.markdown("---")

    if state['positions']:
        for sym, p in state['positions'].items():
            c_cols = st.columns(col_widths)
            entry = p['entry_price']
            curr = p.get('current_price', entry)
            stop = p.get('stop_price', 0.0)
            shares = p['shares']
            pnl_val = p.get('unrealized_pnl', 0)
            pnl_pct = p.get('unrealized_pct', 0)
            plan = calc_exit_plan({"Strategy": p['strategy'], "Entry Price": entry, "Date": p['date']}, strategies_map)
            
            c_cols[0].write(f"**{sym}**")
            c_cols[1].write(f"{shares}")
            c_cols[2].write(f"${entry:.2f}")
            c_cols[3].write(f"${curr:.2f}")
            c_cols[4].markdown(f":red[${stop:.2f}]") 
            color = "green" if pnl_val >= 0 else "red"
            c_cols[5].markdown(f":{color}[${pnl_val:,.2f} ({pnl_pct:+.2f}%)]")
            c_cols[6].caption(plan)
            
            if c_cols[7].button("SELL", key=f"sell_{sym}", use_container_width=True):
                success, msg = trader.close_position(sym, reason="Manual")
                if success:
                    st.toast(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(msg)
            st.markdown("<hr style='margin: 5px 0'>", unsafe_allow_html=True)
    else:
        st.info("Portfolio is empty.")
        
    st.subheader("⏳ Pending Orders")
    if state.get("pending_orders"):
        st.dataframe(pd.DataFrame(state["pending_orders"]))
    else:
        st.caption("No orders queued.")
