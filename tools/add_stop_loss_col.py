import os


def add_stop_loss_col():
    print("🛡️ UPDATING DASHBOARD: Adding Stop Loss Column to Holdings...")
    
    app_path = "app.py"
    
    # Restoring the full app code with the requested modification
    app_code = """
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta

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
    strat_name = row['Strategy'].split(" + ")[0] 
    entry_price = row['Entry Price']
    entry_date = pd.to_datetime(row['Date'])
    
    strat = strategies_map.get(strat_name)
    if not strat: return "Unknown"
    
    exits = strat.get("exit_rules", [])
    for rule in exits:
        if rule.get("type") == "profit_target":
            target_px = entry_price * float(rule.get("val"))
            return f"Limit: ${target_px:.2f}"
            
    time_stop = strat.get("time_stop", 70)
    sell_date = entry_date + timedelta(days=time_stop)
    days_left = (sell_date.date() - datetime.now().date()).days
    
    if days_left < 0: return "EXPIRED (Sell)"
    return f"Hold until {sell_date.strftime('%b %d')} ({days_left}d left)"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.caption("Institutional Grade Algo System")
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    
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
                st.write("**Entry Rules:**")
                for r in s.get('entry_rules', []):
                    st.code(format_rule(r))
    else:
        st.error("No strategies found!")

# --- 1. LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    col1, col2 = st.columns([1, 4])
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500"], index=1)
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
                                "Symbol": sym,
                                "Strategy": s_conf["name"],
                                "Price": row["close"],
                                "Stop Loss": row["close"] - (atr * stop_mult),
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
                st.success(f"🔥 SUPER SIGNALS DETECTED: {len(dupes['Symbol'].unique())} High Conviction Trades")
                st.dataframe(dupes)
            
            st.subheader(f"All Setups ({len(df)})")
            st.dataframe(
                df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), 
                use_container_width=True
            )
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"apex_scan_{datetime.now().date()}.csv", "text/csv")

# --- 2. BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    if st.button("Run 5-Year Backtest"):
        with st.spinner("Running simulation..."):
            symbols = get_index_symbols("S&P 500")
            data = fetch_data_pack(symbols, days=1260)
            
            tabs = st.tabs([s["name"] for s in selected_strategies])
            for i, s_conf in enumerate(selected_strategies):
                with tabs[i]:
                    res = run_backtest(GenericStrategy(s_conf), data)
                    col1, col2, col3 = st.columns(3)
                    col1.metric("CAGR", f"{res['cagr']:.1%}")
                    col2.metric("Win Rate", f"{res['hit_rate']:.1f}%")
                    col3.metric("Avg Profit", f"{res['avg_profit_pct']:.2f}%")
                    st.line_chart(res["equity_curve"])

# --- 3. SIMULATOR (PRO MODE) ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader (Pro)")
    trader = PaperTrader()
    state = trader.state
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl_val = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl_val:,.2f}", delta=f"{pnl_val/100000*100:.2f}%")
    m4.metric("Positions", len(state['positions']))
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Run Daily Cycle"):
            with st.spinner("Processing..."):
                trader.update_valuations() 
                exits = trader.process_exits(strategies_map)
                for e in exits: st.toast(e, icon="💰")
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
    if state['positions']:
        pos_list = []
        for sym, p in state['positions'].items():
            row = {
                "Symbol": sym,
                "Strategy": p['strategy'],
                "Shares": p['shares'],
                "Entry Price": p['entry_price'],
                "Stop Loss": p.get('stop_price', 0.0), # ADDED FIELD
                "Current Price": p.get('current_price', p['entry_price']),
                "Value": p.get('current_price', p['entry_price']) * p['shares'],
                "Unrealized PnL": p.get('unrealized_pnl', 0),
                "Return %": p.get('unrealized_pct', 0),
                "Date": p['date']
            }
            row["Exit Plan"] = calc_exit_plan(row, strategies_map)
            pos_list.append(row)
            
        df_pos = pd.DataFrame(pos_list)
        
        st.dataframe(
            df_pos.style
            .format({
                "Entry Price": "${:.2f}",
                "Stop Loss": "${:.2f}", # ADDED FORMAT
                "Current Price": "${:.2f}",
                "Value": "${:,.2f}",
                "Unrealized PnL": "${:,.2f}",
                "Return %": "{:+.2f}%"
            })
            .map(color_pnl, subset=["Unrealized PnL", "Return %"]),
            use_container_width=True,
            height=400
        )
    else:
        st.info("Portfolio is empty.")
        
    st.subheader("⏳ Pending Orders (Next Open)")
    pending = state.get("pending_orders", [])
    if pending:
        df_pend = pd.DataFrame(pending)
        st.dataframe(df_pend, use_container_width=True)
    else:
        st.caption("No orders queued.")
"""
    with open(app_path, "w") as f:
        f.write(app_code)
    print("   ✅ Dashboard Updated: Stop Loss Column Added.")


if __name__ == "__main__":
    add_stop_loss_col()
