
import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta

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
                st.code(format_rule(s.get('entry_rules', [])[0]))
    else:
        st.error("No strategies found!")

# --- 1. LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    if "scan_results" not in st.session_state: st.session_state.scan_results = None

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
                            results.append({
                                "Symbol": sym, "Strategy": s_conf["name"],
                                "Price": row["close"], "Stop Loss": row["close"] - (atr * stop_mult),
                                "Target": "See Details"
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
                st.success(f"🔥 SUPER SIGNALS DETECTED: {len(dupes['Symbol'].unique())}")
                st.dataframe(dupes)
            st.dataframe(df.style.format({"Price": "${:.2f}", "Stop Loss": "${:.2f}"}), use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"apex_scan.csv", "text/csv")

# --- 2. BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    if st.button("Run 5-Year Backtest"):
        with st.spinner("Simulating strategies..."):
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
    
    # Stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl_val = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl_val:,.2f}", delta=f"{pnl_val/100000*100:.2f}%")
    m4.metric("Positions", len(state['positions']))
    
    # Actions
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Run Daily Cycle"):
            with st.spinner("Processing..."):
                trader.update_valuations() 
                exits = trader.process_exits(strategies_map)
                for e in exits: st.toast(e, icon="💰")
                # Scan logic would go here (abbreviated for brevity)
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

    # --- ACTIVE HOLDINGS (ACTIONABLE) ---
    st.subheader("📂 Active Holdings")
    if state['positions']:
        # Create Header
        h1, h2, h3, h4, h5, h6 = st.columns([1, 1, 2, 2, 2, 1])
        h1.markdown("**Symbol**")
        h2.markdown("**Shares**")
        h3.markdown("**Entry / Current**")
        h4.markdown("**PnL**")
        h5.markdown("**Exit Plan**")
        h6.markdown("**Action**")
        st.markdown("---")

        for sym, p in state['positions'].items():
            r1, r2, r3, r4, r5, r6 = st.columns([1, 1, 2, 2, 2, 1])
            
            # Data prep
            entry = p['entry_price']
            curr = p.get('current_price', entry)
            pnl = p.get('unrealized_pnl', 0)
            pct = p.get('unrealized_pct', 0)
            color = "green" if pnl >= 0 else "red"
            stop = p.get('stop_price', 0.0)
            
            # Render Row
            r1.write(f"**{sym}**")
            r2.write(f"{p['shares']}")
            r3.write(f"${entry:.2f} → ${curr:.2f}")
            r4.markdown(f":{color}[${pnl:,.2f} ({pct:+.2f}%)]")
            
            # Exit Plan Logic
            strat_name = p['strategy'].split(" + ")[0]
            plan = calc_exit_plan({"Strategy": strat_name, "Entry Price": entry, "Date": p['date']}, strategies_map)
            stop_txt = f"Stop: ${stop:.2f}"
            r5.write(f"{plan} | {stop_txt}")
            
            # SELL BUTTON
            if r6.button("SELL", key=f"sell_{sym}"):
                success, msg = trader.close_position(sym)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("Portfolio is empty.")
        
    # --- TRANSACTION HISTORY ---
    st.markdown("---")
    with st.expander("📜 Transaction History & Performance", expanded=True):
        history = state.get("history", [])
        if history:
            df_hist = pd.DataFrame(history)
            # Calculate Metrics
            total_trades = len(df_hist)
            win_rate = len(df_hist[df_hist['pnl'] > 0]) / total_trades * 100 if total_trades > 0 else 0
            total_realized = df_hist['pnl'].sum()
            
            k1, k2, k3 = st.columns(3)
            k1.metric("Realized PnL", f"${total_realized:,.2f}")
            k2.metric("Trades Closed", total_trades)
            k3.metric("Win Rate", f"{win_rate:.1f}%")
            
            # Display Table
            st.dataframe(
                df_hist.sort_values("exit_date", ascending=False).style
                .format({"entry_price": "${:.2f}", "exit_price": "${:.2f}", "pnl": "${:.2f}", "return_pct": "{:+.2f}%"})
                .map(color_pnl, subset=["pnl", "return_pct"]),
                use_container_width=True
            )
        else:
            st.caption("No closed trades yet.")
            
    st.subheader("⏳ Pending Orders")
    pending = state.get("pending_orders", [])
    if pending:
        st.dataframe(pd.DataFrame(pending))
    else:
        st.caption("No orders queued.")
