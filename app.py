import streamlit as st
import pandas as pd
import concurrent.futures
import sys
import os
import json
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
# --- PHASE 3 UPGRADE: Import Scoring Engine ---
from execution.engine import run_backtest, _compute_indicators, calculate_backtest_quality_score, DEFAULT_SCORING_WEIGHTS
from simulation.paper_trader import PaperTrader
from strategies.strategy_loader import load_strategies

CONFIG_PATH = "config/generated_strategies.json"
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")

# --- HELPERS ---
def load_strategy_configs():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return []

def format_rule(r):
    return f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}"

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}'

def score_to_rating(score):
    """Convert numerical score to visual rating for UI"""
    if score >= 85: return "🔥 Excellent"
    elif score >= 75: return "⭐ Strong"
    elif score >= 60: return "✅ Good"
    elif score >= 50: return "⚠️ Fair"
    return "❌ Weak"

def calc_exit_plan(row, strategies_map):
    strat_name = row.get('Strategy', '').split(" + ")[0] 
    entry_price = row.get('Entry Price', 0)
    
    strat = strategies_map.get(strat_name)
    if not strat: return "Unknown"
    
    exits = strat.get("exit_rules", [])
    for rule in exits:
        if rule.get("type") == "profit_target":
            target_px = entry_price * float(rule.get("val"))
            return f"Target: ${target_px:.2f}"
            
    time_stop = strat.get("time_stop", 70)
    try:
        entry_date = pd.to_datetime(row['Date'])
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
    st.caption("Institutional Grade Algo System")
    st.markdown("---")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    
    st.markdown("### 📘 Active Strategies")
    strategies_list = load_strategy_configs()
    strategies_map = {s['name']: s for s in strategies_list}
    
    selected_strategies = []
    if strategies_list:
        for s in strategies_list:
            # --- VISIBILITY FIX: CHECKBOXES ---
            use = st.checkbox(s.get('name'), value=True, key=f"chk_{s.get('name')}")
            if use: selected_strategies.append(s)
            
            # --- DETAILS DROPDOWN ---
            with st.expander(f"🔎 View Rules: {s.get('name')[:12]}..."):
                st.write(f"**Role:** {s.get('type')}")
                st.write(f"**Stop:** {s.get('stop_loss_atr')} ATR")
                st.write(f"**Time:** {s.get('time_stop')} Days")
                
                # Exit Logic Display
                exits = s.get('exit_rules', [])
                if not exits:
                    st.write("**Target:** NONE (Run)")
                else:
                    for rule in exits:
                        if rule.get('type') == 'profit_target':
                            pct = (float(rule.get('val')) - 1) * 100
                            st.write(f"**Target:** +{pct:.1f}%")
                            
                st.write("**Entry Criteria:**")
                for r in s.get('entry_rules', []):
                    st.code(format_rule(r))
    else:
        st.error("⚠️ No strategies found in config file!")

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
            
            # ONLY RUN SELECTED STRATEGIES
            if not selected_strategies:
                st.warning("No strategies selected in sidebar!")
            else:
                strat_objects = load_strategies(selected_strategies)
                for strat in strat_objects:
                    s_conf = strat.params
                    for sym, df in data.items():
                        if df is None or df.empty: 
                            continue
                        try:
                            df_ind = _compute_indicators(df.copy())
                            if df_ind.empty or len(df_ind) < 2:
                                continue

                            # Check entry signal on YESTERDAY's bar (avoid lookahead)
                            if strat.entry(df_ind, len(df_ind) - 2):
                                row_signal = df_ind.iloc[-2]   # Yesterday's bar
                                row_current = df_ind.iloc[-1]  # Today's close (proxy for next open)

                                atr = row_signal.get("atr14", row_signal["close"]*0.02)
                                stop_mult = float(s_conf.get("stop_loss_atr", 3.0))

                                # --- PHASE 3: UNIFIED SCORING ENGINE (NO LOOKAHEAD) ---
                                raw_score = calculate_backtest_quality_score(row_signal, s_conf["name"], DEFAULT_SCORING_WEIGHTS)
                                score = raw_score * 1.3 if "wealth" in s_conf["name"].lower() else raw_score

                                exits = s_conf.get("exit_rules", [])
                                if exits and exits[0].get("type") == "profit_target":
                                    target_txt = f"${row_signal['close'] * float(exits[0].get('val')):.2f}"
                                else:
                                    target_txt = "OPEN (Run)"

                                estimated_entry = row_current["close"]

                                results.append({
                                    "Symbol": sym, 
                                    "Strategy": s_conf["name"],
                                    "Price": row_current["close"],  # Today's close (tomorrow's expected open)
                                    "Stop Loss": estimated_entry - (atr * stop_mult),
                                    "Target": target_txt,
                                    "Score": score,
                                    "Rating": score_to_rating(score)
                                })
                        except: 
                            continue
                
                # Sort by Score Descending (Best setups first)
                if results:
                    results.sort(key=lambda x: x["Score"], reverse=True)
                st.session_state.scan_results = pd.DataFrame(results)

    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results
        if df.empty:
            st.info("No signals found today.")
        else:
            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Signals", len(df))
            c2.metric("Best Score", f"{df['Score'].max():.1f}")
            c3.metric("Top Strategy", df.iloc[0]['Strategy'])
            
            st.dataframe(
                df.style.format({
                    "Price": "${:.2f}", 
                    "Stop Loss": "${:.2f}",
                    "Score": "{:.1f}"
                }), 
                use_container_width=True
            )
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, f"apex_scan_{datetime.now().date()}.csv", "text/csv")

# --- 2. BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    # Super Signal is injected at runtime (do not touch config)
    super_signal_conf = {
        "name": "Super Signal (Gen 12 Wealth)",
        "type": "hybrid",
        "entry_rules": [
            {"col": "cci", "op": "<", "val": 0},
            {"col": "bb_width", "op": ">", "val": 0.17}
        ],
        "exit_rules": [],
        "stop_loss_atr": 3.5,  # FIXED: Matches Apex Income Trend Settings
        "time_stop": 71
    }
    
    col_uni, col_dur = st.columns([1, 3])
    with col_uni:
        bt_universe = st.selectbox("Universe", ["S&P 500", "S&P 100", "S&P 1500"], index=2)
    
    with col_dur:
        st.write("Duration:")
        c1, c2, c3, c4, c5 = st.columns(5)
        dur_map = {"1 Year": 252, "5 Years": 1260, "10 Years": 2520, "20 Years": 5040, "Max": 10000}
        if "bt_duration" not in st.session_state: st.session_state.bt_duration = "5 Years"
        
        for label in dur_map:
            if c1.button(label) if label=="1 Year" else c2.button(label) if label=="5 Years" else c3.button(label) if label=="10 Years" else c4.button(label) if label=="20 Years" else c5.button(label):
                st.session_state.bt_duration = label

    st.info(f"Settings: **{bt_universe}** for **{st.session_state.bt_duration}**")

    if st.button("🚀 RUN BACKTEST", type="primary"):
        if not selected_strategies:
            st.error("Please select at least one strategy in the Sidebar.")
        else:
            with st.spinner("Simulating..."):
                days = dur_map.get(st.session_state.bt_duration, 1260)
                symbols = get_index_symbols(bt_universe)
                data = fetch_data_pack(symbols, days=days + 200)
                
                if "backtest_results" not in st.session_state: st.session_state.backtest_results = {}
                results_map = {}
                # Build run set without mutating sidebar selections
                run_set = list(selected_strategies)
                run_set.append(super_signal_conf)
                run_strategies = load_strategies(run_set)

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    future_map = {
                        executor.submit(run_backtest, strat, data): strat.name
                        for strat in run_strategies
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        name = future_map[future]
                        try:
                            res = future.result()
                            results_map[name] = res
                        except Exception as e:
                            st.error(f"Backtest failed for {name}: {e}")
                    
                st.session_state.backtest_results = results_map
            
    if "backtest_results" in st.session_state and st.session_state.backtest_results:
        preferred_order = [
            "Apex Wealth (Gen 12)",
            "Apex Income (Gen 9)",
            "Super Signal (Gen 12 Wealth)"
        ]
        ordered_names = [n for n in preferred_order if n in st.session_state.backtest_results]
        # add any additional strategies that were selected but not in preferred list
        ordered_names.extend([n for n in st.session_state.backtest_results.keys() if n not in ordered_names])

        tabs = st.tabs(ordered_names)
        for i, name in enumerate(ordered_names):
            res = st.session_state.backtest_results[name]
            with tabs[i]:
                col1, col2, col3 = st.columns(3)
                col1.metric("CAGR", f"{res['cagr']:.1%}")
                col2.metric("Win Rate", f"{res['hit_rate']:.1f}%")
                col3.metric("Avg Profit", f"{res['avg_profit_pct']:.2f}%")
                st.line_chart(res["equity_curve"])
                if res["trades_list"]:
                     csv_t = pd.DataFrame(res["trades_list"]).to_csv(index=False).encode('utf-8')
                     st.download_button(f"📥 Download Trades", csv_t, f"trades_{name}.csv", "text/csv", key=f"dl_{i}")

# --- 3. SIMULATOR (PRO MODE) ---
elif mode == "Simulator":
    st.header("🎮 Paper Trader (Pro)")
    pt = PaperTrader(configs=selected_strategies if selected_strategies else None)
    state = pt.state
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl_val = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl_val:,.2f}", delta=f"{pnl_val/100000*100:.2f}%")
    m4.metric("Positions", len(state['positions']))
    
    if st.button("Run Daily Scan", type="primary"):
        status = st.status("🚀 Initializing Simulation...", expanded=True)
        
        try:
            # Step 1: Health Check
            status.write("1️⃣ Verifying Strategies...")
            if not getattr(pt, 'strategies', None):
                status.update(label="❌ No Strategies Loaded!", state="error")
                st.error("PaperTrader has 0 loaded strategies. Check config.")
                st.stop()
            status.write(f"   - Active: {[s.name for s in pt.strategies]}")
            
            # Step 2: Execution
            status.write("2️⃣ Executing Scan & Governor...")
            symbols = get_index_symbols("S&P 1500")
            data_pack = fetch_data_pack(symbols, days=400)
            new_trades = pt.run_daily_scan(data_pack)
            
            # Step 3: Analysis
            status.write(f"3️⃣ Scan Complete. Orders Queued: {len(new_trades) if new_trades else 0}")
            status.update(label="✅ Simulation Complete", state="complete", expanded=False)
            
            # Step 4: Result Display
            if new_trades:
                st.success(f"📝 Queued {len(new_trades)} order(s) for Market Open.")
                st.dataframe(pd.DataFrame({"Orders": new_trades}))
            else:
                st.info("ℹ️ Scan finished successfully, but NO trades were executed.")
                with st.expander("🔍 Why? Possible Reasons"):
                    st.markdown("""
                    * **No Signals:** Entry rules were not met for any stock today.
                    * **Risk Governor:** 60% Sector Limit prevented new positions.
                    * **Duplicates:** Candidates are already owned.
                    """)
                    
        except Exception as e:
            status.update(label="❌ Simulation Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.exception(e) # Show full stack trace
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🔄 Run Daily Cycle", type="primary"):
            with st.spinner("Processing..."):
                pt.update_valuations()
                pt.process_exits(strategies_map)
            st.success("Cycle Complete.")
            st.rerun()
    with c2:
        if st.button("📡 Refresh Prices"):
            with st.spinner("Fetching quotes..."):
                pt.update_valuations()
            st.success("Prices Updated.")
            st.rerun()
    with c3:
        if st.button("⚠️ Reset Account"):
            pt.reset_account()
            st.rerun()

    st.subheader("📂 Active Holdings")
    col_widths = [1.2, 1.0, 1.2, 1.2, 1.2, 1.5, 2.5, 1.2]
    h_cols = st.columns(col_widths)
    headers = ["Symbol", "Qty", "Entry", "Current", "Stop Loss", "PnL", "Exit Plan", "Action"]
    for col, h in zip(h_cols, headers): col.markdown(f"**{h}**")
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
                success, msg = pt.close_position(sym, reason="Manual")
                if success:
                    st.toast(f"✅ {msg}")
                    st.rerun()
                else: st.error(msg)
            st.markdown("<hr style='margin: 5px 0'>", unsafe_allow_html=True)
    else:
        st.info("Portfolio is empty.")
        
    st.subheader("⏳ Pending Orders (Market-On-Open)")
    
    pending = state.get("pending_orders", [])
    if pending:
        # --- PHASE 3 FIX: Optimized Dataframe Sanitization ---
        df_pending = pd.DataFrame(pending)
        
        # Safe drop of 'strategy_obj' if it exists
        if "strategy_obj" in df_pending.columns:
            df_pending = df_pending.drop(columns=["strategy_obj"])
            
        # Display nicely
        st.dataframe(
            df_pending.style.format({
                "committed_cash": "${:,.0f}", 
                "order_price_estimate": "${:.2f}"
            }), 
            use_container_width=True
        )
        
        # Calculate Logic for display
        reserved = sum(o.get('committed_cash', 0) for o in pending)
        avail = state['cash'] - reserved
        
        c1, c2 = st.columns(2)
        c1.metric("Reserved Cash", f"${reserved:,.0f}")
        c2.metric("True Available", f"${avail:,.0f}")
        
        # Add a clear manual trigger for filling
        if st.button("🔔 Process Pending Orders (Morning Fill)", type="primary"):
            with st.spinner("Executing Market-On-Open orders..."):
                fill_logs = pt.process_pending_orders()
            
            if fill_logs:
                for log in fill_logs:
                    if "✅" in log: st.success(log)
                    else: st.error(log)
                st.session_state['portfolio_updated'] = True
                st.rerun()
            else:
                st.info("No orders filled (check gap protection or data availability).")
                
    else:
        st.caption("No orders queued for tomorrow.")
