import streamlit as st
import pandas as pd
import concurrent.futures
import sys
import os
import json
import time
from datetime import datetime, timedelta

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import (
    DEFAULT_SCORING_WEIGHTS,
    MIN_ENTRY_SCORE,
    _compute_indicators,
    calculate_backtest_quality_score,
    prepare_backtest_data,
    run_backtest,
)
from simulation.paper_trader import PaperTrader
from strategies.strategy_loader import load_strategies

CONFIG_PATH = "config/generated_strategies.json"
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")

# --- HELPERS ---
def _is_wealth_strategy(config: dict) -> bool:
    name = str(config.get("name", "")).lower()
    strat_type = str(config.get("type", "")).lower()
    return "wealth" in strat_type or "wealth" in name

def load_strategy_configs():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [s for s in payload if isinstance(s, dict) and _is_wealth_strategy(s)]
    return []

def format_rule(r):
    return f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}"

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else 'white'
    return f'color: {color}'

def score_to_rating(score):
    if score >= 85: return "🔥 Excellent"
    elif score >= 75: return "⭐ Strong"
    elif score >= 60: return "✅ Good"
    elif score >= 50: return "⚠️ Fair"
    return "❌ Weak"

def calc_exit_plan(row, strategies_map):
    def _unwrap_genome(value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                return None
        if isinstance(value, dict):
            for _ in range(3):
                nested = None
                for key in ("genome", "params", "strategy"):
                    if isinstance(value.get(key), dict):
                        nested = value.get(key)
                        break
                if nested is None:
                    break
                value = nested
        return value if isinstance(value, dict) else None

    entry_price = row.get("Entry Price", 0) or 0
    try:
        entry_price = float(entry_price)
    except Exception:
        entry_price = 0.0

    genome = _unwrap_genome(
        row.get("genome")
        or row.get("Genome")
        or row.get("params")
        or row.get("Params")
        or row.get("strategy_genome")
    )
    if genome is None:
        strat_obj = row.get("strategy_obj")
        if strat_obj is not None:
            genome = _unwrap_genome(getattr(strat_obj, "params", None) or getattr(strat_obj, "genome", None))

    strat = genome
    strat_name = row.get("Strategy") or row.get("strategy_name") or row.get("strategy") or ""
    if strat is None and isinstance(strat_name, dict):
        strat = _unwrap_genome(strat_name)
    if strat is None:
        if isinstance(strat_name, str):
            strat_name = strat_name.split(" + ")[0].strip()
        else:
            strat_name = str(strat_name).split(" + ")[0].strip()
        strat = strategies_map.get(strat_name)
        if not strat and strat_name:
            def _normalize(name: str) -> str:
                return "".join(ch for ch in str(name).lower() if ch.isalnum())

            target = _normalize(strat_name)
            best_key = ""
            best_cfg = None
            for key, cfg in strategies_map.items():
                key_norm = _normalize(str(key))
                if not key_norm:
                    continue
                if key_norm in target or target in key_norm:
                    if len(key_norm) > len(best_key):
                        best_key = key_norm
                        best_cfg = cfg
            strat = best_cfg
    if not strat:
        return "Unknown"

    exits = strat.get("exit_rules") or []
    if isinstance(exits, dict):
        exits = [exits]
    for rule in exits:
        if not isinstance(rule, dict):
            continue
        if rule.get("type") == "profit_target":
            try:
                target_px = entry_price * float(rule.get("val"))
                return f"Target: ${target_px:.2f}"
            except Exception:
                return "Target: N/A"

    time_stop = strat.get("time_stop", 70)
    try:
        time_stop = int(time_stop)
    except Exception:
        time_stop = 70
    try:
        entry_date = pd.to_datetime(row.get("Date"))
        sell_date = entry_date + timedelta(days=time_stop)
        days_left = (sell_date.date() - datetime.now().date()).days
        if days_left < 0:
            return "Time Limit (Sell)"
        return f"Hold ({days_left}d left)"
    except Exception:
        return f"Hold {time_stop}d"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.caption("Institutional Grade Algo System")
    st.markdown("---")
    st.success("🏆 APEX V9 MEDALLION: RAW ALPHA ACTIVE")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    
    st.markdown("### 📘 Active Strategies")
    strategies_list = load_strategy_configs()
    strategies_map = {s['name']: s for s in strategies_list}
    
    selected_strategies = []
    if strategies_list:
        for i, s in enumerate(strategies_list):
            strat_name = s.get("name", "")
            use = st.checkbox(strat_name, value=True, key=f"chk_{strat_name}_{i}")
            if use: selected_strategies.append(s)
            
            with st.expander(f"📘 Strategy Guide: {s.get('name', 'Strategy')}"):
                try:
                    limit_ratio = float(s.get("limit_ratio")) if s.get("limit_ratio") is not None else None
                except Exception:
                    limit_ratio = None

                try:
                    stop_loss_atr = float(s.get("stop_loss_atr", 3.0) or 3.0)
                except Exception:
                    stop_loss_atr = 3.0

                try:
                    trail_activation = float(s.get("trail_activation", 1.0) or 1.0)
                except Exception:
                    trail_activation = 1.0

                try:
                    trail_atr = float(s.get("trail_atr", stop_loss_atr) or stop_loss_atr)
                except Exception:
                    trail_atr = stop_loss_atr

                try:
                    time_stop = int(s.get("time_stop", 45) or 45)
                except Exception:
                    time_stop = 45
                time_stop = max(1, time_stop)

                limit_desc = "market/open (no limit ratio set)"
                if limit_ratio is not None and limit_ratio > 0:
                    limit_desc = f"{limit_ratio:.2f}× prior close (~{(1.0 - limit_ratio) * 100.0:.1f}% below)"

                trail_activation_desc = f"+{(trail_activation - 1.0) * 100.0:.0f}% (activation {trail_activation:.2f})"

                st.markdown(
                    f"""
**🔭 Strategy Mandate**
- **Elite Sniper Gate:** only enter when **Score ≥ {MIN_ENTRY_SCORE:.0f}**.
- **RSI Pullback Rule:** only enter when **RSI2 < 20**.

**⚔️ Execution Protocol**
- **Limit Orders:** fills at {limit_desc}.

**🛡️ Risk & Exit**
- **ATR Stop:** {stop_loss_atr:.2f}× ATR below entry is the sole risk floor.
- **Trailing Stop:** dormant until {trail_activation_desc}; then trails at {trail_atr:.2f}× ATR.

**⏳ Time Exit**
- Automatic exit after **{time_stop}** trading days.
"""
                )
    else:
        st.error("⚠️ No strategies found in config file!")

# --- 1. LIVE SCREENER ---
if mode == "Live Screener":
    st.header("🚀 Daily Opportunity Scanner")
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    col1, col2 = st.columns([1, 4])
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 100", "S&P 1500"], index=2)
        run_btn = st.button("RUN SCAN", type="primary")
    
    if run_btn:
        with st.spinner(f"Scanning {universe}..."):
            symbols = get_index_symbols(universe)
            base_days = 400
            data = fetch_data_pack(symbols, days=base_days)
            g_data = fetch_data_pack(["SPY"], days=base_days + 200) or {}
            spy_df = g_data.get("SPY")
            results = []
            
            if not selected_strategies:
                st.warning("No strategies selected!")
            else:
                strat_objects = load_strategies(selected_strategies)
                total_symbols = len(data)
                progress_bar = st.progress(0)
                status = st.empty()
                for i, (sym, df) in enumerate(data.items(), start=1):
                    status.write(f"Scanning {sym} ({i}/{total_symbols})")
                    if total_symbols:
                        progress_bar.progress(min(i / total_symbols, 1.0))
                    if df is None or df.empty:
                        continue
                    try:
                        df_ind = _compute_indicators(df.copy(), spy_df=spy_df)
                        if df_ind.empty or len(df_ind) < 2:
                            continue

                        signal_i = len(df_ind) - 2
                        row_signal = df_ind.iloc[signal_i]
                        row_current = df_ind.iloc[-1]

                        for strat in strat_objects:
                            s_conf = strat.params or {}
                            if not strat.entry(df_ind, signal_i):
                                continue

                            atr = row_signal.get("atr14", row_signal["close"] * 0.02)
                            stop_mult = float(s_conf.get("stop_loss_atr", 3.0))

                            raw_score = calculate_backtest_quality_score(
                                row_signal,
                                s_conf.get("name", ""),
                                DEFAULT_SCORING_WEIGHTS,
                            )
                            score = raw_score * 1.3 if "wealth" in str(s_conf.get("name", "")).lower() else raw_score

                            exits = s_conf.get("exit_rules", [])
                            target_txt = (
                                f"${row_signal['close'] * float(exits[0].get('val')):.2f}"
                                if exits and exits[0].get("type") == "profit_target"
                                else "OPEN"
                            )

                            estimated_entry = row_current["close"]

                            results.append(
                                {
                                    "Symbol": sym,
                                    "Strategy": s_conf.get("name", strat.name),
                                    "Price": row_current["close"],
                                    "Stop Loss": estimated_entry - (atr * stop_mult),
                                    "Target": target_txt,
                                    "Score": score,
                                }
                            )
                    except Exception:
                        continue
                status.empty()
                progress_bar.empty()
                
                if results:
                    results.sort(
                        key=lambda x: x.get("Score", 0.0),
                        reverse=True,
                    )
                st.session_state.scan_results = pd.DataFrame(results)

    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results

        if df.empty:
            st.info("No signals found today.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Signals", len(df))
            top_score = df["Score"].max()
            c2.metric("Top Score", f"{top_score:.1f}")
            c3.metric("Top Strategy", df.iloc[0]["Strategy"])
            
            st.dataframe(
                df.style.format({
                    "Price": "${:.2f}", 
                    "Stop Loss": "${:.2f}",
                    "Score": "{:.1f}",
                }), 
                use_container_width=True
            )

# --- 2. BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
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
    
    bt_duration = st.session_state.bt_duration
    days = dur_map.get(bt_duration, 1260)
    cache_key = f"{bt_universe}|{bt_duration}"
    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}

    st.info(f"Settings: **{bt_universe}** for **{bt_duration}**")

    if st.button("🚀 RUN BACKTEST", type="primary"):
        if not selected_strategies:
            st.error("Please select at least one strategy.")
        else:
            cache = st.session_state.backtest_cache
            prepared = cache.get(cache_key)
            cache_hit = prepared is not None
            if cache_hit:
                st.success("⚡ Using Cached Data (Instant Mode Active)")

            with st.spinner("Simulating..."):
                if not cache_hit:
                    symbols = get_index_symbols(bt_universe)
                    data = fetch_data_pack(symbols, days=days + 200) or {}

                    # Fetch global context once (required for RS + VIX overlays in the engine)
                    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days + 200) or {}
                    spy_df = g_data.get("SPY")
                    vix_df = g_data.get("$VIX")
                    if vix_df is None:
                        vix_df = g_data.get("VIX")
                    global_data = {"SPY": spy_df, "VIX": vix_df}

                    # Turbo: precompute indicators/arrays once, then reuse across all strategies.
                    prepared = prepare_backtest_data(
                        data,
                        symbol_universe=symbols,
                        start_date=None,
                        global_data=global_data,
                    )
                    cache[cache_key] = prepared

                tested_symbols = len(getattr(prepared, "enriched", {}) or {})
                st.caption(f"Symbols tested: {tested_symbols}")

                results_map = {}
                run_strategies = load_strategies(selected_strategies)

                # Optimized Parallelism: 12 workers for standard runs.
                max_workers = 12

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_map = {
                        executor.submit(
                            run_backtest,
                            strat,
                            prepared,
                            start_cash=100000.0,
                            start_date=None,
                        ): strat.name
                        for strat in run_strategies
                    }

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    total_futures = len(future_map)
                    completed = 0
                    start_time = time.time()
                    while future_map:
                        done, _ = concurrent.futures.wait(
                            future_map.keys(),
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )

                        next_completed = completed + len(done)
                        if total_futures:
                            progress_bar.progress(min(next_completed / total_futures, 1.0))
                        status_text.text(
                            f"Running Simulations... ({next_completed}/{total_futures} Done) "
                            f"[Time Elapsed: {int(time.time() - start_time)}s]"
                        )
                        time.sleep(0.01)

                        for future in done:
                            name = future_map.pop(future, "Unknown")
                            try:
                                results_map[name] = future.result()
                            except Exception as e:
                                st.error(f"Backtest failed for {name}: {e}")
                            completed += 1
                    status_text.empty()
                    progress_bar.empty()

                st.session_state.backtest_results = results_map

    if "backtest_results" in st.session_state and st.session_state.backtest_results:
        tabs = st.tabs(list(st.session_state.backtest_results.keys()))
        for i, name in enumerate(st.session_state.backtest_results.keys()):
            res = st.session_state.backtest_results[name]
            with tabs[i]:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("CAGR", f"{res['cagr']:.1%}")
                col2.metric("Win Rate", f"{res['hit_rate']:.1f}%")
                col3.metric("Avg Profit", f"{res['avg_profit_pct']:.2f}%")
                col4.metric("Total Trades", res["total_trades"])
                
                st.line_chart(res["equity_curve"])
                
                # --- DOWNLOAD BUTTON RESTORED ---
                csv_data = res["equity_curve"].to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_data,
                    file_name=f"{name}_backtest.csv",
                    mime="text/csv",
                    key=f"dl_{i}"
                )

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
            status.write("1️⃣ Verifying Strategies...")
            if not getattr(pt, 'strategies', None):
                status.update(label="❌ No Strategies Loaded!", state="error")
                st.error("PaperTrader has 0 loaded strategies. Check config.")
                st.stop()
            
            status.write("2️⃣ Executing Scan & Governor...")
            symbols = get_index_symbols("S&P 1500")
            data_pack = fetch_data_pack(symbols, days=400)
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress(done, total):
                if total:
                    progress_bar.progress(min(done / total, 1.0))
                    status_text.text(f"Scanning... {done}/{total}")
                else:
                    status_text.text("Scanning...")

            new_trades = pt.run_daily_scan(data_pack, progress_callback=_progress)
            status_text.empty()
            progress_bar.empty()
            
            status.write(f"3️⃣ Scan Complete. Orders Queued: {len(new_trades) if new_trades else 0}")
            status.update(label="✅ Simulation Complete", state="complete", expanded=False)
            
            if new_trades:
                st.success(f"📝 Queued {len(new_trades)} order(s).")
                df_orders = pd.DataFrame(new_trades)
                if "strategy_obj" in df_orders.columns:
                    df_orders = df_orders.drop(columns=["strategy_obj"])
                st.dataframe(df_orders)
            else:
                st.info("ℹ️ Scan finished. No trades executed.")
        except Exception as e:
            status.update(label="❌ Simulation Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
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
    
    if state['positions']:
        col_widths = [1.2, 1.0, 1.2, 1.2, 1.2, 1.5, 2.5, 1.2]
        h_cols = st.columns(col_widths)
        headers = ["Symbol", "Qty", "Entry", "Current", "Stop Loss", "PnL", "Exit Plan", "Action"]
        for col, h in zip(h_cols, headers): col.markdown(f"**{h}**")
        st.markdown("---")

        for sym, p in state['positions'].items():
            c_cols = st.columns(col_widths)
            entry = p['entry_price']
            curr = p.get('current_price', entry)
            stop = p.get('stop_price', 0.0)
            shares = p['shares']
            pnl_val_trade = (curr - entry) * shares
            pnl_pct = ((curr - entry) / entry) * 100
            
            plan = calc_exit_plan(
                {
                    "Strategy": p.get("strategy_name", ""),
                    "Entry Price": entry,
                    "Date": p.get("date", datetime.now()),
                    "genome": p.get("genome"),
                    "strategy_obj": p.get("strategy_obj"),
                },
                strategies_map,
            )
            
            c_cols[0].write(f"**{sym}**")
            c_cols[1].write(f"{shares}")
            c_cols[2].write(f"${entry:.2f}")
            c_cols[3].write(f"${curr:.2f}")
            c_cols[4].markdown(f":red[${stop:.2f}]") 
            color = "green" if pnl_val_trade >= 0 else "red"
            c_cols[5].markdown(f":{color}[${pnl_val_trade:,.2f} ({pnl_pct:+.2f}%)]")
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
        df_pending = pd.DataFrame(pending)
        if "strategy_obj" in df_pending.columns:
            df_pending = df_pending.drop(columns=["strategy_obj"])
        st.dataframe(df_pending)
        
        if st.button("🔔 Process Pending Orders (Morning Fill)", type="primary"):
            with st.spinner("Executing Market-On-Open orders..."):
                fill_logs = pt.process_pending_orders()
            if fill_logs:
                for log in fill_logs:
                    if "✅" in log: st.success(log)
                    else: st.error(log)
                st.rerun()
            else:
                st.info("No orders filled.")
    else:
        st.caption("No orders queued for tomorrow.")
