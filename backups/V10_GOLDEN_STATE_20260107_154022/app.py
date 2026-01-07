import streamlit as st
import pandas as pd
import concurrent.futures
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import List

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import (
    MIN_ENTRY_SCORE,
    _compute_indicators,
    calculate_backtest_quality_score,
    prepare_backtest_data,
    run_backtest,
)
from execution.parity import (
    DEFAULT_SCORING_WEIGHTS,
    resolve_signal_index,
    get_strategy_weights,
    apply_strategy_score_multipliers,
    apply_gap_atr_stop_penalty,
    compute_limit_fill,
)
from execution.shared_logic import calc_exit_plan, _exit_plan_style, rehydrate_exit_state
from simulation.paper_trader import PaperTrader, MAX_POSITIONS
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
    with col2:
        show_all_setups = st.checkbox("🔍 Show All Setups", value=True)
    
    if run_btn:
        progress_bar = st.progress(0, text="📡 Initializing Data Engine...")
        status_msg = st.empty()
        timer_msg = st.empty()
        start_time = time.time()

        status_msg.info("📦 Fetching S&P 1500 Foundation...")
        progress_bar.progress(0.2, text="20% Complete")
        symbols = get_index_symbols(universe)
        base_days = 400
        data = fetch_data_pack(
            symbols,
            days=base_days,
            max_workers=10,
            force_fresh=True,
            inject_live=True,
            max_lag_days=0,
        )
        g_data = fetch_data_pack(
            ["SPY", "$VIX", "VIX"],
            days=600,
            max_workers=10,
            force_fresh=True,
            inject_live=True,
            max_lag_days=0,
        ) or {}
        spy_df = g_data.get("SPY")
        vix_df = g_data.get("$VIX")
        if vix_df is None:
            vix_df = g_data.get("VIX")
        results = []
        
        if not selected_strategies:
            st.warning("No strategies selected!")
            progress_bar.empty()
            status_msg.empty()
            timer_msg.empty()
        else:
            strat_objects = load_strategies(selected_strategies)
            total_symbols = len(data)
            status_msg.write(f"🔍 Analyzing symbols... (0/{total_symbols})")
            ema_seconds = None
            ema_alpha = 0.2
            timer_msg.caption("⏱️ Calibrating...")
            for i, (sym, df) in enumerate(data.items(), start=1):
                symbol_start = time.time()
                if df is not None and not df.empty:
                    try:
                        df_ind = _compute_indicators(df.copy(), spy_df=spy_df, vix_df=vix_df)
                        signal_i, current_i = resolve_signal_index(df_ind)
                        if signal_i < 0:
                            continue
                        row_signal = df_ind.iloc[signal_i]
                        row_current = df_ind.iloc[current_i]

                        for strat in strat_objects:
                            s_conf = strat.params or {}
                            entry_signal = strat.entry(df_ind, signal_i)
                            entry_ok = bool(entry_signal)

                            prev_close = row_signal.get("close", 0.0) or 0.0
                            open_px = row_current.get("open", row_current.get("close", 0.0)) or 0.0
                            low_px = row_current.get("low", open_px)

                            limit_ratio = None
                            if isinstance(entry_signal, dict):
                                limit_ratio = entry_signal.get("limit_ratio")
                            if limit_ratio is None:
                                limit_ratio = s_conf.get("limit_ratio")

                            filled, entry_px = compute_limit_fill(prev_close, open_px, low_px, limit_ratio)
                            if not (entry_px and entry_px == entry_px):
                                entry_px = open_px or row_current.get("close", 0.0)
                            entry_ok = entry_ok and filled and entry_px > 0

                            signal_atr = row_signal.get("atr14", prev_close * 0.02)
                            if not (signal_atr and signal_atr == signal_atr):
                                signal_atr = prev_close * 0.02

                            stop_mult = s_conf.get("stop_loss_atr", 3.0)
                            if isinstance(entry_signal, dict) and "stop_loss_atr" in entry_signal:
                                stop_mult = entry_signal.get("stop_loss_atr", stop_mult)
                            stop_mult = float(stop_mult)

                            gap_pct = ((open_px - prev_close) / prev_close) if prev_close > 0 else 0.0
                            atr_pct = (signal_atr / entry_px) * 100.0 if entry_px > 0 else 0.0
                            adj_mult = apply_gap_atr_stop_penalty(stop_mult, gap_pct, atr_pct)
                            stop_price = entry_px - (signal_atr * adj_mult)

                            weights = get_strategy_weights(s_conf)
                            raw_score = calculate_backtest_quality_score(
                                row_signal,
                                s_conf.get("name", ""),
                                weights,
                            )
                            score = apply_strategy_score_multipliers(raw_score, s_conf)

                            exits = s_conf.get("exit_rules", [])
                            target_txt = (
                                f"${row_signal['close'] * float(exits[0].get('val')):.2f}"
                                if exits and exits[0].get("type") == "profit_target"
                                else "OPEN"
                            )

                            results.append(
                                {
                                    "Symbol": sym,
                                    "Strategy": s_conf.get("name", strat.name),
                                    "Price": row_current["close"],
                                    "EntryPx": entry_px,
                                    "RSI2": row_current.get("rsi2"),
                                    "Stop Loss": stop_price,
                                    "Target": target_txt,
                                    "Score": score,
                                    "Entry_OK": entry_ok,
                                }
                            )
                    except Exception:
                        pass

                step_time = time.time() - symbol_start
                if ema_seconds is None:
                    ema_seconds = step_time
                else:
                    ema_seconds = (ema_alpha * step_time) + ((1 - ema_alpha) * ema_seconds)

                if total_symbols:
                    pct = i / total_symbols
                    progress_pct = 0.2 + (0.8 * pct)
                else:
                    progress_pct = 0.2

                if i <= 50:
                    eta_text = "⏱️ Calibrating..."
                else:
                    est_remaining = (total_symbols - i) * (ema_seconds or 0)
                    eta_text = f"⏱️ Estimated time remaining: {int(est_remaining)}s"

                if i % 15 == 0 or i == total_symbols:
                    progress_bar.progress(
                        min(progress_pct, 1.0),
                        text=f"{int(progress_pct*100)}% Complete",
                    )
                    status_msg.write(f"🔍 Analyzing **{sym}** ({i}/{total_symbols})")
                    timer_msg.caption(eta_text)

            held_syms = set()
            pending_syms = set()
            available_slots = MAX_POSITIONS
            if selected_strategies:
                pt_state = PaperTrader(configs=selected_strategies).state
                held_syms = set(pt_state.get("positions", {}).keys())
                pending_syms = {
                    o.get("symbol")
                    for o in pt_state.get("pending_orders", [])
                    if o.get("symbol")
                }
                available_slots = max(0, MAX_POSITIONS - len(held_syms) - len(pending_syms))

            if results:
                results.sort(
                    key=lambda x: x.get("Score", 0.0),
                    reverse=True,
                )
                slots_remaining = available_slots
                for row in results:
                    sym = row.get("Symbol")
                    score_val = row.get("Score", 0.0) or 0.0
                    entry_ok = bool(row.pop("Entry_OK", False))
                    rsi2_val = row.pop("RSI2", None)
                    rsi2_num = None
                    try:
                        if rsi2_val is not None:
                            rsi2_num = float(rsi2_val)
                    except Exception:
                        rsi2_num = None
                    if sym in held_syms:
                        status_txt = "ℹ️ HELD"
                    elif sym in pending_syms:
                        status_txt = "✅ PENDING"
                    elif not entry_ok:
                        if rsi2_num is not None and rsi2_num > 20:
                            status_txt = "⚠️ WAIT: RSI2 High"
                        else:
                            status_txt = "⚠️ WAIT: Pattern Incomplete"
                    elif score_val < MIN_ENTRY_SCORE:
                        status_txt = f"⚠️ SKIP: Low Score ({score_val:.1f})"
                    elif slots_remaining <= 0:
                        status_txt = "⚠️ REJECTED: Slots Full"
                    else:
                        status_txt = "✅ TRADABLE"
                        slots_remaining -= 1
                    row["Status"] = status_txt
            st.session_state.scan_results = pd.DataFrame(results)
            progress_bar.empty()
            status_msg.empty()
            timer_msg.empty()

    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results
        if not show_all_setups and "Status" in df.columns:
            df = df[df["Status"] == "✅ TRADABLE"]
        if not df.empty and "Status" in df.columns and "Score" in df.columns:
            def get_sort_weight(status: str) -> int:
                if "✅ TRADABLE" in status:
                    return 0
                if "PENDING" in status:
                    return 1
                if "ℹ️ HELD" in status:
                    return 2
                return 3

            df = df.copy()
            df["sort_weight"] = df["Status"].apply(get_sort_weight)
            df = df.sort_values(by=["sort_weight", "Score"], ascending=[True, False])
            df = df.drop(columns=["sort_weight"])

        if df.empty:
            st.info("No signals found today.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Signals", len(df))
            top_score = df["Score"].max()
            c2.metric("Top Score", f"{top_score:.1f}")
            c3.metric("Top Strategy", df.iloc[0]["Strategy"])
            
            def _status_style(series: pd.Series) -> List[str]:
                styles = []
                for val in series:
                    plan_style = _exit_plan_style(val)
                    if plan_style:
                        styles.append(plan_style)
                    elif isinstance(val, str) and (val.startswith("✅") or val.startswith("⏳")):
                        styles.append("background-color: #1a7f37; color: #ffffff; font-weight: 600;")
                    elif isinstance(val, str) and val.startswith("⚠️ WAIT"):
                        styles.append("background-color: #f1c232; color: #000000; font-weight: 600;")
                    elif isinstance(val, str) and val.startswith("ℹ️"):
                        styles.append("background-color: #0b5394; color: #ffffff; font-weight: 600;")
                    elif isinstance(val, str) and val.startswith("⚠️"):
                        styles.append("background-color: #8b0000; color: #ffffff; font-weight: 600;")
                    else:
                        styles.append("")
                return styles

            styled = (
                df.style.format({
                    "Price": "${:.2f}",
                    "EntryPx": "${:.2f}",
                    "Stop Loss": "${:.2f}",
                    "Score": "{:.1f}",
                })
                .apply(_status_style, subset=["Status"])
            )
            st.dataframe(styled, use_container_width=True)

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
                    data = fetch_data_pack(symbols, days=days + 200, backtest_mode=True) or {}

                    # Fetch global context once (required for RS + VIX overlays in the engine)
                    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days + 200, backtest_mode=True) or {}
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
                # Safe data extraction
                equity_curve = res.get("equity_curve")
                # Fallback chain to ensure we never get "Unknown" if the key exists
                strategy_name = res.get("strategy_name") or res.get("strategy") or name or "Backtest_Result"
                # Sanitize filename (remove special chars)
                safe_name = "".join(
                    [c for c in strategy_name if c.isalnum() or c in (" ", "_", "-")]
                ).strip()

                if equity_curve is not None and not isinstance(equity_curve, list) and not equity_curve.empty:
                    try:
                        csv_data = equity_curve.to_csv().encode('utf-8')
                        st.download_button(
                            label="📥 Export Result (CSV)",
                            data=csv_data,
                            file_name=f"{safe_name}_backtest.csv",
                            mime="text/csv",
                            key=f"dl_{safe_name}_{i}"  # Ensure 'i' comes from the loop variable
                        )
                    except Exception as e:
                        st.error(f"⚠️ Export failed for {strategy_name}: {e}")
                else:
                    st.warning(f"⚠️ No equity data for {strategy_name}")

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
    
    if st.button("🔭 PHASE 1: Scan for New Entries (Evening/Market Close)", type="primary"):
        status = st.status("🚀 Initializing Simulation...", expanded=True)
        try:
            status.write("1️⃣ Verifying Strategies...")
            if not getattr(pt, 'strategies', None):
                status.update(label="❌ No Strategies Loaded!", state="error")
                st.error("PaperTrader has 0 loaded strategies. Check config.")
                st.stop()
            
            status.write("2️⃣ Executing Scan & Governor...")
            symbols = get_index_symbols("S&P 1500")
            base_days = 400
            data_pack = fetch_data_pack(
                symbols,
                days=base_days,
                inject_live=True,
                max_lag_days=0,
            )
            g_data = fetch_data_pack(
                ["SPY", "$VIX", "VIX"],
                days=600,
                inject_live=True,
                max_lag_days=0,
            ) or {}
            spy_df = g_data.get("SPY")
            vix_df = g_data.get("$VIX")
            if vix_df is None:
                vix_df = g_data.get("VIX")
            global_data = {"SPY": spy_df, "VIX": vix_df}
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress(done, total):
                if total:
                    progress_bar.progress(min(done / total, 1.0))
                    status_text.text(f"Scanning... {done}/{total}")
                else:
                    status_text.text("Scanning...")

            scan_result = pt.run_daily_scan(
                data_pack,
                global_data=global_data,
                progress_callback=_progress,
            )
            orders = scan_result.get("orders", []) if scan_result else []
            logs = scan_result.get("logs", []) if scan_result else []
            queued_count = scan_result.get("count", len(orders)) if scan_result else 0
            status_text.empty()
            progress_bar.empty()
            
            status.write(f"3️⃣ Scan Complete. Orders Queued: {queued_count}")
            status.update(label="✅ Simulation Complete", state="complete", expanded=False)
            
            if orders:
                st.success(f"📝 Queued {queued_count} order(s).")
                df_orders = pd.DataFrame(orders)
                if "strategy_obj" in df_orders.columns:
                    df_orders = df_orders.drop(columns=["strategy_obj"])
                st.dataframe(df_orders)
            else:
                st.info("ℹ️ Scan finished. No trades executed.")
            if logs:
                rejection_logs = [log for log in logs if "REJECTED" in log or "SKIPPED" in log]
                if rejection_logs:
                    with st.expander("📋 Rejection Logs"):
                        st.text("\n".join(rejection_logs))
        except Exception as e:
            status.update(label="❌ Simulation Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.exception(e)
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("⚙️ PHASE 2: Execute Morning Fills & Exits (Market Open)", type="primary"):
            with st.spinner("Processing..."):
                pt.update_valuations()
                # Process exits before fills to keep the slot limit intact.
                exit_logs = pt.process_exits(strategies_map)
                fill_logs = pt.process_pending_orders()
            if exit_logs:
                for log in exit_logs:
                    st.info(log)
            if fill_logs:
                for log in fill_logs:
                    if "✅" in log:
                        st.success(log)
                    elif "❌" in log:
                        st.error(log)
                    else:
                        st.info(log)
            st.success("Cycle Complete.")
            st.rerun()
    with c2:
        if st.button("📡 Live Portfolio Mark-to-Market"):
            with st.spinner("Fetching quotes..."):
                pt.update_valuations()
            st.success("Prices Updated.")
            st.rerun()
    with c3:
        if st.button("⚠️ Emergency System Reset (Wipe All State)"):
            pt.reset_account()
            st.rerun()

    st.subheader("📂 Active Holdings")
    
    if state['positions']:
        symbols = list(state["positions"].keys())
        history_pack = fetch_data_pack(symbols, days=400, inject_live=True) or {}
        col_widths = [1.2, 1.0, 1.2, 1.2, 1.2, 2.5, 1.8, 1.8, 1.2]
        headers = [
            "Symbol",
            "Qty",
            "Entry",
            "Current",
            "Stop Loss",
            "Exit Plan Mandate",
            "PnL",
            "Current Value",
            "Action",
        ]
        h_cols = st.columns(col_widths)
        for col, h in zip(h_cols, headers):
            col.markdown(f"**{h}**")
        st.markdown("---")

        for sym, p in state['positions'].items():
            entry = p['entry_price']
            curr = p.get('current_price', entry)
            stop = p.get('stop_price', 0.0)
            shares = p['shares']
            pnl_val_trade = (curr - entry) * shares
            pnl_pct = ((curr - entry) / entry) * 100 if entry else 0.0
            current_val = shares * curr
            entry_val = shares * entry

            plan = calc_exit_plan(
                {
                    "Symbol": sym,
                    "Strategy": p.get("strategy_name", ""),
                    "Entry Price": entry,
                    "Stop Loss": stop,
                    "Date": p.get("date", datetime.now()),
                    "genome": p.get("genome"),
                    "strategy_obj": p.get("strategy_obj"),
                    "entry_i": p.get("entry_i"),
                },
                strategies_map,
                history_pack,
            )
            if not plan:
                plan = "Unknown"

            c_cols = st.columns(col_widths)
            c_cols[0].write(f"**{sym}**")
            c_cols[1].write(f"{shares}")
            c_cols[2].write(f"${entry:,.2f}")
            c_cols[3].write(f"${curr:,.2f}")
            c_cols[4].write(f"${stop:,.2f}" if stop else "N/A")
            plan_style = _exit_plan_style(plan)
            if plan_style:
                c_cols[5].markdown(
                    f"<span style='{plan_style} padding: 2px 6px; border-radius: 4px; display: inline-block;'>{plan}</span>",
                    unsafe_allow_html=True,
                )
            else:
                c_cols[5].write(plan)

            pnl_color = "#1a7f37" if pnl_val_trade > 0 else "#b00020" if pnl_val_trade < 0 else "#6b7280"
            pnl_text = f"${pnl_val_trade:,.2f} ({pnl_pct:+.2f}%)"
            c_cols[6].markdown(
                f"<span style='color:{pnl_color}; font-weight:600;'>{pnl_text}</span>",
                unsafe_allow_html=True,
            )

            val_color = "#1a7f37" if current_val > entry_val else "#b00020"
            c_cols[7].markdown(
                f"<span style='color:{val_color}; font-weight:600;'>${current_val:,.2f}</span>",
                unsafe_allow_html=True,
            )

            if c_cols[8].button("SELL", key=f"sell_{sym}", use_container_width=True):
                success, msg = pt.close_position(sym, reason="Manual")
                if success:
                    st.toast(f"✅ {msg}")
                    st.rerun()
                else:
                    st.error(msg)

            st.markdown("<hr style='margin: 5px 0'>", unsafe_allow_html=True)
    else:
        st.info("Portfolio is empty.")
    
    st.subheader("⏳ Pending Orders (Market-On-Open)")
    if "pending_fill_logs" not in st.session_state:
        st.session_state.pending_fill_logs = []
    pending = state.get("pending_orders", [])
    if pending:
        col_widths = [1.2, 0.8, 1.0, 2.2, 1.6, 1.0]
        h_cols = st.columns(col_widths)
        headers = ["Symbol", "Shares", "Est Price", "Strategy", "Queued At", "Action"]
        for col, h in zip(h_cols, headers):
            col.markdown(f"**{h}**")
        st.markdown("---")

        for idx, order in enumerate(pending):
            c_cols = st.columns(col_widths)
            sym = order.get("symbol", "")
            shares = order.get("shares", 0)
            est_price = order.get("order_price_estimate", 0)
            strat = order.get("strategy_name") or order.get("strategy") or ""
            queued_at = order.get("queued_at", "")
            c_cols[0].write(f"**{sym}**")
            c_cols[1].write(f"{shares}")
            c_cols[2].write(f"${est_price:.2f}" if est_price else "N/A")
            c_cols[3].write(strat)
            c_cols[4].write(queued_at)
            if c_cols[5].button("CANCEL", key=f"cancel_{sym}_{idx}", use_container_width=True):
                st.session_state.pending_fill_logs = pt.cancel_pending_order(sym)
                st.rerun()
            st.markdown("<hr style='margin: 5px 0'>", unsafe_allow_html=True)

        if st.button("🔔 Process Pending Orders (Morning Fill)", type="primary"):
            with st.spinner("Executing Market-On-Open orders..."):
                fill_logs = pt.process_pending_orders()
            st.session_state.pending_fill_logs = fill_logs or ["ℹ️ No orders filled."]
    else:
        st.caption("No orders queued for tomorrow.")

    if st.session_state.pending_fill_logs:
        for log in st.session_state.pending_fill_logs:
            if "✅" in log:
                st.success(log)
            elif "❌" in log:
                st.error(log)
            elif "⏳" in log:
                st.warning(log)
            else:
                st.info(log)

    st.subheader("📜 Professional Trade Ledger")
    if st.button("🔄 Sync Ledger with Active Holdings"):
        pt.sync_active_to_ledger()
        st.success("Ledger updated with current positions.")
        st.rerun()
    ledger_path = "data/sim_trade_history.csv"
    if os.path.exists(ledger_path):
        try:
            history_df = pd.read_csv(ledger_path)
        except Exception as e:
            st.warning(f"Could not load trade history: {e}")
        else:
            if history_df.empty:
                st.caption("No closed trades yet.")
            else:
                if "Reason" in history_df.columns:
                    initial_buys = history_df[history_df["Reason"] == "INITIAL_BUY"].tail(10)
                    if not initial_buys.empty:
                        st.caption("Latest INITIAL_BUY entries")
                        st.dataframe(initial_buys, use_container_width=True)
                st.dataframe(history_df.tail(20), use_container_width=True)
                csv_data = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Performance Audit (CSV)",
                    data=csv_data,
                    file_name="sim_trade_history.csv",
                    mime="text/csv",
                )
    else:
        st.caption("No closed trades yet.")
