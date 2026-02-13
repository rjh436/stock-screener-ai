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
from data.universe import get_universe_symbols
from execution.engine import (
    MIN_ENTRY_SCORE,
    calculate_backtest_quality_score,
    compute_stop_fill,
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
    return True  # SHOW ALL STRATEGIES (Debug Mode)

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


def normalize_equity_curve_df(equity_curve) -> pd.DataFrame:
    """Normalize equity rows to one tz-naive calendar date row for chart/export."""
    df_ec = pd.DataFrame(equity_curve or [])
    if df_ec.empty or "Date" not in df_ec.columns or "Equity" not in df_ec.columns:
        return pd.DataFrame(columns=["Date", "Equity"])
    df_ec["Date"] = pd.to_datetime(df_ec["Date"], errors="coerce")
    df_ec["Equity"] = pd.to_numeric(df_ec["Equity"], errors="coerce")
    df_ec = df_ec.dropna(subset=["Date", "Equity"])
    if df_ec.empty:
        return pd.DataFrame(columns=["Date", "Equity"])
    try:
        df_ec["Date"] = df_ec["Date"].dt.tz_localize(None)
    except Exception:
        pass
    df_ec["Date"] = df_ec["Date"].dt.normalize()
    df_ec = df_ec.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return df_ec[["Date", "Equity"]].reset_index(drop=True)

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

                if str(strat_name).strip().lower() == "superperformance":
                    rs_gate = float(s.get("rs_gate_min", 85) or 85)
                    growth_gate = float(s.get("fundamental_growth_min_pct", 20) or 20)
                    min_score = float(s.get("min_entry_score", 0) or 0)
                    max_stop = float(s.get("max_stop_pct", 0.06) or 0.06) * 100.0
                    time_stop_days = int(s.get("time_stop_days", 5) or 5)
                    risk_trade = float(s.get("risk_per_trade", 0.01) or 0.01) * 100.0
                    entry_mode = str(s.get("entry_mode", "both") or "both").upper()
                    st.markdown(
                        f"""
**🔭 Gate Logic**
- **Trend Template:** close > SMA10 > SMA20 > SMA50 > SMA150 > SMA200.
- **RS Gate:** percentile must be **≥ {rs_gate:.0f}**.
- **Fundamentals:** EPS or Sales YoY growth **≥ {growth_gate:.0f}%** (or HTF override when missing).

**⚔️ Entry Logic**
- **Archetype Union:** {entry_mode} (VCP breakout and/or Episodic Pivot).
- **Composite Score Floor:** **≥ {min_score:.1f}**.

**🛡️ Risk & Exit**
- **Initial Stop Cap:** max **{max_stop:.1f}%** risk width.
- **Risk Per Trade:** **{risk_trade:.2f}%** of equity.
- **Time Stop:** exit if dead money after **{time_stop_days}** days.
"""
                    )
                else:
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
        universe = st.selectbox(
            "Universe",
            ["SP500", "SP100", "SP1500", "NASDAQ100", "RUSSELL3000"],
            index=2,
        )
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
        if universe in ("Russell 3000", "RUSSELL3000"):
            # Explicit call to the Universe module for R3000
            symbols = get_universe_symbols("RUSSELL3000")
        else:
            # Standard indices
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
        global_data = {"SPY": spy_df, "VIX": vix_df}
        results = []
        
        if not selected_strategies:
            st.warning("No strategies selected!")
            progress_bar.empty()
            status_msg.empty()
            timer_msg.empty()
        else:
            strat_objects = load_strategies(selected_strategies)
            prepared_live = prepare_backtest_data(
                data,
                symbol_universe=symbols,
                start_date=None,
                global_data=global_data,
            )
            scan_items = list((prepared_live.enriched or {}).items())
            total_symbols = len(scan_items)
            status_msg.write(f"🔍 Analyzing symbols... (0/{total_symbols})")
            ema_seconds = None
            ema_alpha = 0.2
            timer_msg.caption("⏱️ Calibrating...")
            for i, (sym, sym_data) in enumerate(scan_items, start=1):
                symbol_start = time.time()
                df_ind = sym_data.df if sym_data is not None else None
                if df_ind is not None and not df_ind.empty:
                    try:
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
                            high_px = row_current.get("high", open_px) or open_px
                            low_px = row_current.get("low", open_px)

                            limit_ratio = None
                            trigger_px = None
                            stop_limit_pct = s_conf.get("stop_limit_pct", 0.02)
                            if isinstance(entry_signal, dict):
                                limit_ratio = entry_signal.get("limit_ratio")
                                trigger_px = entry_signal.get("trigger_price")
                                stop_limit_pct = entry_signal.get("stop_limit_pct", stop_limit_pct)
                            if limit_ratio is None:
                                limit_ratio = s_conf.get("limit_ratio")

                            filled = False
                            entry_px = open_px or row_current.get("close", 0.0)
                            try:
                                trigger_val = float(trigger_px) if trigger_px is not None else float("nan")
                            except Exception:
                                trigger_val = float("nan")
                            if trigger_val == trigger_val and trigger_val > 0:
                                filled, entry_px = compute_stop_fill(
                                    open_px,
                                    high_px,
                                    trigger_val,
                                    stop_limit_pct,
                                )
                            else:
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
                            stop_price = None
                            if isinstance(entry_signal, dict):
                                try:
                                    decision_stop = float(entry_signal.get("stop_price"))
                                except Exception:
                                    decision_stop = float("nan")
                                if decision_stop == decision_stop and 0 < decision_stop < entry_px:
                                    stop_price = decision_stop
                            if stop_price is None:
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
        # --- DATA PREP ---
        df = st.session_state.scan_results.copy()

        if df.empty:
            st.info("No signals found today.")
        else:
            # Bucket 1: Alpha Targets (Strictly Tradable)
            # Used for the Left Panel ("Buy Now")
            targets = df[
                (df["Status"].str.contains("✅ TRADABLE", na=False)) &
                (df["Score"] >= MIN_ENTRY_SCORE)
            ].sort_values("Score", ascending=False).head(15)

            # Bucket 2: Swap Pool (Tradable OR Blocked by Slots)
            # Used for the Middle Panel ("Smart Swaps")
            # CRITICAL: We include candidates blocked by 'Slots Full', but EXCLUDE 'Sector Cap' or 'Risk' blocks.
            swap_pool = df[
                (
                    df["Status"].str.contains("✅ TRADABLE", na=False) |
                    df["Status"].str.contains("REJECTED: Slots Full", na=False)
                ) &
                (df["Score"] >= MIN_ENTRY_SCORE)
            ].sort_values("Score", ascending=False)

            # Bucket 3: Watchtower
            watchtower = df[
                (df["Status"].str.contains("WAIT", na=False) | df["Status"].str.contains("REJECTED", na=False)) &
                (~df["Status"].str.contains("Slots Full", na=False)) &
                (df["Score"] >= 140)
            ].sort_values("Score", ascending=False).head(15)

            # --- LOGIC: SMART SWAPS ---
            # 1. Identify Stagnant Holdings
            pt = PaperTrader(configs=selected_strategies)
            pt_state = pt.state
            current_positions = pt_state.get("positions", {})

            stagnant_candidates = []
            for sym, pos in current_positions.items():
                entry_date = pd.to_datetime(pos["date"]).date()
                days_held = (datetime.now().date() - entry_date).days
                pnl_pct = pos.get("unrealized_pct", 0.0)

                # Stagnation Criteria: Held > 10 days AND PnL < 2.0%
                if days_held > 10 and pnl_pct < 2.0:
                    stagnant_candidates.append({
                        "Sell": sym,
                        "Days": days_held,
                        "PnL": f"{pnl_pct:.1f}%",
                        "Strategy": pos.get("strategy_name", "")
                    })

            # 2. Identify Best Swap Target (from the broader Swap Pool)
            best_buy = None
            if not swap_pool.empty:
                best_buy = swap_pool.iloc[0]

            # 3. Build Recommendations
            swap_recommendations = []
            for item in stagnant_candidates:
                rec = item.copy()
                if best_buy is not None:
                    # Scenario A: Swap into the waiting "Monster Setup"
                    rec["Action"] = "🔄 SWAP"
                    rec["Buy"] = best_buy["Symbol"]
                    rec["Upgrade_Score"] = f"{best_buy['Score']:.1f}"

                    # Context: Is the buy target currently blocked?
                    if "Slots Full" in str(best_buy["Status"]):
                        rec["Buy"] += " (Queue)"
                else:
                    # Scenario B: No valid targets -> Cash is King
                    rec["Action"] = "💰 LIQUIDATE"
                    rec["Buy"] = "CASH / WAIT"
                    rec["Upgrade_Score"] = "-"
                swap_recommendations.append(rec)

            st.markdown("### 🛸 Apex Command Center")
            # --- COMMAND CENTER DISPLAY LOGIC (Updated) ---

            # 1. PREPARE DATA (Robust Filter)
            # Capture ALL items that have a valid "Sell" symbol (covers both SWAP and LIQUIDATE)
            liquidation_list = [
                item for item in swap_recommendations 
                if item.get("Sell")
            ]
            liquidation_symbols = [item["Sell"] for item in liquidation_list]

            # 2. GLOBAL ACTION BUTTON (Full Width - Above Columns)
            if liquidation_symbols:
                st.warning(f"⚠️ Action Required: {len(liquidation_symbols)} positions are stagnant.")
                
                # Layout: Button on left, Summary on right
                b_col1, b_col2 = st.columns([1, 4])
                with b_col1:
                    # Updated Label: Clarifies that this handles Sells for both Swaps and Liquidations
                    if st.button(
                        f"💸 Flash Sell / Liquidate ({len(liquidation_symbols)})", 
                        type="primary", 
                        use_container_width=True,
                        key="btn_flash_liq"
                    ):
                        with st.spinner(f"Liquidating {', '.join(liquidation_symbols)}..."):
                            # Force Fresh State (Critical)
                            pt = PaperTrader(configs=selected_strategies)
                            logs = pt.liquidate_stagnant_holdings(liquidation_symbols)
                            
                        # Display Results
                        for log in logs:
                            if "✅" in log:
                                st.toast(log, icon="✅")
                            elif "⏳" in log:
                                st.toast(log, icon="⏳")
                            else:
                                st.error(log)
                        
                        time.sleep(1.5)
                        st.rerun()
                
                with b_col2:
                    # Added Context: Remind user that Swaps require a manual buy step
                    st.caption(f"**Queued for Exit:** {', '.join(liquidation_symbols)}. (Note: Swap entries must be queued manually).")
                    
            st.divider()

            # 3. PANELS LAYOUT
            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("🎯 Alpha Targets")
                if not targets.empty:
                    st.dataframe(
                        targets[["Symbol", "Strategy", "Score", "Price", "Stop Loss", "Target"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No Alpha Targets. Market quiet or slots full.")

            with col2:
                st.subheader("🔄 Smart Swaps (Advisory)")
                if swap_recommendations:
                    st.dataframe(pd.DataFrame(swap_recommendations), use_container_width=True, hide_index=True)
                    if best_buy is None:
                        st.caption("Strategy: Raise Cash (No Buys Available)")
                    else:
                        st.caption("Strategy: Swap to Upgrade")
                else:
                    st.success("🛡️ Portfolio Optimized")
        
                st.subheader("🔭 Watchtower")
                if not watchtower.empty:
                    st.dataframe(watchtower[["Symbol", "Score", "Status"]], use_container_width=True, hide_index=True)

            # --- SECTOR RADAR (Preserved) ---
            with st.expander("📊 Sector Risk Radar"):
                exposure = pt._current_sector_exposure()
                total_equity = pt_state["equity"]
                cols = st.columns(4)
                for i, (sec, val) in enumerate(exposure.items()):
                    pct = val / total_equity
                    with cols[i % 4]:
                        st.metric(sec, f"{pct:.1%}")
                        st.progress(min(pct / 0.60, 1.0))

            with st.expander("📂 View Full Raw Feed"):
                st.dataframe(df)

# --- 2. BACKTEST ---
elif mode == "Backtest":
    st.header("📈 Historical Performance Lab")
    
    col_uni, col_dur = st.columns([1, 3])
    with col_uni:
        bt_universe = st.selectbox(
            "Universe",
            ["SP500", "SP100", "SP1500", "NASDAQ100", "RUSSELL3000"],
            index=2,
        )
    
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
    year_map = {"1 Year": 1, "5 Years": 5, "10 Years": 10, "20 Years": 20}
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    bt_years = year_map.get(bt_duration)
    bt_start_ts = (today - pd.DateOffset(years=bt_years)).normalize() if bt_years else None
    bt_start_date = bt_start_ts.date().isoformat() if bt_start_ts is not None else None

    # Pull enough calendar history for the requested window + warmup bars.
    if bt_start_ts is not None:
        fetch_days = max(int((today - bt_start_ts).days), days) + 320
    else:
        fetch_days = days + 320

    # Versioned key avoids reusing old, shallow cached payloads from prior app sessions.
    cache_key = f"btv2|{bt_universe}|{bt_duration}|{bt_start_date or 'max'}"
    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}

    if bt_start_date:
        st.info(f"Settings: **{bt_universe}** for **{bt_duration}** (from **{bt_start_date}**)")
    else:
        st.info(f"Settings: **{bt_universe}** for **{bt_duration}**")

    if st.button("🚀 RUN BACKTEST", type="primary"):
        if not selected_strategies:
            st.error("Please select at least one strategy.")
        else:
            cache = st.session_state.backtest_cache
            prepared = None
            global_data = {}
            cached_payload = cache.get(cache_key)
            if isinstance(cached_payload, dict):
                prepared = cached_payload.get("prepared")
                global_data = cached_payload.get("global_data") or {}
            else:
                prepared = cached_payload
            cache_hit = prepared is not None
            if cache_hit:
                st.success("⚡ Using Cached Data (Instant Mode Active)")

            with st.spinner("Simulating..."):
                if not cache_hit:
                    if bt_universe in ("Russell 3000", "RUSSELL3000"):
                        # Explicit call to the Universe module for R3000
                        symbols = get_universe_symbols("RUSSELL3000")
                    else:
                        # Standard indices
                        symbols = get_index_symbols(bt_universe)
                    data = fetch_data_pack(symbols, days=fetch_days, backtest_mode=False) or {}

                    # Fetch global context once (required for RS + VIX overlays in the engine)
                    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=fetch_days, backtest_mode=False) or {}
                    spy_df = g_data.get("SPY")
                    vix_df = g_data.get("$VIX")
                    if vix_df is None:
                        vix_df = g_data.get("VIX")
                    global_data = {"SPY": spy_df, "VIX": vix_df}

                    # Turbo: precompute indicators/arrays once, then reuse across all strategies.
                    prepared = prepare_backtest_data(
                        data,
                        symbol_universe=symbols,
                        start_date=bt_start_date,
                        global_data=global_data,
                    )
                    cache[cache_key] = {
                        "prepared": prepared,
                        "global_data": global_data,
                    }
                elif not global_data:
                    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=fetch_days, backtest_mode=False) or {}
                    spy_df = g_data.get("SPY")
                    vix_df = g_data.get("$VIX")
                    if vix_df is None:
                        vix_df = g_data.get("VIX")
                    global_data = {"SPY": spy_df, "VIX": vix_df}

                tested_symbols = len(getattr(prepared, "enriched", {}) or {})
                st.caption(f"Symbols tested: {tested_symbols}")
                if getattr(prepared, "all_dates", None) is not None and len(prepared.all_dates) > 0:
                    loaded_start = pd.Timestamp(prepared.all_dates[0]).date().isoformat()
                    loaded_end = pd.Timestamp(prepared.all_dates[-1]).date().isoformat()
                    st.caption(f"Loaded data range: {loaded_start} to {loaded_end}")

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
                            start_date=bt_start_date,
                            global_data=global_data,
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
                # --- METRICS FIX: Use closed-trade diagnostics only ---
                avg_trade_pct_display = 0.0
                expectancy_dollar_display = 0.0
                profit_factor_display = None
                trade_count_display = int(res.get("total_trades", 0) or 0)
                win_rate_display = float(res.get("hit_rate", 0.0) or 0.0)
                trades = res.get("trades_list", [])
                if trades:
                    df_trades = pd.DataFrame(trades)
                    if "Reason" in df_trades.columns:
                        df_trades = df_trades[df_trades["Reason"] != "PYRAMID_ADD"]
                    # Unweighted trade return %
                    if "Return %" in df_trades.columns:
                        ret_series = pd.to_numeric(df_trades["Return %"], errors='coerce').dropna()
                        if not ret_series.empty:
                            avg_trade_pct_display = float(ret_series.mean())
                    # PnL-based expectancy + profit factor + win rate
                    if "PnL" in df_trades.columns:
                        pnl_series = pd.to_numeric(df_trades["PnL"], errors="coerce").dropna()
                        if not pnl_series.empty:
                            trade_count_display = int(len(pnl_series))
                            win_rate_display = float((pnl_series > 0).mean() * 100.0)
                            expectancy_dollar_display = float(pnl_series.mean())
                            gross_profit = float(pnl_series[pnl_series > 0].sum())
                            gross_loss = float(-pnl_series[pnl_series < 0].sum())
                            if gross_loss > 0:
                                profit_factor_display = gross_profit / gross_loss
                            elif gross_profit > 0:
                                profit_factor_display = float("inf")

                profit_factor_label = "n/a"
                if profit_factor_display == float("inf"):
                    profit_factor_label = "∞"
                elif profit_factor_display is not None and pd.notna(profit_factor_display):
                    profit_factor_label = f"{profit_factor_display:.2f}"

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("CAGR", f"{res.get('cagr', 0):.1%}")
                col2.metric("Win Rate", f"{win_rate_display:.1f}%")
                col3.metric("Profit Factor", profit_factor_label)
                col4.metric("Expectancy ($)", f"${expectancy_dollar_display:,.2f}")
                col5.metric("Total Trades", trade_count_display)
                st.caption(f"Avg Trade % (unweighted): {avg_trade_pct_display:.2f}%")
                # -------------------------------------------------

                # --- CHART CRASH FIX: Normalize Date Types ---
                ec_data = res.get("equity_curve", [])
                df_ec = normalize_equity_curve_df(ec_data)
                if not df_ec.empty:
                    st.line_chart(df_ec.set_index("Date")["Equity"])
                elif ec_data:
                    st.warning("Equity data malformed.")
                # ---------------------------------------------
                
                # --- DOWNLOAD BUTTON RESTORED ---
                # Safe data extraction
                equity_df = df_ec.copy()
                # Fallback chain to ensure we never get "Unknown" if the key exists
                strategy_name = res.get("strategy_name") or res.get("strategy") or name or "Backtest_Result"
                # Sanitize filename (remove special chars)
                safe_name = "".join(
                    [c for c in strategy_name if c.isalnum() or c in (" ", "_", "-")]
                ).strip()

                if not equity_df.empty:
                    try:
                        csv_data = equity_df.to_csv().encode('utf-8')
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
