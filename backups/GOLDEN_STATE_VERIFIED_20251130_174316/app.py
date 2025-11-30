import streamlit as st
import pandas as pd
import sys
import os
import json
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(__file__))
from data.schwab_client import sd
from data.indices import get_index_symbols
from data.loader import fetch_single_symbol, fetch_data_pack
from execution.engine import run_backtest, run_compare, _compute_indicators
from strategies.generic import GenericStrategy
from simulation.paper_trader import PaperTrader

st.set_page_config(page_title="Apex Sniper", layout="wide", page_icon="🎯")
st.sidebar.title("🎯 Apex Sniper")

# --- API Status ---
with st.sidebar.expander("📡 API Status", expanded=False):
    if st.button("Test Connection"):
        res = sd.health_check("VOO")
        if res.get("ok"): st.success("Online")
        else: st.error("Offline")

mode = st.sidebar.radio("Mode", ["Live Screener", "Backtest", "Simulator"])

@st.cache_data(ttl=3600)
def get_global_data(days=400):
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days)
    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
    return {"SPY": g_data.get("SPY"), "VIX": vix}

def _safe_vix(vix_df, target_index):
    if vix_df is None or "close" not in vix_df:
        return pd.Series(20.0, index=target_index)
    cleaned = vix_df.copy()
    if isinstance(cleaned.index, pd.DatetimeIndex) and cleaned.index.tz is not None:
        cleaned.index = cleaned.index.tz_localize(None)
    return cleaned["close"].reindex(target_index, method="ffill").fillna(20.0)

def calculate_quality_score(row, strategy_name):
    score = 50.0 
    adx = row.get("adx", 20)
    if adx > 25: score += 5
    
    if "Sniper" in strategy_name or "VIX" in strategy_name:
        rs_trend = row.get("rs_trend", 0)
        score += (rs_trend * 100.0) 
        rsi2 = row.get("rsi2", 50)
        if rsi2 < 5: score += 20
        elif rsi2 < 10: score += 10
    else: 
        rsi2 = row.get("rsi2", 50)
        score += (100 - rsi2) * 2.0
        vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)
        if vol_rel > 1.5: score += 10
        
    return max(0.0, min(100.0, score))

# --- 1. Live Screener ---
if mode == "Live Screener":
    st.header("🔭 Live Market Screener")
    col1, col2, col3 = st.columns(3)
    with col1: universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=1)
    with col2:
        strategies = {}
        try:
            with open("config/generated_strategies.json", "r") as f:
                for s in json.load(f): strategies[s["name"]] = s
        except: st.error("No Strategies Found!")
        all_names = list(strategies.keys())
        selected_names = st.multiselect("Strategies", all_names, default=all_names)
    with col3:
        top_n = st.slider("Show Top N Results", min_value=5, max_value=100, value=20)

    if st.button("Run Scan"):
        symbols = get_index_symbols(universe)
        global_data = get_global_data(days=1260)
        spy_df = global_data.get("SPY")
        vix_df = global_data.get("VIX")
        
        progress = st.progress(0, text="Starting Scan...")
        status_text = st.empty()
        all_hits = []
        
        for i, sym in enumerate(symbols):
            if i % 10 == 0: 
                progress.progress(i / len(symbols))
                status_text.text(f"Scanning {sym}...")
            
            df = fetch_single_symbol(sym, days=400, force_fresh=True)
            if df is None or len(df) < 200: continue 
            
            try:
                df = _compute_indicators(df, spy_df=spy_df)
                df["vix"] = _safe_vix(vix_df, df.index)
                
                signal_idx = len(df) - 1
                price_row = df.iloc[signal_idx]
                
                for strat_name in selected_names:
                    strat = GenericStrategy(strategies[strat_name])
                    signal = strat.entry(df, signal_idx)
                    
                    if signal:
                        quality = calculate_quality_score(price_row, strat_name)
                        stop = signal.get("stop_price", price_row["close"] * 0.95)
                        risk_pct = (price_row["close"] - stop) / price_row["close"]
                        target = price_row["close"] * (1 + (risk_pct * 2.0))
                        
                        all_hits.append({
                            "Score": round(quality, 1),
                            "Strategy": strat_name,
                            "Symbol": sym,
                            "Price": price_row['close'],
                            "Stop": stop,
                            "Target": target,
                            "Risk %": round(risk_pct * 100, 2),
                            "RS_Trend": round(price_row.get('rs_trend', 0), 4),
                            "RSI2": round(price_row.get('rsi2', 50), 1)
                        })
            except Exception as e: pass

        progress.progress(100)
        status_text.text("Scan Complete!")
        
        if all_hits:
            df_results = pd.DataFrame(all_hits)
            # TIE BREAKER SORTING: Score DESC, then RSI2 ASC
            df_results = df_results.sort_values(by=["Score", "RSI2"], ascending=[False, True])
            
            top_results = df_results.head(top_n)
            
            st.success(f"Found {len(all_hits)} total setups. Showing Top {len(top_results)}.")
            st.dataframe(
                top_results.style.format({
                    "Price": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "Score": "{:.1f}"
                }).background_gradient(subset=["Score"], cmap="Greens")
            )
            csv = top_results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Top Setups CSV", csv, "apex_top_setups.csv", "text/csv")
        else:
            st.warning("No setups found.")

# --- 2. Backtest ---
elif mode == "Backtest":
    st.header("🧪 Backtest Engine")
    col1, col2, col3 = st.columns(3)
    with col1: bt_universe = st.selectbox("Universe", ["S&P 100", "S&P 500", "S&P 1500"], index=1)
    with col2:
        gen_strategies = []
        try:
            with open("config/generated_strategies.json", "r") as f:
                gen_strategies = [s["name"] for s in json.load(f)]
        except: pass
        bt_strategies = st.multiselect("Strategies", gen_strategies, default=gen_strategies)
    with col3: timeframe = st.selectbox("Timeframe", ["1 Year", "5 Years", "Max"], index=1)

    if st.button("Run Backtest"):
        days_map = {"1 Year": 365, "5 Years": 1260, "Max": 10000}
        start_date = (datetime.now(timezone.utc) - timedelta(days=days_map.get(timeframe, 1260))).date()
        
        with st.status("Loading Data...") as status:
            symbols = get_index_symbols(bt_universe)
            data_map = fetch_data_pack(symbols, days=days_map.get(timeframe, 1260))
            g_data = get_global_data(days=days_map.get(timeframe, 1260))
            vix = g_data.get("$VIX") or g_data.get("VIX")
            global_context = {"SPY": g_data.get("SPY"), "VIX": vix}
            
            status.update(label=f"Backtesting {len(data_map)} symbols...", state="running")
            results = run_compare(bt_strategies, data_map, symbols, start_cash=100000.0, start_date=start_date, global_data=global_context)
            status.update(label="Complete!", state="complete")
        
        if not results.empty:
            st.dataframe(results.style.format({
                "hit_rate": "{:.1f}%", "avg_profit_pct": "{:.2f}%", "cagr": "{:.1%}", 
                "Score": "{:.1f}", "beta": "{:.2f}", "sortino": "{:.2f}",
                "exposure_pct": "{:.1f}%", "calmar": "{:.2f}"
            }))
        else:
            st.error("No trades generated.")

# --- 3. Simulator ---
elif mode == "Simulator":
    st.header("🎰 Paper Trading Simulator")
    
    trader = PaperTrader()
    state = trader.state
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl:,.2f}", delta=f"{pnl/100000*100:.2f}%")
    m4.metric("Open Positions", len(state['positions']))

    col_main, col_side = st.columns([3, 1])
    
    with col_side:
        st.markdown("### Actions")
        if st.button("🔄 Run Daily Cycle", type="primary"):
            with st.spinner("Updating Market Data..."):
                trader.update_valuations()
                
                strategies = {}
                try:
                    with open("config/generated_strategies.json", "r") as f:
                        for s in json.load(f): strategies[s["name"]] = s
                except: pass
                
                exits = trader.process_exits(strategies)
                for e in exits: st.toast(e, icon="💰")
                
                symbols = get_index_symbols("S&P 500")
                g_data = get_global_data(days=400)
                spy_df = g_data.get("SPY")
                vix_df = g_data.get("VIX")
                
                candidates = []
                progress = st.progress(0, text="Scanning for Entries...")
                
                for i, sym in enumerate(symbols):
                    if i % 20 == 0: progress.progress(i/len(symbols))
                    df = fetch_single_symbol(sym, days=400)
                    if df is None or len(df) < 200: continue
                    
                    try:
                        df = _compute_indicators(df, spy_df=spy_df)
                        df["vix"] = _safe_vix(vix_df, df.index)
                        
                        idx = len(df) - 1
                        price_row = df.iloc[idx]
                        
                        for strat_name, strat_config in strategies.items():
                            strat = GenericStrategy(strat_config)
                            signal = strat.entry(df, idx)
                            if signal:
                                score = calculate_quality_score(price_row, strat_name)
                                candidates.append({
                                    "Symbol": sym, "Strategy": strat_name, 
                                    "Price": price_row["close"], 
                                    "Stop": signal.get("stop_price", price_row["close"]*0.95),
                                    "Score": score
                                })
                    except: pass
                
                progress.empty()
                entries = trader.execute_entries(candidates)
                for e in entries: st.toast(e, icon="🚀")
            
            st.success("Daily Cycle Complete!")
            st.rerun()

        if st.button("⚠️ Reset Account"):
            trader.reset_account()
            st.rerun()

    with col_main:
        st.subheader("📂 Current Holdings")
        if state['positions']:
            pos_data = []
            for sym, p in state['positions'].items():
                pos_data.append({
                    "Symbol": sym, "Shares": p['shares'], "Entry": f"${p['entry_price']:.2f}",
                    "Current": f"${p.get('current_price', 0):.2f}",
                    "PnL": f"${p.get('unrealized_pnl', 0):.2f}",
                    "Return": f"{p.get('unrealized_pct', 0):.2f}%",
                    "Strategy": p['strategy']
                })
            st.dataframe(pd.DataFrame(pos_data))
        else:
            st.info("No open positions. Cash is king.")
            
        st.subheader("📜 Trade History")
        if state['history']:
            hist_df = pd.DataFrame(state['history'])
            st.dataframe(hist_df.sort_values("exit_date", ascending=False))
        else:
            st.caption("No closed trades yet.")
