"""Upgrade app.py with a richer Streamlit dashboard focused on Apex Duo."""

import os


APP_CODE = """import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.indices import get_index_symbols
from data.loader import fetch_data_pack
from execution.engine import _compute_indicators, calculate_backtest_quality_score
from strategies.generic import GenericStrategy

CONFIG_PATH = "config/generated_strategies.json"

st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")


def load_strategies():
    try:
        with open(CONFIG_PATH, "r") as fh:
            return json.load(fh)
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"Could not load strategies: {exc}")
        return []


def profit_target_multiple(strategy):
    for rule in strategy.get("exit_rules", []):
        if rule.get("type") == "profit_target":
            try:
                return float(rule.get("val"))
            except Exception:
                return None
    return None


def compute_trade_plan(row, strategy):
    entry_price = float(row.get("close", 0.0))
    atr = float(row.get("atr14", entry_price * 0.02))
    stop_mult = float(strategy.get("stop_loss_atr", 0) or 0)
    stop_price = entry_price - (atr * stop_mult) if stop_mult else entry_price * 0.95

    target_mult = profit_target_multiple(strategy)
    target_price = entry_price * target_mult if target_mult else None

    risk = entry_price - stop_price
    reward = (target_price - entry_price) if target_price else None
    rr_text = "Open" if reward is None or risk <= 0 else f"1:{(reward / risk):.1f}"

    return {
        "entry": entry_price,
        "atr": atr,
        "stop": stop_price,
        "target": target_price,
        "rr": rr_text,
        "target_mult": target_mult,
        "stop_mult": stop_mult,
    }


def scan_market(strategies, universe="S&P 1500"):
    symbols = get_index_symbols(universe)
    if not symbols:
        st.warning(f"No symbols found for {universe}.")
        return pd.DataFrame()

    st.info(f"Scanning {len(symbols)} symbols in {universe}...")
    data_pack = fetch_data_pack(symbols, days=420)

    results = []
    progress = st.progress(0.0)
    status = st.empty()

    for idx, sym in enumerate(symbols):
        df = data_pack.get(sym)
        if df is None or df.empty:
            continue

        try:
            enriched = _compute_indicators(df.copy())
            if enriched is None or enriched.empty:
                continue

            price_row = enriched.iloc[-1]
            last_idx = len(enriched) - 1

            for genome in strategies:
                strat = GenericStrategy(genome)
                signal = strat.entry(enriched, last_idx)
                if not signal:
                    continue

                plan = compute_trade_plan(price_row, genome)
                signal_date = price_row.name.strftime("%Y-%m-%d") if hasattr(price_row, "name") else ""
                score = calculate_backtest_quality_score(price_row, genome.get("name", ""))

                results.append(
                    {
                        "Symbol": sym,
                        "Strategy": genome.get("name", "Strategy"),
                        "Close Price": plan["entry"],
                        "ATR": plan["atr"],
                        "Stop Loss ($)": plan["stop"],
                        "Profit Target ($)": plan["target"],
                        "Risk/Reward": plan["rr"],
                        "Action": "Buy MOO (tomorrow)",
                        "Signal Date": signal_date,
                        "Stop Mult (ATR)": plan["stop_mult"],
                        "Target Multiple": plan["target_mult"],
                        "Time Stop (Days)": genome.get("time_stop"),
                        "Score": score,
                    }
                )
        except Exception:
            continue

        if (idx + 1) % 10 == 0 or idx == len(symbols) - 1:
            progress.progress((idx + 1) / len(symbols))
            status.text(f"Processed {idx + 1}/{len(symbols)}: {sym}")

    progress.empty()
    status.empty()

    df_results = pd.DataFrame(results)
    if df_results.empty:
        return df_results

    ordered_cols = [
        "Symbol",
        "Strategy",
        "Close Price",
        "ATR",
        "Stop Loss ($)",
        "Profit Target ($)",
        "Risk/Reward",
        "Action",
        "Signal Date",
        "Stop Mult (ATR)",
        "Target Multiple",
        "Time Stop (Days)",
        "Score",
    ]
    df_results = df_results[ordered_cols]
    return df_results.sort_values(by=["Score", "Symbol"], ascending=[False, True]).reset_index(drop=True)


def sidebar_playbook():
    with st.sidebar:
        st.title("Apex Duo")
        st.caption("Unified Velocity Engine")

        st.markdown("### Apex Execution Protocol")
        st.markdown(
            "- Scan after market close (4:00 PM ET)\\n"
            "- Place Market on Open orders for next session\\n"
            "- Priority: Super Signals > Gen 9 Income > Gen 12 Growth"
        )

        st.markdown("### Strategy Rules")
        st.markdown("**The Wealth Builder (Gen 12 Classic)**")
        st.markdown(
            "- Entry: Market on Open\\n"
            "- Stop: Entry - 4.4 x ATR\\n"
            "- Target: None (let winners run)\\n"
            "- Time Stop: 71 days"
        )
        st.markdown("**The Income Generator (Gen 9 Evolved)**")
        st.markdown(
            "- Entry: Market on Open\\n"
            "- Stop: Entry - 5.1 x ATR\\n"
            "- Target: Entry + 8%\\n"
            "- Time Stop: 45 days"
        )


def render_table(df_results: pd.DataFrame):
    if df_results.empty:
        st.warning("No setups found.")
        return

    dupes = df_results[df_results.duplicated(subset=["Symbol"], keep=False)]
    if not dupes.empty:
        st.success(f"Super Signals: {dupes['Symbol'].nunique()} symbols in both lists.")
        st.dataframe(
            dupes.style.format({"Close Price": "${:.2f}", "ATR": "{:.2f}", "Stop Loss ($)": "${:.2f}"}),
            use_container_width=True,
        )
        st.markdown("---")

    display_cols = df_results.columns.tolist()
    st.subheader(f"All Setups ({len(df_results)})")
    st.dataframe(
        df_results.style.format(
            {
                "Close Price": "${:.2f}",
                "ATR": "{:.2f}",
                "Stop Loss ($)": "${:.2f}",
                "Profit Target ($)": lambda x: "None" if pd.isna(x) else f"${x:.2f}",
                "Score": "{:.1f}",
            }
        ),
        use_container_width=True,
        height=620,
    )

    csv = df_results[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV (matches table)",
        csv,
        file_name=f"apex_duo_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main():
    sidebar_playbook()
    st.title("Apex Duo Trade Console")
    st.caption("Live screener with persistent state and precise trade plans.")

    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None
        st.session_state.last_run = None

    strategies = load_strategies()
    if not strategies:
        st.stop()

    st.markdown("#### Live Screener")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        universe = st.selectbox("Universe", ["S&P 500", "S&P 1500", "Nasdaq 100"], index=1)
    with col2:
        run_scan = st.button("Run Scan", type="primary", use_container_width=True)
    with col3:
        clear = st.button("Clear Results", use_container_width=True)

    if clear:
        st.session_state.scan_results = None
        st.session_state.last_run = None

    if run_scan:
        with st.spinner("Scanning market..."):
            results = scan_market(strategies, universe)
        st.session_state.scan_results = results
        st.session_state.last_run = {
            "universe": universe,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    if st.session_state.scan_results is not None:
        meta = st.session_state.last_run or {}
        if meta:
            st.info(
                f"Last run: {meta.get('time', 'n/a')} · Universe: {meta.get('universe', 'n/a')} "
                "(results persist until cleared)"
            )
        render_table(st.session_state.scan_results)
    else:
        st.info("Run a scan to populate the trade sheet. Results will persist until you clear them.")


if __name__ == "__main__":
    main()
"""


def upgrade_dashboard_ui():
    print("Upgrading dashboard UI (persistence + trading plans)...")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(project_root, "app.py")
    with open(app_path, "w") as fh:
        fh.write(APP_CODE)
    print(f"Dashboard updated at {app_path}")


if __name__ == "__main__":
    upgrade_dashboard_ui()
