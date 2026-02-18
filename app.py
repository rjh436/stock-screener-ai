import streamlit as st
import pandas as pd
import sys
import os
import json
import hashlib
import math
import time
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo
import gc

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.schwab_client import sd
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from data.universe import (
    get_universe_symbols,
    get_universe_symbols_pit_with_meta,
    get_universe_symbols_pit_window_with_meta,
    build_russell3000_membership_by_day,
    get_russell3000_pit_status,
)
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
BASELINE_CONFIG_PATH = os.path.join("config", "backtest_baselines.json")
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


def _is_market_open_et(now: datetime | None = None) -> bool:
    if now is None:
        now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def _env_flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default) or default).strip().lower() in {"1", "true", "yes", "on"}


def _get_cache_cap() -> int:
    try:
        cap = int(os.getenv("APEX_BACKTEST_CACHE_MAX_ENTRIES", "1") or "1")
    except Exception:
        cap = 1
    return max(1, min(cap, 6))


def _recommended_fetch_workers(symbol_count: int, *, cache_only: bool = False) -> int:
    env_override = str(os.getenv("DATA_FETCH_WORKERS", "") or "").strip()
    if env_override:
        try:
            return max(1, min(int(env_override), 32))
        except Exception:
            pass
    cpu = os.cpu_count() or 8
    if symbol_count >= 2000:
        workers = min(24, max(8, cpu * (2 if cache_only else 1)))
    elif symbol_count >= 800:
        workers = min(18, max(8, int(cpu * (1.5 if cache_only else 1.0))))
    elif symbol_count >= 250:
        workers = min(14, max(6, cpu if cache_only else max(4, cpu // 2)))
    else:
        workers = min(10, max(4, cpu // 2))
    return max(1, min(int(workers), 32))


def _init_backtest_cache() -> tuple[dict, list]:
    if "backtest_cache" not in st.session_state:
        st.session_state.backtest_cache = {}
    if "backtest_cache_order" not in st.session_state:
        st.session_state.backtest_cache_order = []
    cache = st.session_state.backtest_cache
    order = [k for k in st.session_state.backtest_cache_order if k in cache]
    st.session_state.backtest_cache_order = order
    return cache, order


def _touch_cache_key(key: str) -> None:
    cache, order = _init_backtest_cache()
    if key in cache:
        if key in order:
            order.remove(key)
        order.append(key)
    st.session_state.backtest_cache_order = order


def _set_backtest_cache(key: str, payload: dict) -> None:
    cache, order = _init_backtest_cache()
    cache[key] = payload
    if key in order:
        order.remove(key)
    order.append(key)
    max_entries = _get_cache_cap()
    while len(order) > max_entries:
        evict_key = order.pop(0)
        cache.pop(evict_key, None)
    st.session_state.backtest_cache = cache
    st.session_state.backtest_cache_order = order


def _drop_backtest_cache(key: str) -> None:
    cache, order = _init_backtest_cache()
    cache.pop(key, None)
    order = [k for k in order if k != key]
    st.session_state.backtest_cache = cache
    st.session_state.backtest_cache_order = order


def _strategy_fingerprint(strategies: List[dict]) -> str:
    """
    Stable short hash of selected strategy payloads.
    Ensures session cache is invalidated when strategy params change.
    """
    try:
        payload = json.dumps(strategies or [], sort_keys=True, separators=(",", ":"))
    except Exception:
        payload = str(strategies or [])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _load_backtest_baselines() -> dict:
    if not os.path.exists(BASELINE_CONFIG_PATH):
        return {}
    try:
        with open(BASELINE_CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _save_backtest_baselines(payload: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE_CONFIG_PATH), exist_ok=True)
    tmp_path = f"{BASELINE_CONFIG_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, BASELINE_CONFIG_PATH)


def _baseline_key(
    *,
    universe: str,
    duration: str,
    start_date: str,
    strategy_name: str,
    strategy_fingerprint: str,
) -> str:
    parts = [
        str(universe or "").upper(),
        str(duration or ""),
        str(start_date or "max"),
        str(strategy_name or ""),
        str(strategy_fingerprint or ""),
    ]
    return "|".join(parts)


def _safe_float(val, default: float = 0.0) -> float:
    try:
        out = float(val)
    except Exception:
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _trade_diagnostics(res: dict) -> dict:
    avg_trade_pct_display = 0.0
    expectancy_dollar_display = 0.0
    profit_factor_display = None
    trade_count_display = int(res.get("total_trades", 0) or 0)
    win_rate_display = float(res.get("hit_rate", 0.0) or 0.0)

    trades = res.get("trades_list", [])
    if not trades:
        return {
            "avg_trade_pct": avg_trade_pct_display,
            "expectancy_dollar": expectancy_dollar_display,
            "profit_factor": profit_factor_display,
            "trade_count": trade_count_display,
            "win_rate_pct": win_rate_display,
        }

    df_trades = pd.DataFrame(trades)
    if "Reason" in df_trades.columns:
        df_trades = df_trades[df_trades["Reason"] != "PYRAMID_ADD"]

    if "Return %" in df_trades.columns:
        ret_series = pd.to_numeric(df_trades["Return %"], errors='coerce').dropna()
        if not ret_series.empty:
            avg_trade_pct_display = float(ret_series.mean())

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

    return {
        "avg_trade_pct": float(avg_trade_pct_display),
        "expectancy_dollar": float(expectancy_dollar_display),
        "profit_factor": profit_factor_display,
        "trade_count": int(trade_count_display),
        "win_rate_pct": float(win_rate_display),
    }


def _accuracy_mode() -> str:
    """
    Accuracy policy:
    - block: stop runs that fail coverage threshold
    - warn: allow run but flag results as lower-confidence
    - off: no coverage enforcement
    """
    raw = str(os.getenv("APEX_BACKTEST_ACCURACY_MODE", "warn") or "warn").strip().lower()
    if raw in {"block", "strict", "hard", "enforce"}:
        return "block"
    if raw in {"off", "none", "disable", "disabled"}:
        return "off"
    # Default mode keeps backtesting available while still surfacing accuracy risk.
    return "warn"


def _effective_accuracy_mode(*, verified_run: bool) -> str:
    return "block" if verified_run else _accuracy_mode()


def _min_coverage_threshold(universe_name: str, *, accuracy_mode: str | None = None) -> float:
    u = str(universe_name or "").upper()
    mode = str(accuracy_mode or _accuracy_mode()).strip().lower()
    if u in {"RUSSELL3000", "RUSSELL 3000"}:
        default = "0.75" if mode == "block" else "0.70"
        raw = os.getenv("APEX_BACKTEST_MIN_COVERAGE_RUSSELL", default)
    else:
        default = "0.80" if mode == "block" else "0.65"
        raw = os.getenv("APEX_BACKTEST_MIN_COVERAGE", default)
    try:
        val = float(raw or 0.0)
    except Exception:
        val = float(default)
    return max(0.10, min(val, 1.00))


def _recent_data_coverage(
    prepared,
    *,
    expected_symbol_count: int,
    max_lag_days: int,
    symbol_scope: Optional[set[str]] = None,
) -> tuple[int, int, float]:
    enriched = getattr(prepared, "enriched", {}) or {}
    if not enriched:
        return 0, int(max(0, expected_symbol_count)), 0.0
    now_et = datetime.now(ZoneInfo("America/New_York")).date()
    fresh = 0
    if symbol_scope:
        scope = {str(s).strip().upper() for s in symbol_scope if str(s).strip()}
        by_upper = {str(sym).upper(): sym_data for sym, sym_data in enriched.items()}
        for sym_u in scope:
            sym_data = by_upper.get(sym_u)
            if sym_data is None:
                continue
            idx = getattr(sym_data, "index", None)
            if idx is None or len(idx) == 0:
                continue
            try:
                last_dt = pd.Timestamp(idx[-1]).tz_localize(None).date()
            except Exception:
                continue
            if (now_et - last_dt).days <= max_lag_days:
                fresh += 1
        total = int(len(scope))
    else:
        for sym_data in enriched.values():
            idx = getattr(sym_data, "index", None)
            if idx is None or len(idx) == 0:
                continue
            try:
                last_dt = pd.Timestamp(idx[-1]).tz_localize(None).date()
            except Exception:
                continue
            if (now_et - last_dt).days <= max_lag_days:
                fresh += 1
        total = int(max(0, expected_symbol_count))
    cov = (fresh / float(max(1, total))) if total > 0 else 0.0
    return int(fresh), total, float(cov)


def _write_missing_symbols_report(
    *,
    universe_name: str,
    start_date: str,
    universe_source: str,
    expected_symbols: List[str],
    tested_symbols: List[str],
) -> str:
    os.makedirs("exports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_uni = "".join(ch for ch in str(universe_name).upper() if ch.isalnum()) or "UNIVERSE"
    path = os.path.join("exports", f"{ts}_missing_symbols_{safe_uni}.txt")
    exp = {str(s).upper() for s in (expected_symbols or []) if str(s).strip()}
    tst = {str(s).upper() for s in (tested_symbols or []) if str(s).strip()}
    missing = sorted(exp - tst)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Universe: {universe_name}\n")
        f.write(f"Universe Source: {universe_source}\n")
        f.write(f"Start Date: {start_date}\n")
        f.write(f"Expected Symbols: {len(exp)}\n")
        f.write(f"Tested Symbols: {len(tst)}\n")
        f.write(f"Missing Symbols: {len(missing)}\n\n")
        f.write("# Missing symbols\n")
        for sym in missing:
            f.write(f"{sym}\n")
    return path

# --- SIDEBAR ---
with st.sidebar:
    st.title("🎯 Apex Sniper")
    st.caption("Institutional Grade Algo System")
    st.markdown("---")
    st.success("🏆 APEX V9 MEDALLION: RAW ALPHA ACTIVE")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])

    st.markdown("### ✅ Accuracy")
    pit_as_of = pd.Timestamp.utcnow().tz_localize(None).date().isoformat()
    pit_status = get_russell3000_pit_status(pit_as_of)
    require_pit_sidebar = _env_flag("APEX_REQUIRE_PIT_UNIVERSE", "1")
    pit_source = str(pit_status.get("source", "unavailable") or "unavailable")
    pit_count = int(pit_status.get("symbol_count", 0) or 0)
    if pit_source in {"pit_snapshot", "pit_ranges"} and pit_count > 0:
        st.success(f"PIT universe ready ({pit_source}, {pit_count} symbols).")
    elif require_pit_sidebar:
        st.error("PIT universe missing. Russell 3000 backtests will be blocked for accuracy.")
    else:
        st.warning("PIT universe missing. Russell 3000 backtests may be survivorship-biased.")
    with st.expander("PIT Setup"):
        st.caption(f"`RUSSELL3000_PIT_DIR`: {pit_status.get('pit_dir', '')}")
        st.caption(f"`RUSSELL3000_PIT_MEMBERSHIP_CSV`: {pit_status.get('range_csv', '') or '(not set)'}")
        st.caption("Template: `data/russell3000_membership/template_membership_ranges.csv`")
        st.caption("Strict mode: `APEX_REQUIRE_PIT_UNIVERSE=1`")
        st.caption("Accuracy mode: `APEX_BACKTEST_ACCURACY_MODE=warn|block|off`")
        st.caption("Russell min coverage: `APEX_BACKTEST_MIN_COVERAGE_RUSSELL` (default 0.75 in block mode, 0.70 otherwise)")
        st.caption("Run validator: `./.venv/bin/python tools/validate_pit_universe.py --strict`")
    
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

        status_msg.info(f"📦 Resolving {universe} constituents...")
        progress_bar.progress(0.2, text="20% Complete")
        if universe in ("Russell 3000", "RUSSELL3000"):
            live_as_of = datetime.utcnow().date().isoformat()
            symbols, live_universe_source = get_universe_symbols_pit_with_meta("RUSSELL3000", live_as_of)
        else:
            symbols = get_universe_symbols(universe)
            live_universe_source = "current_index"
        symbols = list(symbols or [])
        status_msg.info(
            f"📦 Loaded {len(symbols):,} symbols for {universe} "
            f"(source: {live_universe_source})."
        )
        if not symbols:
            st.error(f"No symbols loaded for {universe}.")
            progress_bar.empty()
            status_msg.empty()
            timer_msg.empty()
            st.stop()
        base_days = 400
        data = fetch_data_pack(
            symbols,
            days=base_days,
            max_workers=_recommended_fetch_workers(len(symbols), cache_only=False),
            force_fresh=True,
            inject_live=True,
            max_lag_days=0,
        )
        g_data = fetch_data_pack(
            ["SPY", "$VIX", "VIX"],
            days=600,
            max_workers=_recommended_fetch_workers(3, cache_only=False),
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
    bt_end_date = today.date().isoformat()

    # Pull enough calendar history for the requested window + warmup bars.
    if bt_start_ts is not None:
        fetch_days = max(int((today - bt_start_ts).days), days) + 320
    else:
        fetch_days = days + 320

    strategy_fp = _strategy_fingerprint(selected_strategies)
    # Versioned key avoids reusing old payloads from prior app sessions.
    # Include strategy fingerprint so param edits cannot silently reuse stale prepared data.
    cache_key = ""
    _init_backtest_cache()

    if bt_start_date:
        st.info(f"Settings: **{bt_universe}** for **{bt_duration}** (from **{bt_start_date}**)")
    else:
        st.info(f"Settings: **{bt_universe}** for **{bt_duration}**")
    st.caption(f"Strategy fingerprint: `{strategy_fp}`")
    if "bt_verified_run" not in st.session_state:
        st.session_state.bt_verified_run = False
    verified_run = st.toggle(
        "Verified run (strict accuracy gate)",
        value=bool(st.session_state.bt_verified_run),
        help=(
            "When enabled, this run uses strict accuracy rules (higher minimum coverage, "
            "full-lookback enforcement, and lock gating)."
        ),
    )
    st.session_state.bt_verified_run = bool(verified_run)
    run_accuracy_mode = _effective_accuracy_mode(verified_run=bool(verified_run))
    cache_mode_token = "strict" if run_accuracy_mode == "block" else run_accuracy_mode
    cache_key = (
        f"btv6|{bt_universe}|{bt_duration}|{bt_start_date or 'max'}|"
        f"{strategy_fp}|{cache_mode_token}"
    )
    strict_cov_hint = _min_coverage_threshold(bt_universe, accuracy_mode=run_accuracy_mode)
    st.caption(
        f"Run accuracy mode: `{run_accuracy_mode}` | "
        f"Min coverage required for this run: {strict_cov_hint:.0%}"
    )

    if st.button("🚀 RUN BACKTEST", type="primary"):
        if not selected_strategies:
            st.error("Please select at least one strategy.")
        else:
            cache, _ = _init_backtest_cache()
            prepared = None
            global_data = {}
            expected_symbol_count = 0
            expected_symbols: List[str] = []
            universe_source = "unknown"
            require_pit_universe = _env_flag("APEX_REQUIRE_PIT_UNIVERSE", "1")
            cached_payload = cache.get(cache_key)
            if isinstance(cached_payload, dict):
                prepared = cached_payload.get("prepared")
                global_data = cached_payload.get("global_data") or {}
                expected_symbol_count = int(cached_payload.get("symbol_count", 0) or 0)
                expected_symbols = list(cached_payload.get("symbol_universe") or [])
                universe_source = str(cached_payload.get("universe_source", "unknown") or "unknown")
                if expected_symbol_count <= 0 and expected_symbols:
                    expected_symbol_count = len(expected_symbols)
                if (
                    bt_universe in ("Russell 3000", "RUSSELL3000")
                    and require_pit_universe
                    and universe_source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}
                ):
                    st.warning(
                        "Discarding cached Russell 3000 dataset because PIT provenance is missing. "
                        "Rebuilding with strict accuracy rules."
                    )
                    prepared = None
                    global_data = {}
                    expected_symbol_count = 0
            else:
                prepared = cached_payload
            cache_hit = prepared is not None
            if cache_hit:
                st.success("⚡ Using Cached Data (Instant Mode Active)")
                _touch_cache_key(cache_key)

            with st.spinner("Simulating..."):
                stage_msg = st.empty()
                stage_msg.caption("Stage: Initializing backtest run...")
                quality_union = {
                    "requested": set(),
                    "loaded": set(),
                    "missing": set(),
                    "incomplete": set(),
                    "stale": set(),
                }

                def _merge_quality(rep: dict | None) -> None:
                    if not rep:
                        return
                    quality_union["requested"].update(rep.get("requested_symbols") or [])
                    quality_union["loaded"].update(rep.get("loaded_symbols") or [])
                    quality_union["missing"].update(rep.get("missing_symbols") or [])
                    quality_union["incomplete"].update(rep.get("incomplete_symbols") or [])
                    quality_union["stale"].update(rep.get("stale_symbols") or [])

                strict_full_lookback = False
                if not cache_hit:
                    stage_msg.caption("Stage: Resolving universe membership...")
                    if bt_universe in ("Russell 3000", "RUSSELL3000"):
                        symbols, universe_source = get_universe_symbols_pit_window_with_meta(
                            "RUSSELL3000",
                            bt_start_date,
                            bt_end_date,
                        )
                        if universe_source == "fallback_current":
                            msg = (
                                "Point-in-time Russell 3000 membership data was not found. "
                                "Using current constituents introduces survivorship bias."
                            )
                            if require_pit_universe:
                                st.error(
                                    f"{msg} Set `RUSSELL3000_PIT_DIR` or "
                                    "`RUSSELL3000_PIT_MEMBERSHIP_CSV` to run accurate long-horizon backtests."
                                )
                                st.session_state.backtest_results = {}
                                symbols = []
                            else:
                                st.warning(msg)
                    else:
                        symbols = get_index_symbols(bt_universe)
                        universe_source = "current_index"
                    expected_symbols = list(symbols or [])
                    expected_symbol_count = len(expected_symbols)
                    if not symbols:
                        st.error(f"No symbols loaded for universe: {bt_universe}")
                        st.session_state.backtest_results = {}
                        symbols = []

                    ny_now = datetime.now(ZoneInfo("America/New_York"))
                    market_open = _is_market_open_et(ny_now)
                    prefer_cache_first = _env_flag("APEX_BACKTEST_CACHE_FIRST", "1")
                    prefer_cache_only_when_closed = str(
                        os.getenv("APEX_BACKTEST_CACHE_ONLY_WHEN_CLOSED", "1") or "1"
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    refresh_when_closed = _env_flag("APEX_BACKTEST_REFRESH_WHEN_CLOSED", "0")
                    cache_only_first = bool(
                        prefer_cache_first or ((not market_open) and prefer_cache_only_when_closed)
                    )
                    try:
                        min_cache_coverage = float(
                            os.getenv("APEX_BACKTEST_CACHE_MIN_COVERAGE", "0.80") or "0.80"
                        )
                    except Exception:
                        min_cache_coverage = 0.80
                    min_cache_coverage = max(0.50, min(min_cache_coverage, 1.00))
                    accuracy_mode = run_accuracy_mode
                    accuracy_active = accuracy_mode in {"warn", "block"}
                    force_refresh_for_accuracy = _env_flag("APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY", "1")
                    incremental_refresh_enabled = _env_flag("APEX_BACKTEST_INCREMENTAL_REFRESH", "1")
                    strict_full_lookback = (
                        accuracy_mode == "block"
                        and _env_flag("APEX_BACKTEST_ENFORCE_FULL_LOOKBACK", "1")
                    )
                    if accuracy_mode == "block" and force_refresh_for_accuracy:
                        try:
                            strict_cap = int(
                                os.getenv("APEX_BACKTEST_REFRESH_MAX_SYMBOLS_STRICT", "0") or "0"
                            )
                        except Exception:
                            strict_cap = 0
                        refresh_cap = strict_cap if strict_cap > 0 else max(3000, len(symbols))
                    else:
                        try:
                            refresh_cap = int(
                                os.getenv("APEX_BACKTEST_REFRESH_MAX_SYMBOLS", "2500") or "2500"
                            )
                        except Exception:
                            refresh_cap = 2500
                    refresh_cap = max(100, min(refresh_cap, 10000))

                    data = {}
                    if symbols:
                        stage_msg.caption("Stage: Loading historical price data...")
                        workers_cache = _recommended_fetch_workers(len(symbols), cache_only=True)
                        workers_refresh = _recommended_fetch_workers(len(symbols), cache_only=False)
                        loader_progress = st.empty()
                        _loader_ui_last = {"t": 0.0}

                        def _loader_progress_cb(state):
                            event = str((state or {}).get("event", "") or "")
                            completed = int((state or {}).get("completed", 0) or 0)
                            total_syms = int((state or {}).get("total", len(symbols)) or len(symbols))
                            pending_syms = int(
                                (state or {}).get("pending", max(0, total_syms - completed))
                                or max(0, total_syms - completed)
                            )
                            elapsed_sec = float((state or {}).get("elapsed_sec", 0.0) or 0.0)
                            now_mono = time.monotonic()
                            if event not in {"start", "stall", "done"} and (now_mono - _loader_ui_last["t"]) < 1.0:
                                return
                            prefix = "Data loader"
                            if event == "stall":
                                prefix = "Data loader stall guard"
                            elif event == "done":
                                prefix = "Data loader complete"
                            loader_progress.caption(
                                f"{prefix}: {completed}/{total_syms} complete, "
                                f"{pending_syms} pending, elapsed {elapsed_sec:.0f}s"
                            )
                            _loader_ui_last["t"] = now_mono

                        st.caption(
                            f"Data loader workers: {workers_cache if cache_only_first else workers_refresh} "
                            "(threaded, single Python process)"
                        )
                        if cache_only_first:
                            quality_first = {}
                            data = fetch_data_pack(
                                symbols,
                                days=fetch_days,
                                backtest_mode=True,
                                max_workers=workers_cache,
                                progress_callback=_loader_progress_cb,
                                require_full_lookback=strict_full_lookback,
                                quality_report=quality_first,
                            ) or {}
                            _merge_quality(quality_first)
                            initial_cov = (len(data) / float(len(symbols))) if symbols else 0.0
                            st.caption(
                                f"Cache-first load: {len(data)}/{len(symbols)} "
                                f"symbols ({initial_cov:.1%} coverage)"
                            )
                            allow_refresh_when_closed = (
                                refresh_when_closed or (accuracy_active and force_refresh_for_accuracy)
                            )
                            allow_incremental_refresh = (
                                initial_cov < min_cache_coverage
                                and incremental_refresh_enabled
                                and (
                                    market_open
                                    or allow_refresh_when_closed
                                    or initial_cov <= 0.05
                                )
                            )
                            if allow_incremental_refresh:
                                missing_symbols = [s for s in symbols if s not in data]
                                refresh_symbols = list(missing_symbols)
                                if (
                                    refresh_symbols
                                    and len(refresh_symbols) > refresh_cap
                                    and not (accuracy_active and force_refresh_for_accuracy)
                                ):
                                    skipped = len(refresh_symbols) - refresh_cap
                                    refresh_symbols = refresh_symbols[:refresh_cap]
                                    st.warning(
                                        f"Refresh cap active: fetching first {len(refresh_symbols)} "
                                        f"of {len(missing_symbols)} missing symbols this run "
                                        f"({skipped} deferred)."
                                    )
                                st.warning(
                                    f"Cache coverage {initial_cov:.1%} below threshold "
                                    f"({min_cache_coverage:.0%}); fetching {len(refresh_symbols)} "
                                    "missing symbols only."
                                )
                                if refresh_symbols:
                                    stage_msg.caption("Stage: Refreshing missing symbols incrementally...")
                                    quality_refresh = {}
                                    fresh = fetch_data_pack(
                                        refresh_symbols,
                                        days=fetch_days,
                                        backtest_mode=False,
                                        max_workers=workers_refresh,
                                        progress_callback=_loader_progress_cb,
                                        require_full_lookback=strict_full_lookback,
                                        quality_report=quality_refresh,
                                    ) or {}
                                    _merge_quality(quality_refresh)
                                    if fresh:
                                        data.update(fresh)
                                merged_cov = (len(data) / float(len(symbols))) if symbols else 0.0
                                st.caption(
                                    f"Post-refresh coverage: {len(data)}/{len(symbols)} "
                                    f"symbols ({merged_cov:.1%})"
                                )
                            elif initial_cov < min_cache_coverage:
                                st.warning(
                                    f"Cache coverage {initial_cov:.1%} below threshold "
                                    f"({min_cache_coverage:.0%}), but network refresh is skipped while "
                                    "market is closed. Set `APEX_BACKTEST_REFRESH_WHEN_CLOSED=1` "
                                    "to override."
                                )
                        else:
                            quality_direct = {}
                            data = fetch_data_pack(
                                symbols,
                                days=fetch_days,
                                backtest_mode=False,
                                max_workers=workers_refresh,
                                progress_callback=_loader_progress_cb,
                                require_full_lookback=strict_full_lookback,
                                quality_report=quality_direct,
                            ) or {}
                            _merge_quality(quality_direct)

                    # Fetch global context once (required for RS + VIX overlays in the engine).
                    stage_msg.caption("Stage: Loading market context (SPY/VIX)...")
                    global_workers = _recommended_fetch_workers(3, cache_only=cache_only_first)
                    g_data = fetch_data_pack(
                        ["SPY", "$VIX", "VIX"],
                        days=fetch_days,
                        backtest_mode=cache_only_first,
                        max_workers=global_workers,
                    ) or {}
                    spy_df = g_data.get("SPY")
                    if spy_df is None or spy_df.empty:
                        g_data = fetch_data_pack(
                            ["SPY", "$VIX", "VIX"],
                            days=fetch_days,
                            backtest_mode=False,
                            max_workers=_recommended_fetch_workers(3, cache_only=False),
                        ) or {}
                    spy_df = g_data.get("SPY")
                    vix_df = g_data.get("$VIX")
                    if vix_df is None:
                        vix_df = g_data.get("VIX")
                    global_data = {"SPY": spy_df, "VIX": vix_df}

                    # Turbo: precompute indicators/arrays once, then reuse across all strategies.
                    stage_msg.caption("Stage: Building indicators and PIT-aligned data...")
                    prepared = prepare_backtest_data(
                        data,
                        symbol_universe=symbols,
                        start_date=bt_start_date,
                        global_data=global_data,
                    )
                    _set_backtest_cache(
                        cache_key,
                        {
                            "prepared": prepared,
                            "global_data": global_data,
                            "symbol_count": len(symbols),
                            "symbol_universe": list(symbols),
                            "universe_source": universe_source,
                        },
                    )
                    expected_symbol_count = len(symbols)
                    expected_symbols = list(symbols)
                    del data
                    del g_data
                    gc.collect()
                    cache_hit = prepared is not None
                elif not global_data:
                    g_data = fetch_data_pack(
                        ["SPY", "$VIX", "VIX"],
                        days=fetch_days,
                        backtest_mode=True,
                        max_workers=_recommended_fetch_workers(3, cache_only=True),
                    ) or {}
                    spy_df = g_data.get("SPY")
                    if spy_df is None or spy_df.empty:
                        g_data = fetch_data_pack(
                            ["SPY", "$VIX", "VIX"],
                            days=fetch_days,
                            backtest_mode=False,
                            max_workers=_recommended_fetch_workers(3, cache_only=False),
                        ) or {}
                    spy_df = g_data.get("SPY")
                    vix_df = g_data.get("$VIX")
                    if vix_df is None:
                        vix_df = g_data.get("VIX")
                    global_data = {"SPY": spy_df, "VIX": vix_df}
                    del g_data

                if cache_hit and getattr(prepared, "all_dates", None) is not None and len(prepared.all_dates) > 0:
                    stage_msg.caption("Stage: Validating cached dataset freshness...")
                    max_end_lag_days = int(os.getenv("APEX_BACKTEST_MAX_END_LAG_DAYS", "7") or "7")
                    max_end_lag_days = max(0, max_end_lag_days)
                    loaded_end_dt = pd.Timestamp(prepared.all_dates[-1]).tz_localize(None)
                    now_et = datetime.now(ZoneInfo("America/New_York")).date()
                    end_lag = (now_et - loaded_end_dt.date()).days
                    if end_lag > max_end_lag_days:
                        st.warning(
                            f"Cached prepared data ends on {loaded_end_dt.date().isoformat()} "
                            f"({end_lag} days stale). Refreshing for accuracy."
                        )
                        cache_hit = False
                        prepared = None

                if not cache_hit:
                    # Re-enter with fresh data for stale cache case.
                    stage_msg.caption("Stage: Refreshing stale prepared dataset...")
                    if bt_universe in ("Russell 3000", "RUSSELL3000"):
                        symbols, universe_source = get_universe_symbols_pit_window_with_meta(
                            "RUSSELL3000",
                            bt_start_date,
                            bt_end_date,
                        )
                        if universe_source == "fallback_current" and require_pit_universe:
                            st.error(
                                "Point-in-time Russell 3000 membership is required for accurate backtests. "
                                "Configure `RUSSELL3000_PIT_DIR` or `RUSSELL3000_PIT_MEMBERSHIP_CSV`."
                            )
                            st.session_state.backtest_results = {}
                            symbols = []
                    else:
                        symbols = get_index_symbols(bt_universe)
                        universe_source = "current_index"
                    expected_symbols = list(symbols or [])
                    expected_symbol_count = len(expected_symbols)
                    if symbols:
                        quality_rebuild = {}
                        data = fetch_data_pack(
                            symbols,
                            days=fetch_days,
                            backtest_mode=False,
                            max_workers=_recommended_fetch_workers(len(symbols), cache_only=False),
                            require_full_lookback=strict_full_lookback,
                            quality_report=quality_rebuild,
                        ) or {}
                        _merge_quality(quality_rebuild)
                        g_data = fetch_data_pack(
                            ["SPY", "$VIX", "VIX"],
                            days=fetch_days,
                            backtest_mode=False,
                            max_workers=_recommended_fetch_workers(3, cache_only=False),
                        ) or {}
                        spy_df = g_data.get("SPY")
                        vix_df = g_data.get("$VIX")
                        if vix_df is None:
                            vix_df = g_data.get("VIX")
                        global_data = {"SPY": spy_df, "VIX": vix_df}
                        prepared = prepare_backtest_data(
                            data,
                            symbol_universe=symbols,
                            start_date=bt_start_date,
                            global_data=global_data,
                        )
                        _set_backtest_cache(
                            cache_key,
                            {
                                "prepared": prepared,
                                "global_data": global_data,
                                "symbol_count": len(symbols),
                                "symbol_universe": list(symbols),
                                "universe_source": universe_source,
                            },
                        )
                        expected_symbol_count = len(symbols)
                        expected_symbols = list(symbols)
                        del data
                        del g_data
                        gc.collect()
                        cache_hit = prepared is not None

                tested_symbols = len(getattr(prepared, "enriched", {}) or {})
                st.caption(f"Symbols tested: {tested_symbols}")
                if universe_source != "unknown":
                    st.caption(f"Universe source: `{universe_source}`")
                tested_cov = 0.0
                loaded_start = None
                loaded_end = None
                tested_symbol_list = list((getattr(prepared, "enriched", {}) or {}).keys())
                min_cov_required = _min_coverage_threshold(bt_universe, accuracy_mode=accuracy_mode)
                if expected_symbol_count > 0:
                    tested_cov = tested_symbols / float(max(1, expected_symbol_count))
                    st.caption(
                        f"Universe coverage used in run: {tested_symbols}/{expected_symbol_count} "
                        f"({tested_cov:.1%})"
                    )
                    if tested_cov < min_cov_required:
                        st.warning(
                            "Universe coverage is below this run's target and can bias results. "
                            "Use strict verified mode for baseline-quality runs."
                        )
                if getattr(prepared, "all_dates", None) is not None and len(prepared.all_dates) > 0:
                    loaded_start = pd.Timestamp(prepared.all_dates[0]).date().isoformat()
                    loaded_end = pd.Timestamp(prepared.all_dates[-1]).date().isoformat()
                    st.caption(f"Loaded data range: {loaded_start} to {loaded_end}")

                max_end_lag_days = int(os.getenv("APEX_BACKTEST_MAX_END_LAG_DAYS", "7") or "7")
                max_end_lag_days = max(0, max_end_lag_days)
                recent_scope = None
                if bt_universe in ("Russell 3000", "RUSSELL3000") and bt_end_date:
                    end_members, end_source = get_universe_symbols_pit_with_meta("RUSSELL3000", bt_end_date)
                    if end_members:
                        recent_scope = {str(s).upper() for s in end_members if str(s).strip()}
                        st.caption(
                            f"Recent coverage scope: end-of-window PIT membership "
                            f"({len(recent_scope)} symbols, source `{end_source}`)."
                        )
                fresh_symbols, fresh_total, fresh_cov = _recent_data_coverage(
                    prepared,
                    expected_symbol_count=len(recent_scope) if recent_scope else expected_symbol_count,
                    max_lag_days=max_end_lag_days,
                    symbol_scope=recent_scope,
                )
                if fresh_total > 0:
                    st.caption(
                        f"Recent-bar coverage (<= {max_end_lag_days} days lag): "
                        f"{fresh_symbols}/{fresh_total} ({fresh_cov:.1%})"
                    )

                loaded_set = set(tested_symbol_list)
                expected_set = set(expected_symbols or [])
                missing_set = set(quality_union.get("missing", set()))
                if expected_set:
                    missing_set.update(expected_set - loaded_set)
                missing_set -= loaded_set
                incomplete_set = set(quality_union.get("incomplete", set()))
                stale_set = set(quality_union.get("stale", set()))
                incomplete_set -= missing_set
                stale_set -= missing_set

                if accuracy_mode == "block":
                    min_recent_cov_required = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MIN_RECENT_COVERAGE_STRICT",
                                    "0.75",
                                )
                                or "0.75"
                            ),
                        ),
                    )
                    max_incomplete_ratio = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MAX_INCOMPLETE_RATIO_STRICT",
                                    "0.10",
                                )
                                or "0.10"
                            ),
                        ),
                    )
                    max_stale_ratio = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MAX_STALE_RATIO_STRICT",
                                    "0.10",
                                )
                                or "0.10"
                            ),
                        ),
                    )
                elif accuracy_mode == "warn":
                    min_recent_cov_required = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MIN_RECENT_COVERAGE_WARN",
                                    "0.60",
                                )
                                or "0.60"
                            ),
                        ),
                    )
                    max_incomplete_ratio = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MAX_INCOMPLETE_RATIO_WARN",
                                    "0.20",
                                )
                                or "0.20"
                            ),
                        ),
                    )
                    max_stale_ratio = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                os.getenv(
                                    "APEX_BACKTEST_MAX_STALE_RATIO_WARN",
                                    "0.20",
                                )
                                or "0.20"
                            ),
                        ),
                    )
                else:
                    min_recent_cov_required = 0.0
                    max_incomplete_ratio = 1.0
                    max_stale_ratio = 1.0

                incomplete_base = max(1, len(loaded_set) + len(incomplete_set))
                incomplete_ratio = len(incomplete_set) / float(incomplete_base)
                stale_ratio = len(stale_set) / float(max(1, len(loaded_set)))
                spy_ok = bool(global_data.get("SPY") is not None and not global_data.get("SPY").empty)
                vix_ok = bool(global_data.get("VIX") is not None and not global_data.get("VIX").empty)
                recent_cov_pass = (fresh_total <= 0) or (fresh_cov >= min_recent_cov_required)
                incomplete_pass = incomplete_ratio <= max_incomplete_ratio
                stale_pass = stale_ratio <= max_stale_ratio
                coverage_pass = not (expected_symbol_count > 0 and tested_cov < min_cov_required)
                lock_blockers = []
                if not coverage_pass:
                    lock_blockers.append(
                        f"coverage {tested_cov:.1%} below required {min_cov_required:.0%}"
                    )
                if not recent_cov_pass:
                    lock_blockers.append(
                        f"recent-bar coverage {fresh_cov:.1%} below {min_recent_cov_required:.0%}"
                    )
                if not incomplete_pass:
                    lock_blockers.append(
                        f"incomplete-history ratio {incomplete_ratio:.1%} above {max_incomplete_ratio:.0%}"
                    )
                if not stale_pass:
                    lock_blockers.append(
                        f"stale-data ratio {stale_ratio:.1%} above {max_stale_ratio:.0%}"
                    )
                if not spy_ok:
                    lock_blockers.append("SPY market context missing")

                scorecard_rows = [
                    {
                        "Metric": "Universe coverage",
                        "Value": f"{tested_symbols}/{expected_symbol_count} ({tested_cov:.1%})",
                        "Threshold": f">= {min_cov_required:.0%}",
                        "Status": "PASS" if coverage_pass else "FAIL",
                    },
                    {
                        "Metric": "Recent-bar coverage",
                        "Value": f"{fresh_symbols}/{fresh_total} ({fresh_cov:.1%})" if fresh_total > 0 else "n/a",
                        "Threshold": f">= {min_recent_cov_required:.0%}",
                        "Status": "PASS" if recent_cov_pass else "FAIL",
                    },
                    {
                        "Metric": "Incomplete history ratio",
                        "Value": f"{len(incomplete_set)} symbol(s) ({incomplete_ratio:.1%})",
                        "Threshold": f"<= {max_incomplete_ratio:.0%}",
                        "Status": "PASS" if incomplete_pass else "FAIL",
                    },
                    {
                        "Metric": "Stale data ratio",
                        "Value": f"{len(stale_set)} symbol(s) ({stale_ratio:.1%})",
                        "Threshold": f"<= {max_stale_ratio:.0%}",
                        "Status": "PASS" if stale_pass else "FAIL",
                    },
                    {
                        "Metric": "Market context",
                        "Value": f"SPY={'OK' if spy_ok else 'MISSING'}, VIX={'OK' if vix_ok else 'MISSING'}",
                        "Threshold": "SPY required (VIX recommended)",
                        "Status": "PASS" if spy_ok else "FAIL",
                    },
                ]
                st.markdown("**Accuracy Scorecard**")
                st.dataframe(pd.DataFrame(scorecard_rows), use_container_width=True, hide_index=True)

                low_cov = expected_symbol_count > 0 and tested_cov < min_cov_required
                if low_cov:
                    if accuracy_mode == "block":
                        st.error(
                            "Accuracy gate blocked this run. "
                            f"Coverage {tested_cov:.1%} is below required {min_cov_required:.0%} "
                            f"for {bt_universe}. Results withheld to avoid biased metrics."
                        )
                        st.caption(
                            "Current gap indicates data-provider coverage limits for this PIT window "
                            "(commonly delisted/renamed symbols)."
                        )
                        _drop_backtest_cache(cache_key)
                        st.session_state.backtest_results = {}
                        prepared = None
                    elif accuracy_mode == "warn":
                        st.warning(
                            "Accuracy warning: run allowed in best-effort mode. "
                            f"Coverage {tested_cov:.1%} is below target {min_cov_required:.0%} "
                            f"for {bt_universe}, so results may be biased."
                        )
                        export_missing = _env_flag(
                            "APEX_EXPORT_MISSING_SYMBOLS_ON_LOW_COVERAGE",
                            "1",
                        )
                        if export_missing and expected_symbols:
                            try:
                                report_path = _write_missing_symbols_report(
                                    universe_name=bt_universe,
                                    start_date=bt_start_date or "max",
                                    universe_source=universe_source,
                                    expected_symbols=expected_symbols,
                                    tested_symbols=tested_symbol_list,
                                )
                                st.caption(f"Missing symbols report: `{report_path}`")
                            except Exception:
                                pass
                        # Do not persist low-coverage prepared payloads; force rebuild on next run.
                        _drop_backtest_cache(cache_key)

                results_map = {}
                if prepared is None or tested_symbols == 0:
                    st.error("Backtest dataset is unavailable or empty; run aborted for accuracy.")
                    st.session_state.backtest_results = {}
                else:
                    run_strategies = load_strategies(selected_strategies)
                    if not run_strategies:
                        st.error("No runnable strategies were loaded.")
                        st.session_state.backtest_results = {}
                    else:
                        status_text = st.empty()
                        stage_msg.caption("Stage: Running strategy simulation...")
                        universe_membership_by_day = None
                        membership_source = "none"
                        if bt_universe in ("Russell 3000", "RUSSELL3000"):
                            prepared_dates = getattr(prepared, "all_dates", None)
                            if prepared_dates is None:
                                prepared_dates_seq = []
                            else:
                                prepared_dates_seq = list(prepared_dates)
                            membership_series, membership_source = build_russell3000_membership_by_day(
                                prepared_dates_seq
                            )
                            if membership_series and len(membership_series) == len(prepared_dates_seq):
                                universe_membership_by_day = membership_series
                                st.caption(
                                    "PIT timeline applied for entries: "
                                    f"`{membership_source}`"
                                )
                            else:
                                st.warning(
                                    "PIT timeline could not be constructed for this run; "
                                    "falling back to static start-window membership."
                                )
                        start_time = time.time()
                        status_text.text(f"Running {len(run_strategies)} strategy simulation(s)...")
                        try:
                            raw_results = run_backtest(
                                run_strategies,
                                prepared,
                                start_cash=100000.0,
                                start_date=bt_start_date,
                                global_data=global_data,
                                universe_membership_by_day=universe_membership_by_day,
                            )
                        except Exception as e:
                            st.error(f"Backtest run failed: {e}")
                            raw_results = []

                        elapsed = max(0.0, time.time() - start_time)
                        status_text.text(f"Completed in {elapsed:.1f}s")
                        time.sleep(0.15)
                        status_text.empty()
                        stage_msg.empty()

                        if isinstance(raw_results, dict):
                            raw_results = [raw_results]
                        for res in raw_results or []:
                            if not isinstance(res, dict):
                                continue
                            strategy_name = str(
                                res.get("strategy")
                                or res.get("strategy_name")
                                or f"Strategy_{len(results_map) + 1}"
                            )
                            results_map[strategy_name] = res

                st.session_state.backtest_results = results_map
                lock_eligible = len(lock_blockers) == 0
                st.session_state.backtest_run_context = {
                    "universe": bt_universe,
                    "duration": bt_duration,
                    "start_date": bt_start_date or "max",
                    "strategy_fingerprint": strategy_fp,
                    "tested_symbols": int(tested_symbols),
                    "expected_symbol_count": int(expected_symbol_count),
                    "coverage": float(tested_cov),
                    "universe_source": universe_source,
                    "loaded_start": loaded_start,
                    "loaded_end": loaded_end,
                    "accuracy_mode": accuracy_mode,
                    "verified_run": bool(verified_run),
                    "quality": {
                        "requested": int(max(len(expected_set), len(quality_union.get("requested", set())))),
                        "loaded": int(tested_symbols),
                        "missing": int(len(missing_set)),
                        "incomplete_history": int(len(incomplete_set)),
                        "stale": int(len(stale_set)),
                        "fresh_cov": float(fresh_cov),
                        "fresh_total": int(fresh_total),
                    },
                    "lock_eligible": bool(lock_eligible),
                    "lock_blockers": list(lock_blockers),
                }

    if "backtest_results" in st.session_state and st.session_state.backtest_results:
        run_ctx = st.session_state.get("backtest_run_context", {}) or {}
        baselines = _load_backtest_baselines()
        tol_final_rel = max(0.0, _safe_float(os.getenv("APEX_BASELINE_TOL_FINAL_VALUE_REL", "0.02"), 0.02))
        tol_cagr_rel = max(0.0, _safe_float(os.getenv("APEX_BASELINE_TOL_CAGR_REL", "0.02"), 0.02))
        tol_trades_rel = max(0.0, _safe_float(os.getenv("APEX_BASELINE_TOL_TRADES_REL", "0.03"), 0.03))
        tol_win_rate_pts = max(0.0, _safe_float(os.getenv("APEX_BASELINE_TOL_WIN_RATE_PTS", "1.5"), 1.5))
        tol_cov_rel = max(0.0, _safe_float(os.getenv("APEX_BASELINE_TOL_COVERAGE_REL", "0.03"), 0.03))

        tabs = st.tabs(list(st.session_state.backtest_results.keys()))
        for i, name in enumerate(st.session_state.backtest_results.keys()):
            res = st.session_state.backtest_results[name]
            with tabs[i]:
                ec_data = res.get("equity_curve", [])
                df_ec = normalize_equity_curve_df(ec_data)

                diagnostics = _trade_diagnostics(res)
                avg_trade_pct_display = diagnostics["avg_trade_pct"]
                expectancy_dollar_display = diagnostics["expectancy_dollar"]
                profit_factor_display = diagnostics["profit_factor"]
                trade_count_display = diagnostics["trade_count"]
                win_rate_display = diagnostics["win_rate_pct"]

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

                strategy_name = str(res.get("strategy_name") or res.get("strategy") or name or "Backtest_Result")
                baseline_key = _baseline_key(
                    universe=str(run_ctx.get("universe", bt_universe)),
                    duration=str(run_ctx.get("duration", bt_duration)),
                    start_date=str(run_ctx.get("start_date", bt_start_date or "max")),
                    strategy_name=strategy_name,
                    strategy_fingerprint=str(run_ctx.get("strategy_fingerprint", strategy_fp)),
                )
                baseline_record = baselines.get(baseline_key)

                current_final_value = _safe_float(res.get("final_value"), 0.0)
                if current_final_value <= 0 and not df_ec.empty:
                    current_final_value = _safe_float(df_ec["Equity"].iloc[-1], 0.0)
                current_cagr = _safe_float(res.get("cagr"), 0.0)
                current_total_trades = int(trade_count_display)
                current_win_rate = _safe_float(win_rate_display, 0.0)
                current_cov = _safe_float(run_ctx.get("coverage"), 0.0)
                lock_eligible = bool(run_ctx.get("lock_eligible", True))
                lock_blockers = list(run_ctx.get("lock_blockers") or [])

                pf_store = None
                if profit_factor_display == float("inf"):
                    pf_store = "inf"
                elif profit_factor_display is not None and pd.notna(profit_factor_display):
                    pf_store = float(profit_factor_display)

                if not lock_eligible:
                    blocker_msg = "; ".join(lock_blockers) if lock_blockers else "accuracy thresholds not met"
                    st.warning(f"Baseline lock disabled for this run: {blocker_msg}.")

                if st.button(
                    "🔒 Lock This Run As Baseline",
                    key=f"lock_baseline_{i}_{strategy_name}",
                    disabled=not lock_eligible,
                ):
                    payload = _load_backtest_baselines()
                    payload[baseline_key] = {
                        "locked_at_utc": datetime.utcnow().isoformat() + "Z",
                        "context": {
                            "universe": str(run_ctx.get("universe", bt_universe)),
                            "duration": str(run_ctx.get("duration", bt_duration)),
                            "start_date": str(run_ctx.get("start_date", bt_start_date or "max")),
                            "strategy_name": strategy_name,
                            "strategy_fingerprint": str(run_ctx.get("strategy_fingerprint", strategy_fp)),
                            "universe_source": str(run_ctx.get("universe_source", "")),
                            "loaded_start": run_ctx.get("loaded_start"),
                            "loaded_end": run_ctx.get("loaded_end"),
                        },
                        "coverage": {
                            "tested_symbols": int(run_ctx.get("tested_symbols", 0) or 0),
                            "expected_symbol_count": int(run_ctx.get("expected_symbol_count", 0) or 0),
                            "tested_cov": float(run_ctx.get("coverage", 0.0) or 0.0),
                        },
                        "metrics": {
                            "final_value": float(current_final_value),
                            "cagr": float(current_cagr),
                            "max_drawdown_pct": float(_safe_float(res.get("max_drawdown_pct"), 0.0)),
                            "total_trades": int(current_total_trades),
                            "win_rate_pct": float(current_win_rate),
                            "profit_factor": pf_store,
                        },
                    }
                    _save_backtest_baselines(payload)
                    baselines = payload
                    baseline_record = payload.get(baseline_key)
                    st.success("Baseline locked for this strategy fingerprint and backtest window.")

                if baseline_record:
                    base_metrics = baseline_record.get("metrics", {}) or {}
                    base_cov = (baseline_record.get("coverage", {}) or {}).get("tested_cov")
                    base_cov = _safe_float(base_cov, 0.0)
                    base_final = _safe_float(base_metrics.get("final_value"), 0.0)
                    base_cagr = _safe_float(base_metrics.get("cagr"), 0.0)
                    base_trades = int(base_metrics.get("total_trades", 0) or 0)
                    base_win_rate = _safe_float(base_metrics.get("win_rate_pct"), 0.0)

                    rel = lambda curr, base: abs(curr - base) / max(abs(base), 1e-9)
                    checks = [
                        {
                            "Check": "Final Value",
                            "Delta": f"{rel(current_final_value, base_final):.2%}",
                            "Threshold": f"{tol_final_rel:.2%}",
                            "Status": "PASS" if rel(current_final_value, base_final) <= tol_final_rel else "DRIFT",
                        },
                        {
                            "Check": "CAGR",
                            "Delta": f"{rel(current_cagr, base_cagr):.2%}",
                            "Threshold": f"{tol_cagr_rel:.2%}",
                            "Status": "PASS" if rel(current_cagr, base_cagr) <= tol_cagr_rel else "DRIFT",
                        },
                        {
                            "Check": "Total Trades",
                            "Delta": f"{rel(float(current_total_trades), float(base_trades)):.2%}",
                            "Threshold": f"{tol_trades_rel:.2%}",
                            "Status": "PASS" if rel(float(current_total_trades), float(base_trades)) <= tol_trades_rel else "DRIFT",
                        },
                        {
                            "Check": "Win Rate",
                            "Delta": f"{abs(current_win_rate - base_win_rate):.2f} pts",
                            "Threshold": f"{tol_win_rate_pts:.2f} pts",
                            "Status": "PASS" if abs(current_win_rate - base_win_rate) <= tol_win_rate_pts else "DRIFT",
                        },
                        {
                            "Check": "Coverage",
                            "Delta": f"{abs(current_cov - base_cov):.2%}",
                            "Threshold": f"{tol_cov_rel:.2%}",
                            "Status": "PASS" if abs(current_cov - base_cov) <= tol_cov_rel else "DRIFT",
                        },
                    ]
                    pass_all = all(r["Status"] == "PASS" for r in checks)
                    if pass_all:
                        st.success("Repeatability check: PASS (within baseline tolerances).")
                    else:
                        st.warning("Repeatability check: DRIFT detected vs locked baseline.")
                    st.caption(
                        "Baseline locked at "
                        f"`{baseline_record.get('locked_at_utc', 'unknown')}` "
                        f"for fingerprint `{run_ctx.get('strategy_fingerprint', strategy_fp)}`."
                    )
                    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

                    if st.button("🗑️ Clear Baseline", key=f"clear_baseline_{i}_{strategy_name}"):
                        payload = _load_backtest_baselines()
                        payload.pop(baseline_key, None)
                        _save_backtest_baselines(payload)
                        st.success("Baseline cleared.")

                if not df_ec.empty:
                    st.line_chart(df_ec.set_index("Date")["Equity"])
                elif ec_data:
                    st.warning("Equity data malformed.")
                
                equity_df = df_ec.copy()
                safe_name = "".join(
                    [c for c in strategy_name if c.isalnum() or c in (" ", "_", "-")]
                ).strip()

                if not equity_df.empty:
                    try:
                        csv_data = equity_df.to_csv(index=False).encode('utf-8')
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
