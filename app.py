import streamlit as st
import pandas as pd
import sys
import os
import json
import hashlib
import math
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo
import gc
from pathlib import Path

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
PRIMARY_STRATEGY_CONFIG_PATH = os.path.join("config", "superperformance_winner.json")
BASELINE_CONFIG_PATH = os.path.join("config", "backtest_baselines.json")
UNIVERSE_OPTIONS = ["SP500", "SP100", "SP1500", "NASDAQ100", "RUSSELL3000"]
DEFAULT_UNIVERSE = "RUSSELL3000"
ETF_FROZEN_BENCHMARKS = {
    "Validated ETF Benchmark (Frozen)": os.path.join(
        "config", "etf_rotation_growth_core5_residual_defensive_calmar_v1.json"
    ),
    "Higher Return ETF Benchmark": os.path.join(
        "config", "etf_rotation_growth_core5_residual_defensive_top1_v1.json"
    ),
}
ETF_FROZEN_DEFAULT_LABEL = "Validated ETF Benchmark (Frozen)"
STOCK_BENCHMARK_CANDIDATES = {
    "Validated Stock Benchmark (Risk-Adjusted)": os.path.join(
        "config", "smid_pullback_r3000_tc4_tb003_v1.json"
    ),
    "Higher Return Stock Benchmark": os.path.join(
        "config", "smid_pullback_r3000_tb006_v1.json"
    ),
}
STOCK_BENCHMARK_DEFAULT_LABEL = "Validated Stock Benchmark (Risk-Adjusted)"
HYBRID_BENCHMARK_CANDIDATES = {
    "Validated Hybrid Benchmark (25/75)": {
        "etf_profile_label": ETF_FROZEN_DEFAULT_LABEL,
        "stock_profile_label": STOCK_BENCHMARK_DEFAULT_LABEL,
        "etf_weight": 0.25,
    },
    "Balanced Hybrid Benchmark (50/50)": {
        "etf_profile_label": ETF_FROZEN_DEFAULT_LABEL,
        "stock_profile_label": STOCK_BENCHMARK_DEFAULT_LABEL,
        "etf_weight": 0.50,
    },
}
HYBRID_BENCHMARK_DEFAULT_LABEL = "Validated Hybrid Benchmark (25/75)"
HYBRID_VALIDATED_BENCHMARK_CONFIG = os.path.join("config", "hybrid_benchmark_v2.json")
HYBRID_VALIDATED_RISK_ALT_CONFIG = os.path.join("config", "hybrid_benchmark_v2_risk_alt.json")
HYBRID_LEGACY_HOLDOUT_ARTIFACT = os.path.join("tmp", "hybrid_holdout_promoted.json")
PRIMARY_STRATEGY_OPTIONS = ["Apex Swing"]
ETF_PAPER_STATE_FILE = os.path.join("data", "etf_paper_state.json")
HYBRID_BENCHMARK_PAPER_STATE_FILE = os.path.join("data", "hybrid_benchmark_paper_state.json")
STOCK_BENCHMARK_PAPER_STATE_FILE = os.path.join("data", "stock_benchmark_paper_state.json")
ETF_LIVE_SNAPSHOT_FILE = os.path.join("data", "etf_live_snapshot.pkl")
HYBRID_LIVE_SNAPSHOT_FILE = os.path.join("data", "hybrid_live_snapshot.pkl")
STOCK_LIVE_SNAPSHOT_FILE = os.path.join("data", "stock_live_snapshot.pkl")
st.set_page_config(page_title="Apex Sniper AI", layout="wide", page_icon="🎯")


def _inject_global_styles() -> None:
    """Apply a consistent, professional visual system across Streamlit surfaces."""
    st.markdown(
        """
<style>
:root {
  --apx-bg-0: #f4f7fb;
  --apx-bg-1: #eef3f9;
  --apx-card: #ffffff;
  --apx-surface: #ffffff;
  --apx-border: #d8e0eb;
  --apx-text: #202739;
  --apx-muted: #617086;
  --apx-sidebar-0: #f5f8fc;
  --apx-sidebar-1: #edf2f8;
  --apx-accent-1: #ff5d5d;
  --apx-accent-2: #e93b3b;
}

@media (prefers-color-scheme: dark) {
  :root {
    --apx-bg-0: #0f1722;
    --apx-bg-1: #132031;
    --apx-card: #1a2736;
    --apx-surface: #172334;
    --apx-border: #2f425a;
    --apx-text: #e8eff8;
    --apx-muted: #9eb2ca;
    --apx-sidebar-0: #101a27;
    --apx-sidebar-1: #142134;
    --apx-accent-1: #ff6b6b;
    --apx-accent-2: #f04f4f;
  }
}

html, body, [class*="css"] {
  font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
}

body {
  background: var(--apx-bg-0) !important;
  color: var(--apx-text) !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 600px at 100% -5%, rgba(137, 171, 214, 0.28) 0%, transparent 45%),
    linear-gradient(180deg, var(--apx-bg-0) 0%, var(--apx-bg-1) 100%);
  min-height: 100vh;
  color: var(--apx-text);
}

[data-testid="stHeader"] {
  display: none !important;
}

[data-testid="stDecoration"] {
  display: none !important;
}

[data-testid="stToolbar"] {
  display: none !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--apx-sidebar-0) 0%, var(--apx-sidebar-1) 100%);
  border-right: 1px solid var(--apx-border);
}
[data-testid="stSidebar"] * {
  color: var(--apx-text) !important;
}

[data-testid="stMainBlockContainer"] {
  max-width: 1400px;
  padding-top: 0.7rem;
  padding-bottom: 2.8rem;
}

.apx-mode-header {
  background: linear-gradient(120deg, var(--apx-surface) 0%, var(--apx-card) 100%);
  border: 1px solid var(--apx-border);
  border-radius: 14px;
  padding: 0.9rem 1rem 0.85rem 1rem;
  margin-bottom: 0.8rem;
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}

.apx-mode-title {
  font-size: 2rem;
  font-weight: 700;
  color: var(--apx-text);
  letter-spacing: -0.01em;
  line-height: 1.15;
}

.apx-mode-subtitle {
  margin-top: 0.25rem;
  color: var(--apx-muted);
  font-size: 0.95rem;
  font-weight: 500;
}

h1, h2, h3 {
  color: var(--apx-text);
  letter-spacing: -0.01em;
}

p, label, .stCaption, .stMarkdown, .stText {
  color: var(--apx-text) !important;
}

[data-testid="stMetric"] {
  background: var(--apx-card);
  border: 1px solid var(--apx-border);
  border-radius: 14px;
  padding: 0.5rem 0.75rem;
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.04);
}

div.stButton > button {
  border-radius: 10px;
  border: 1px solid var(--apx-border);
  min-height: 2.35rem;
  font-weight: 600;
  letter-spacing: 0.01em;
  box-shadow: 0 1px 4px rgba(16, 24, 40, 0.05);
}

[data-testid="baseButton-primary"] {
  background: linear-gradient(180deg, var(--apx-accent-1) 0%, var(--apx-accent-2) 100%);
  border: none !important;
  color: #ffffff !important;
  box-shadow: 0 6px 16px rgba(233, 59, 59, 0.24);
}

[data-testid="baseButton-primary"]:hover {
  filter: brightness(0.98);
}

[data-baseweb="select"] > div {
  border-radius: 10px;
  border: 1px solid var(--apx-border);
  background: var(--apx-card);
  color: var(--apx-text);
  box-shadow: 0 1px 4px rgba(16, 24, 40, 0.04);
}

[data-testid="stAlert"] {
  border-radius: 12px;
  border: 1px solid var(--apx-border);
}

[data-testid="stExpander"] {
  background: var(--apx-card);
  border: 1px solid var(--apx-border);
  border-radius: 12px;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--apx-border);
  border-radius: 12px;
  overflow: hidden;
  background: var(--apx-card);
}

[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
  color: var(--apx-text) !important;
}

[data-baseweb="menu"], [role="listbox"] {
  background: var(--apx-card) !important;
  color: var(--apx-text) !important;
}

[data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
  background: var(--apx-card) !important;
  border-color: var(--apx-border) !important;
}

input, textarea {
  color: var(--apx-text) !important;
}

[data-testid="stProgress"] > div > div {
  height: 0.58rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


_inject_global_styles()

# --- HELPERS ---
def _is_wealth_strategy(config: dict) -> bool:
    return True  # SHOW ALL STRATEGIES (Debug Mode)

def load_strategy_configs():
    if os.path.exists(PRIMARY_STRATEGY_CONFIG_PATH):
        with open(PRIMARY_STRATEGY_CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            winner = dict(payload)
            winner["enabled_by_default"] = True
            return [winner]
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [s for s in payload if isinstance(s, dict) and _is_wealth_strategy(s)]
    return []

def format_rule(r):
    return f"{r.get('col')} {r.get('op')} {r.get('val', r.get('ref'))}"

def color_pnl(val):
    color = 'green' if val > 0 else 'red' if val < 0 else '#6b7280'
    return f'color: {color}'

def score_to_rating(score):
    if score >= 85: return "🔥 Excellent"
    elif score >= 75: return "⭐ Strong"
    elif score >= 60: return "✅ Good"
    elif score >= 50: return "⚠️ Fair"
    return "❌ Weak"


def render_mode_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="apx-mode-header">
  <div class="apx-mode-title">{title}</div>
  <div class="apx-mode-subtitle">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _utc_now_iso() -> str:
    return datetime.now(ZoneInfo("UTC")).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@st.cache_data(ttl=900, show_spinner=False)
def _load_hybrid_benchmark_catalog() -> Dict[str, Any]:
    def _load_json(path: str) -> Dict[str, Any]:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return {}

    legacy_payload = _load_json(HYBRID_LEGACY_HOLDOUT_ARTIFACT)
    legacy_default_holdout: Dict[str, Any] = {}
    for row in list(legacy_payload.get("hybrids") or []):
        try:
            etf_weight = float(row.get("etf_weight"))
        except Exception:
            continue
        if abs(etf_weight - 0.25) < 1e-9:
            legacy_default_holdout = dict(row.get("holdout") or {})
            break

    return {
        "promoted": _load_json(HYBRID_VALIDATED_BENCHMARK_CONFIG),
        "risk_alt": _load_json(HYBRID_VALIDATED_RISK_ALT_CONFIG),
        "legacy_default_holdout": legacy_default_holdout,
    }


def _fmt_pct(value: Any, decimals: int = 2) -> str:
    try:
        num = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(num):
        return "n/a"
    return f"{num:.{decimals}f}%"


def _render_hybrid_benchmark_reference_notice() -> None:
    catalog = _load_hybrid_benchmark_catalog()
    promoted = dict(catalog.get("promoted") or {})
    promoted_validated = dict(promoted.get("validated") or {})
    risk_alt = dict(catalog.get("risk_alt") or {})
    risk_alt_validated = dict(risk_alt.get("validated") or {})
    legacy_default_holdout = dict(catalog.get("legacy_default_holdout") or {})

    st.warning(
        "The current Hybrid Benchmark GUI still runs the legacy ETF + SMID implementation. "
        "The promoted cash-only v2 benchmark adds a B17 sleeve and is shown here as the validated reference until live B17 integration is wired."
    )

    promoted_label = str(promoted.get("label") or "Hybrid Benchmark v2")
    st.caption(
        f"Promoted benchmark: {promoted_label} | "
        f"holdout {_fmt_pct(promoted_validated.get('holdout_cagr_pct'))} CAGR / "
        f"{_fmt_pct(promoted_validated.get('holdout_dd_pct'))} max DD | "
        f"full walk-forward {_fmt_pct(promoted_validated.get('full_cagr_pct'))} CAGR / "
        f"{_fmt_pct(promoted_validated.get('full_max_dd_pct'))} max DD."
    )

    risk_alt_label = str(risk_alt.get("label") or "Hybrid Benchmark v2 Risk Alt")
    st.caption(
        f"Risk alt: {risk_alt_label} | "
        f"holdout {_fmt_pct(risk_alt_validated.get('holdout_cagr_pct'))} CAGR / "
        f"{_fmt_pct(risk_alt_validated.get('holdout_dd_pct'))} max DD | "
        f"full walk-forward {_fmt_pct(risk_alt_validated.get('full_cagr_pct'))} CAGR / "
        f"{_fmt_pct(risk_alt_validated.get('full_max_dd_pct'))} max DD."
    )
    if legacy_default_holdout:
        st.caption(
            f"Legacy two-sleeve reference (25/75 ETF/SMID): "
            f"{_fmt_pct(legacy_default_holdout.get('cagr_pct'))} CAGR / "
            f"{_fmt_pct(legacy_default_holdout.get('max_dd_pct'))} max DD on the 2021-01-01 to 2025-12-31 holdout."
        )


def _display_profile_label(label: str) -> str:
    mapping = {
        ETF_FROZEN_DEFAULT_LABEL: "ETF Baseline",
        "Higher Return ETF Benchmark": "ETF Higher Return",
        HYBRID_BENCHMARK_DEFAULT_LABEL: "Legacy Hybrid 25/75",
        "Balanced Hybrid Benchmark (50/50)": "Legacy Hybrid 50/50",
        STOCK_BENCHMARK_DEFAULT_LABEL: "Stock Leader",
        "Higher Return Stock Benchmark": "Stock Higher Return",
    }
    return str(mapping.get(str(label), label))


def _current_benchmark_planning_capital() -> float:
    return max(float(_safe_float(st.session_state.get("benchmark_planning_capital"), 100000.0)), 1000.0)


def _planning_capital_label(capital: float) -> str:
    capital = float(_safe_float(capital, 100000.0))
    if capital >= 1000.0 and abs(capital % 1000.0) < 1e-9:
        return f"${capital / 1000.0:,.0f}k"
    return f"${capital:,.0f}"


def _estimate_shares_for_notional(notional: Any, price: Any) -> float:
    px = float(_safe_float(price, float("nan")))
    ntl = float(_safe_float(notional, float("nan")))
    if not math.isfinite(px) or px <= 0.0 or not math.isfinite(ntl):
        return float("nan")
    return ntl / px


def _extract_close_map_from_snapshot(snapshot: Mapping[str, Any]) -> Dict[str, float]:
    close_map: Dict[str, float] = {}
    for key in ("target_df", "rebalance_df"):
        df = snapshot.get(key)
        if isinstance(df, pd.DataFrame) and not df.empty and {"Symbol", "Last Close"}.issubset(df.columns):
            for row in df[["Symbol", "Last Close"]].to_dict("records"):
                sym = str(row.get("Symbol", "")).strip()
                if not sym:
                    continue
                close_map[sym] = float(_safe_float(row.get("Last Close"), float("nan")))
    return close_map


def _prepare_target_order_table(
    df: pd.DataFrame,
    *,
    planning_capital: float,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), {}

    out = df.copy()
    capital_label = _planning_capital_label(planning_capital)
    target_notional_col = f"Target $ @ {capital_label}"
    est_shares_col = f"Est. Shares @ {capital_label}"

    if "Target Weight" in out.columns:
        weights = pd.to_numeric(out["Target Weight"], errors="coerce").fillna(0.0)
    else:
        weights = pd.to_numeric(out.get("Target %"), errors="coerce").fillna(0.0) / 100.0
    planning_px = pd.to_numeric(out.get("Last Close"), errors="coerce")

    out["Model Fill"] = "Next Open"
    out["Planning Px"] = planning_px
    out[target_notional_col] = weights * float(planning_capital)
    out[est_shares_col] = [
        _estimate_shares_for_notional(notional, price)
        for notional, price in zip(out[target_notional_col], out["Planning Px"])
    ]

    order_cols = [col for col in ("Sleeve", "Source Profile", "Symbol", "Target %", "Weight %") if col in out.columns]
    order_cols.extend(["Model Fill", "Planning Px", target_notional_col, est_shares_col])
    column_config = {
        "Target %": st.column_config.NumberColumn(format="%.2f%%"),
        "Weight %": st.column_config.NumberColumn(format="%.2f%%"),
        "Planning Px": st.column_config.NumberColumn(format="$%.2f"),
        target_notional_col: st.column_config.NumberColumn(format="$%.0f"),
        est_shares_col: st.column_config.NumberColumn(format="%.1f"),
    }
    return out[order_cols], column_config


def _target_weight_fraction_series(df: pd.DataFrame) -> pd.Series:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype="float64")
    if "Target Weight" in df.columns:
        return pd.to_numeric(df["Target Weight"], errors="coerce").fillna(0.0)
    if "Weight %" in df.columns:
        return pd.to_numeric(df["Weight %"], errors="coerce").fillna(0.0) / 100.0
    if "Target %" in df.columns:
        return pd.to_numeric(df["Target %"], errors="coerce").fillna(0.0) / 100.0
    return pd.Series(dtype="float64")


def _render_target_allocation_with_tail_controls(
    *,
    title: str,
    df: pd.DataFrame,
    planning_capital: float,
    key_prefix: str,
    default_min_display_pct: float = 0.5,
) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    weights = _target_weight_fraction_series(df)
    if weights.empty:
        order_df, column_config = _prepare_target_order_table(df, planning_capital=planning_capital)
        st.subheader(title)
        st.dataframe(order_df, width="stretch", hide_index=True, column_config=column_config)
        return

    st.subheader(title)
    min_display_pct = st.slider(
        "Hide allocations smaller than (%)",
        min_value=0.0,
        max_value=2.0,
        value=float(default_min_display_pct),
        step=0.1,
        key=f"{key_prefix}_min_display_pct",
        help="Use this to collapse phased-in or phased-out dust positions created by turnover budgeting.",
    )
    min_display_weight = float(min_display_pct) / 100.0
    keep_mask = weights >= min_display_weight
    primary_df = df.loc[keep_mask].copy()
    tail_df = df.loc[~keep_mask].copy()

    total_positions = int((weights > 0).sum())
    displayed_positions = int(keep_mask.sum())
    dust_positions = int((~keep_mask).sum())
    displayed_gross = float(weights.loc[keep_mask].sum()) if displayed_positions else 0.0
    dust_gross = float(weights.loc[~keep_mask].sum()) if dust_positions else 0.0
    meaningful_1pct = int((weights >= 0.01).sum())
    meaningful_2pct = int((weights >= 0.02).sum())
    top5_gross = float(weights.sort_values(ascending=False).head(5).sum()) if total_positions else 0.0
    st.caption(
        f"Configured view: {displayed_positions}/{total_positions} rows shown at >= {min_display_pct:.1f}% each. "
        f"Displayed gross {displayed_gross:.1%}; hidden dust gross {dust_gross:.1%}."
    )
    st.caption(
        f"Concentration snapshot: {meaningful_1pct} names >= 1%, {meaningful_2pct} names >= 2%, "
        f"top 5 gross {top5_gross:.1%}."
    )

    if dust_positions > 0:
        st.info(
            "Tiny rows are usually phased exits/entries from turnover budgeting. "
            "They do not mean the strategy is intentionally targeting that many equal-conviction names."
        )

    display_df = primary_df.copy()
    if dust_positions > 0:
        summary_row: Dict[str, Any] = {}
        if "Sleeve" in display_df.columns:
            summary_row["Sleeve"] = "Residual"
        if "Source Profile" in display_df.columns:
            summary_row["Source Profile"] = "Phased residuals"
        summary_row["Symbol"] = f"Residual tail ({dust_positions} names)"
        if "Target Weight" in display_df.columns:
            summary_row["Target Weight"] = float(dust_gross)
        if "Target %" in display_df.columns:
            summary_row["Target %"] = float(dust_gross) * 100.0
        if "Weight %" in display_df.columns:
            summary_row["Weight %"] = float(dust_gross) * 100.0
        if "Last Close" in display_df.columns:
            summary_row["Last Close"] = float("nan")
        display_df = pd.concat([display_df, pd.DataFrame([summary_row])], ignore_index=True)

    order_df, column_config = _prepare_target_order_table(display_df, planning_capital=planning_capital)
    st.dataframe(order_df, width="stretch", hide_index=True, column_config=column_config)

    if dust_positions > 0:
        with st.expander(f"Show hidden small allocations ({dust_positions} rows, {dust_gross:.1%} gross)"):
            dust_order_df, dust_column_config = _prepare_target_order_table(tail_df, planning_capital=planning_capital)
            st.dataframe(dust_order_df, width="stretch", hide_index=True, column_config=dust_column_config)


def _render_rebalance_with_tail_controls(
    *,
    title: str,
    df: pd.DataFrame,
    planning_capital: float,
    key_prefix: str,
    default_min_delta_pct: float = 0.5,
) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    delta_weights = pd.to_numeric(df.get("Delta %"), errors="coerce").fillna(0.0).abs() / 100.0
    if delta_weights.empty:
        order_df, column_config = _prepare_rebalance_order_table(df, planning_capital=planning_capital)
        st.subheader(title)
        st.dataframe(order_df, width="stretch", hide_index=True, column_config=column_config)
        return

    st.subheader(title)
    min_delta_pct = st.slider(
        "Hide rebalance deltas smaller than (%)",
        min_value=0.0,
        max_value=2.0,
        value=float(default_min_delta_pct),
        step=0.1,
        key=f"{key_prefix}_min_delta_pct",
        help="Collapse tiny rebalance legs that come from phased turnover budgeting rather than a fresh high-conviction decision.",
    )
    min_delta_weight = float(min_delta_pct) / 100.0
    keep_mask = delta_weights >= min_delta_weight
    primary_df = df.loc[keep_mask].copy()
    tail_df = df.loc[~keep_mask].copy()

    total_rows = int((delta_weights > 0).sum())
    displayed_rows = int(keep_mask.sum())
    hidden_rows = int((~keep_mask).sum())
    displayed_gross = float(delta_weights.loc[keep_mask].sum()) if displayed_rows else 0.0
    hidden_gross = float(delta_weights.loc[~keep_mask].sum()) if hidden_rows else 0.0
    st.caption(
        f"Configured view: {displayed_rows}/{total_rows} rebalance rows shown at >= {min_delta_pct:.1f}% each. "
        f"Displayed rebalance gross {displayed_gross:.1%}; hidden tail gross {hidden_gross:.1%}."
    )
    if hidden_rows > 0:
        st.info(
            "Tiny rebalance rows are usually phased trims/adds from the turnover budget, not new full-conviction entries."
        )

    display_df = primary_df.copy()
    if hidden_rows > 0:
        summary_row: Dict[str, Any] = {}
        if "Sleeve" in display_df.columns:
            summary_row["Sleeve"] = "Residual"
        if "Source Profile" in display_df.columns:
            summary_row["Source Profile"] = "Phased residuals"
        summary_row["Symbol"] = f"Residual rebalance tail ({hidden_rows} rows)"
        if "Current %" in display_df.columns:
            summary_row["Current %"] = float("nan")
        if "Prev %" in display_df.columns:
            summary_row["Prev %"] = float("nan")
        if "Target %" in display_df.columns:
            summary_row["Target %"] = float("nan")
        summary_row["Delta %"] = float(hidden_gross) * 100.0
        summary_row["Action"] = "Mixed phased rebalance"
        if "Last Close" in display_df.columns:
            summary_row["Last Close"] = float("nan")
        display_df = pd.concat([display_df, pd.DataFrame([summary_row])], ignore_index=True)

    order_df, column_config = _prepare_rebalance_order_table(display_df, planning_capital=planning_capital)
    st.dataframe(order_df, width="stretch", hide_index=True, column_config=column_config)

    if hidden_rows > 0:
        with st.expander(f"Show hidden rebalance tail ({hidden_rows} rows, {hidden_gross:.1%} gross)"):
            dust_order_df, dust_column_config = _prepare_rebalance_order_table(tail_df, planning_capital=planning_capital)
            st.dataframe(dust_order_df, width="stretch", hide_index=True, column_config=dust_column_config)


def _prepare_rebalance_order_table(
    df: pd.DataFrame,
    *,
    planning_capital: float,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), {}

    out = df.copy()
    capital_label = _planning_capital_label(planning_capital)
    order_notional_col = f"Order $ @ {capital_label}"
    est_shares_col = f"Est. Shares @ {capital_label}"
    delta_pct = pd.to_numeric(out.get("Delta %"), errors="coerce").fillna(0.0).abs() / 100.0
    planning_px = pd.to_numeric(out.get("Last Close"), errors="coerce")

    out["Model Fill"] = "Next Open"
    out["Planning Px"] = planning_px
    out[order_notional_col] = delta_pct * float(planning_capital)
    out[est_shares_col] = [
        _estimate_shares_for_notional(notional, price)
        for notional, price in zip(out[order_notional_col], out["Planning Px"])
    ]

    order_cols = [col for col in ("Sleeve", "Source Profile", "Symbol", "Current %", "Prev %", "Target %", "Delta %", "Action") if col in out.columns]
    order_cols.extend(["Model Fill", "Planning Px", order_notional_col, est_shares_col])
    column_config = {
        "Current %": st.column_config.NumberColumn(format="%.2f%%"),
        "Prev %": st.column_config.NumberColumn(format="%.2f%%"),
        "Target %": st.column_config.NumberColumn(format="%.2f%%"),
        "Delta %": st.column_config.NumberColumn(format="%.2f%%"),
        "Planning Px": st.column_config.NumberColumn(format="$%.2f"),
        order_notional_col: st.column_config.NumberColumn(format="$%.0f"),
        est_shares_col: st.column_config.NumberColumn(format="%.1f"),
    }
    return out[order_cols], column_config


def _render_benchmark_execution_note(planning_capital: float) -> None:
    st.caption(
        "Execution model: signals are generated after the close and modeled as next-session open fills plus configured slippage. "
        f"Planning columns use the latest close as a sizing estimate on a {_planning_capital_label(planning_capital)} account basis. "
        "Actual shares can differ if the next open gaps."
    )


def _render_snapshot_resolution_note(snapshot: Mapping[str, Any], *, label: str) -> None:
    requested_end = str(snapshot.get("requested_end_date") or "").strip()
    resolved_end = str(snapshot.get("resolved_end_date") or "").strip()
    if requested_end and resolved_end and requested_end != resolved_end:
        st.caption(f"{label} live snapshot used the most recent valid market date: `{resolved_end}` (requested `{requested_end}`).")
    elif snapshot.get("used_lookback_fallback"):
        st.caption(f"{label} live snapshot expanded its lookback window to find the latest valid signal.")


def _resolve_live_snapshot(
    *,
    session_key: str,
    session_label_key: str,
    snapshot_file: str,
    profile_label: str,
    expected_end_date: str,
    builder,
    force_refresh: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if force_refresh:
        st.session_state.pop(session_key, None)
        st.session_state.pop(session_label_key, None)
        _clear_pickle_payload(snapshot_file)

    snapshot = st.session_state.get(session_key)
    cached_label = st.session_state.get(session_label_key)
    if cached_label == profile_label and _snapshot_is_current(snapshot, profile_label=profile_label, expected_end_date=expected_end_date):
        return dict(snapshot), "session"

    disk_snapshot = _read_pickle_payload(snapshot_file, None)
    if _snapshot_is_current(disk_snapshot, profile_label=profile_label, expected_end_date=expected_end_date):
        st.session_state[session_key] = disk_snapshot
        st.session_state[session_label_key] = profile_label
        return dict(disk_snapshot), "disk"

    try:
        built = builder()
    except Exception:
        if cached_label == profile_label and isinstance(snapshot, Mapping):
            return dict(snapshot), "stale-session"
        disk_label = str((disk_snapshot or {}).get("profile_label") or (disk_snapshot or {}).get("config_name") or "").strip()
        if isinstance(disk_snapshot, Mapping) and (not disk_label or disk_label == str(profile_label)):
            st.session_state[session_key] = disk_snapshot
            st.session_state[session_label_key] = profile_label
            return dict(disk_snapshot), "stale-disk"
        raise
    built = dict(built or {})
    built["profile_label"] = profile_label
    if "requested_end_date" not in built:
        built["requested_end_date"] = expected_end_date
    st.session_state[session_key] = built
    st.session_state[session_label_key] = profile_label
    _write_pickle_payload(snapshot_file, built)
    return built, "built"


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


def _load_etf_runner_helpers():
    from scripts.run_etf_rotation_walkforward import (
        _build_allocator_target_schedule,
        _build_residual_defensive_target_schedule,
        _build_rotation_scores,
        _schedule_row_weights,
        _download_yfinance_pack,
        _load_config,
        _normalize_symbol_list,
        _price_frame_from_data,
        _run_rotation_window,
        _run_rotation_window_with_targets,
    )

    return {
        "build_allocator_target_schedule": _build_allocator_target_schedule,
        "build_residual_defensive_target_schedule": _build_residual_defensive_target_schedule,
        "build_rotation_scores": _build_rotation_scores,
        "schedule_row_weights": _schedule_row_weights,
        "download_yfinance_pack": _download_yfinance_pack,
        "load_config": _load_config,
        "normalize_symbol_list": _normalize_symbol_list,
        "price_frame_from_data": _price_frame_from_data,
        "run_rotation_window": _run_rotation_window,
        "run_rotation_window_with_targets": _run_rotation_window_with_targets,
    }


def _etf_window_metrics(run: Mapping[str, Any]) -> Dict[str, float | int]:
    cagr_pct = float(_safe_float(run.get("cagr"), 0.0) * 100.0)
    dd_pct = float(_safe_float(run.get("max_drawdown_pct"), 0.0) * 100.0)
    audit = dict(run.get("audit_report") or {})
    return {
        "cagr_pct": cagr_pct,
        "max_dd_pct": dd_pct,
        "calmar": float(cagr_pct / dd_pct) if dd_pct > 0 else float("nan"),
        "avg_annual_turnover_pct": float(_safe_float(run.get("avg_annual_turnover_pct"), float("nan"))),
        "total_trades": int(run.get("total_trades", 0) or 0),
        "final_value": float(_safe_float(run.get("final_value"), 0.0)),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
    }


def _etf_config_symbols(
    cfg: Mapping[str, Any],
    normalize_symbol_list,
) -> List[str]:
    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    if isinstance(sleeves, Mapping) and sleeves:
        sleeve_symbols: List[str] = []
        for sleeve_cfg in sleeves.values():
            if isinstance(sleeve_cfg, Mapping):
                sleeve_symbols.extend(normalize_symbol_list(sleeve_cfg.get("symbols", [])))
        return normalize_symbol_list(sleeve_symbols)
    if defensive_cfg and bool(defensive_cfg.get("enabled", False)):
        return normalize_symbol_list(list(cfg.get("symbols", [])) + list(defensive_cfg.get("symbols", [])))
    return normalize_symbol_list(cfg.get("symbols", []))


@st.cache_data(ttl=900, show_spinner=False)
def _run_etf_benchmark(
    *,
    config_path: str,
    start_date: str,
    end_date: str,
    train_end_date: Optional[str] = None,
    holdout_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    helpers = _load_etf_runner_helpers()
    cfg = helpers["load_config"](Path(config_path).resolve())
    symbols = _etf_config_symbols(cfg, helpers["normalize_symbol_list"])
    global_symbols = helpers["normalize_symbol_list"](cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))
    requested = helpers["normalize_symbol_list"](list(symbols) + list(global_symbols))
    data = helpers["download_yfinance_pack"](requested, start_date=str(start_date), end_date=str(end_date))
    global_data = {sym: data[sym] for sym in global_symbols if sym in data}

    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    uses_allocator = isinstance(sleeves, Mapping) and bool(sleeves)
    uses_residual_defensive = defensive_cfg and bool(defensive_cfg.get("enabled", False))

    if uses_allocator:
        close_px, open_px, target_schedule, _ = helpers["build_allocator_target_schedule"](data, cfg)
        full_run = helpers["run_rotation_window_with_targets"](
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        run_cfg = cfg
    elif uses_residual_defensive:
        close_px, open_px, target_schedule = helpers["build_residual_defensive_target_schedule"](data, cfg, global_data)
        run_cfg = dict(cfg)
        run_cfg["market_regime"] = {"enabled": False}
        run_cfg["exposure_control"] = {}
        full_run = helpers["run_rotation_window_with_targets"](
            cfg=run_cfg,
            close_px=close_px,
            open_px=open_px,
            target_schedule=target_schedule,
            global_data=global_data,
            start_date=str(start_date),
            end_date=str(end_date),
        )
    else:
        close_px = helpers["price_frame_from_data"](data, symbols, "close")
        open_px = helpers["price_frame_from_data"](data, symbols, "open").reindex(close_px.index)
        scores = helpers["build_rotation_scores"](close_px, cfg)
        full_run = helpers["run_rotation_window"](
            cfg=cfg,
            close_px=close_px,
            open_px=open_px,
            scores=scores,
            global_data=global_data,
            start_date=str(start_date),
            end_date=str(end_date),
        )
        run_cfg = cfg

    payload: Dict[str, Any] = {
        "config_name": str(cfg.get("name", "") or os.path.basename(config_path)),
        "config_path": str(Path(config_path).resolve()),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "symbols": list(symbols),
        "loaded_symbols": [sym for sym in symbols if sym in data],
        "full": _etf_window_metrics(full_run),
        "full_run": full_run,
    }

    if train_end_date and holdout_start_date:
        if uses_allocator:
            train_run = helpers["run_rotation_window_with_targets"](
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=str(start_date),
                end_date=str(train_end_date),
            )
            holdout_run = helpers["run_rotation_window_with_targets"](
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=str(holdout_start_date),
                end_date=str(end_date),
            )
        elif uses_residual_defensive:
            train_run = helpers["run_rotation_window_with_targets"](
                cfg=run_cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=str(start_date),
                end_date=str(train_end_date),
            )
            holdout_run = helpers["run_rotation_window_with_targets"](
                cfg=run_cfg,
                close_px=close_px,
                open_px=open_px,
                target_schedule=target_schedule,
                global_data=global_data,
                start_date=str(holdout_start_date),
                end_date=str(end_date),
            )
        else:
            train_run = helpers["run_rotation_window"](
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                scores=scores,
                global_data=global_data,
                start_date=str(start_date),
                end_date=str(train_end_date),
            )
            holdout_run = helpers["run_rotation_window"](
                cfg=cfg,
                close_px=close_px,
                open_px=open_px,
                scores=scores,
                global_data=global_data,
                start_date=str(holdout_start_date),
                end_date=str(end_date),
            )

        payload["train_end_date"] = str(train_end_date)
        payload["holdout_start_date"] = str(holdout_start_date)
        payload["train"] = _etf_window_metrics(train_run)
        payload["holdout"] = _etf_window_metrics(holdout_run)
        payload["holdout_run"] = holdout_run

    return payload


def _read_json_payload(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _write_json_payload(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _persist_streamlit_export(file_name: str, payload_json: str) -> Dict[str, Any]:
    safe_name = Path(file_name).name or "benchmark_result.json"
    export_targets = {
        "workspace_path": (Path("exports") / "streamlit_results" / safe_name).resolve(),
        "downloads_path": Path.home() / "Downloads" / safe_name,
    }
    saved_paths: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for key, path in export_targets.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload_json, encoding="utf-8")
            saved_paths[key] = str(path)
        except Exception as exc:
            errors[key] = f"{path}: {exc}"
    return {"saved_paths": saved_paths, "errors": errors}


def _persist_streamlit_bytes(file_name: str, payload: bytes) -> Dict[str, Any]:
    safe_name = Path(file_name).name or "streamlit_export.bin"
    export_targets = {
        "workspace_path": (Path("exports") / "streamlit_results" / safe_name).resolve(),
        "downloads_path": Path.home() / "Downloads" / safe_name,
    }
    saved_paths: Dict[str, str] = {}
    errors: Dict[str, str] = {}
    for key, path in export_targets.items():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            saved_paths[key] = str(path)
        except Exception as exc:
            errors[key] = f"{path}: {exc}"
    return {"saved_paths": saved_paths, "errors": errors}


def _render_streamlit_export_status(export_meta: Mapping[str, Any]) -> None:
    saved_paths = dict(export_meta.get("saved_paths") or {})
    workspace_path = saved_paths.get("workspace_path")
    downloads_path = saved_paths.get("downloads_path")
    if workspace_path:
        st.caption(f"Saved workspace copy: `{workspace_path}`")
    if downloads_path:
        st.caption(f"Saved Downloads copy: `{downloads_path}`")
    errors = dict(export_meta.get("errors") or {})
    if errors:
        st.warning(
            "Automatic JSON export failed for: "
            + " | ".join(str(message) for message in errors.values())
        )


def _read_pickle_payload(path: str, default: Any) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return default


def _write_pickle_payload(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _clear_pickle_payload(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _latest_completed_market_session_date() -> str:
    now_ts = pd.Timestamp.now(tz=ZoneInfo("America/New_York"))
    try:
        import pandas_market_calendars as mcal  # type: ignore

        cal = mcal.get_calendar("NYSE")
        sched = cal.schedule(
            start_date=(now_ts - pd.Timedelta(days=14)).date().isoformat(),
            end_date=now_ts.date().isoformat(),
        )
        if not sched.empty:
            market_close = sched["market_close"]
            eligible = market_close[market_close <= now_ts.tz_convert("UTC")]
            if len(eligible) > 0:
                return pd.Timestamp(eligible.index[-1]).tz_localize(None).date().isoformat()
            return pd.Timestamp(sched.index[0]).tz_localize(None).date().isoformat()
    except Exception:
        pass

    effective = now_ts.tz_localize(None).normalize()
    if now_ts.weekday() >= 5:
        while effective.weekday() >= 5:
            effective = (effective - pd.Timedelta(days=1)).normalize()
        return effective.date().isoformat()
    if now_ts.hour < 16 or (now_ts.hour == 16 and now_ts.minute < 15):
        effective = (effective - pd.tseries.offsets.BDay(1)).normalize()
    return effective.date().isoformat()


def _snapshot_is_current(
    snapshot: Mapping[str, Any] | None,
    *,
    profile_label: str,
    expected_end_date: str,
) -> bool:
    if not isinstance(snapshot, Mapping):
        return False
    cached_label = str(snapshot.get("profile_label") or snapshot.get("config_name") or "").strip()
    if cached_label and cached_label != str(profile_label):
        return False
    requested_end = str(snapshot.get("requested_end_date") or "").strip()
    if requested_end:
        return requested_end == str(expected_end_date)
    resolved_end = str(snapshot.get("resolved_end_date") or "").strip()
    if resolved_end:
        return resolved_end == str(expected_end_date)
    latest_signal = str(snapshot.get("latest_signal_date") or "").strip()
    return latest_signal == str(expected_end_date)


@st.cache_data(ttl=900, show_spinner=False)
def _build_etf_live_snapshot(
    *,
    config_path: str,
    end_date: Optional[str] = None,
    lookback_days: int = 900,
) -> Dict[str, Any]:
    helpers = _load_etf_runner_helpers()
    cfg = helpers["load_config"](Path(config_path).resolve())
    requested_end_ts = pd.Timestamp(end_date or pd.Timestamp.utcnow().tz_localize(None).date().isoformat()).normalize()
    end_ts = requested_end_ts
    start_ts = (end_ts - pd.Timedelta(days=int(lookback_days))).normalize()

    symbols = _etf_config_symbols(cfg, helpers["normalize_symbol_list"])
    global_symbols = helpers["normalize_symbol_list"](cfg.get("global_symbols", ["SPY", "VIX", "HYG", "LQD"]))
    requested = helpers["normalize_symbol_list"](list(symbols) + list(global_symbols))
    data = helpers["download_yfinance_pack"](
        requested,
        start_date=start_ts.date().isoformat(),
        end_date=end_ts.date().isoformat(),
    )
    global_data = {sym: data[sym] for sym in global_symbols if sym in data}

    sleeves = cfg.get("sleeves")
    defensive_cfg = dict(cfg.get("defensive_sleeve") or {})
    uses_allocator = isinstance(sleeves, Mapping) and bool(sleeves)
    uses_residual_defensive = defensive_cfg and bool(defensive_cfg.get("enabled", False))

    if uses_allocator:
        close_px, _open_px, target_schedule, allocator_state = helpers["build_allocator_target_schedule"](data, cfg)
    elif uses_residual_defensive:
        close_px, _open_px, target_schedule = helpers["build_residual_defensive_target_schedule"](data, cfg, global_data)
        allocator_state = None
    else:
        raise ValueError("ETF live snapshot currently supports allocator and residual-defensive ETF configs only.")

    signal_schedule = target_schedule.dropna(how="all")
    if signal_schedule.empty:
        raise RuntimeError("ETF target schedule produced no active signal dates.")

    latest_signal_dt = pd.Timestamp(signal_schedule.index[-1]).tz_localize(None).normalize()
    previous_signal_dt = (
        pd.Timestamp(signal_schedule.index[-2]).tz_localize(None).normalize()
        if len(signal_schedule.index) > 1
        else None
    )
    latest_weights = helpers["schedule_row_weights"](target_schedule, latest_signal_dt)
    previous_weights = helpers["schedule_row_weights"](target_schedule, previous_signal_dt) if previous_signal_dt is not None else {}
    if not latest_weights:
        raise RuntimeError("Latest ETF signal has no target weights.")

    latest_close_row = {}
    if latest_signal_dt in close_px.index:
        latest_close_row = pd.to_numeric(close_px.loc[latest_signal_dt], errors="coerce").dropna().to_dict()

    current_symbols = sorted(set(latest_weights.keys()) | set(previous_weights.keys()))
    rebalance_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    aggressive_symbols = {str(sym).upper() for sym in cfg.get("symbols", [])}
    defensive_symbols = {str(sym).upper() for sym in defensive_cfg.get("symbols", [])}
    for sym in current_symbols:
        latest_wt = float(latest_weights.get(sym, 0.0) or 0.0)
        prev_wt = float(previous_weights.get(sym, 0.0) or 0.0)
        delta = latest_wt - prev_wt
        sleeve = "ETF"
        if str(sym).upper() in aggressive_symbols:
            sleeve = "Aggressive"
        elif str(sym).upper() in defensive_symbols:
            sleeve = "Defensive"
        last_close = latest_close_row.get(sym)
        if latest_wt > 0.0:
            target_rows.append(
                {
                    "Sleeve": sleeve,
                    "Symbol": str(sym),
                    "Target Weight": latest_wt,
                    "Target %": latest_wt * 100.0,
                    "Last Close": float(last_close) if last_close is not None else float("nan"),
                    "Notional per $100k": latest_wt * 100000.0,
                }
            )
        if abs(delta) > 1e-9:
            rebalance_rows.append(
                {
                    "Sleeve": sleeve,
                    "Symbol": str(sym),
                    "Prev %": prev_wt * 100.0,
                    "Target %": latest_wt * 100.0,
                    "Delta %": delta * 100.0,
                    "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                    "Last Close": float(last_close) if last_close is not None else float("nan"),
                }
            )

    target_df = pd.DataFrame(target_rows).sort_values(["Sleeve", "Target Weight"], ascending=[True, False])
    rebalance_df = pd.DataFrame(rebalance_rows)
    if not rebalance_df.empty:
        rebalance_df = rebalance_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

    allocator_state_value = None
    if isinstance(allocator_state, pd.Series) and latest_signal_dt in allocator_state.index:
        allocator_state_value = str(allocator_state.loc[latest_signal_dt])

    return {
        "config_name": str(cfg.get("name", "") or os.path.basename(config_path)),
        "config_path": str(Path(config_path).resolve()),
        "latest_signal_date": latest_signal_dt.date().isoformat(),
        "previous_signal_date": previous_signal_dt.date().isoformat() if previous_signal_dt is not None else None,
        "latest_target_weights": latest_weights,
        "previous_target_weights": previous_weights,
        "target_df": target_df,
        "rebalance_df": rebalance_df,
        "signal_count": int(len(signal_schedule)),
        "max_gross_exposure_pct": float(sum(float(v) for v in latest_weights.values())),
        "allocator_state": allocator_state_value,
        "symbols": symbols,
        "loaded_symbols": [sym for sym in symbols if sym in data],
        "requested_end_date": requested_end_ts.date().isoformat(),
        "resolved_end_date": latest_signal_dt.date().isoformat(),
        "built_at_utc": _utc_now_iso(),
    }


def _load_etf_paper_state() -> Dict[str, Any]:
    default_state = {
        "profile_label": ETF_FROZEN_DEFAULT_LABEL,
        "adopted_signal_date": None,
        "holdings": {},
        "updated_at_utc": None,
    }
    payload = _read_json_payload(ETF_PAPER_STATE_FILE, default_state)
    if not isinstance(payload, dict):
        return dict(default_state)
    state = dict(default_state)
    state.update(payload)
    if not isinstance(state.get("holdings"), dict):
        state["holdings"] = {}
    if str(state.get("profile_label") or "") not in ETF_FROZEN_BENCHMARKS:
        state["profile_label"] = ETF_FROZEN_DEFAULT_LABEL
    return state


def _save_etf_paper_state(state: Mapping[str, Any]) -> None:
    payload = {
        "profile_label": str(state.get("profile_label", ETF_FROZEN_DEFAULT_LABEL) or ETF_FROZEN_DEFAULT_LABEL),
        "adopted_signal_date": state.get("adopted_signal_date"),
        "holdings": {
            str(sym): float(weight)
            for sym, weight in dict(state.get("holdings") or {}).items()
            if _safe_float(weight, 0.0) > 0.0
        },
        "updated_at_utc": _utc_now_iso(),
    }
    _write_json_payload(ETF_PAPER_STATE_FILE, payload)


def _render_etf_live_screener(etf_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "🏦 ETF Live Screener",
        "Review the latest frozen ETF allocation, current rebalance deltas, and next-session target weights.",
    )
    st.caption(
        "This path uses the same frozen ETF strategy family as Backtest and Simulator. "
        "Signals are built from end-of-day data and are intended for next-session execution only."
    )
    st.caption("Use this view to review the latest target weights before the next trading session.")
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh ETF Snapshot", type="primary", key="refresh_etf_live_snapshot")
    if refresh_requested:
        _build_etf_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        with st.spinner("Building ETF live snapshot..."):
            snapshot, snapshot_source = _resolve_live_snapshot(
                session_key="etf_live_snapshot",
                session_label_key="etf_live_snapshot_label",
                snapshot_file=ETF_LIVE_SNAPSHOT_FILE,
                profile_label=etf_profile_label,
                expected_end_date=expected_end_date,
                builder=lambda: _build_etf_live_snapshot(
                    config_path=ETF_FROZEN_BENCHMARKS[etf_profile_label],
                    end_date=expected_end_date,
                ),
                force_refresh=refresh_requested,
            )
    except Exception as exc:
        st.error(f"ETF benchmark snapshot unavailable: {exc}")
        return
    if snapshot_source in {"session", "disk"}:
        st.caption(f"Loaded cached ETF snapshot for `{expected_end_date}`.")
    elif snapshot_source in {"stale-session", "stale-disk"}:
        resolved = str(snapshot.get("resolved_end_date") or snapshot.get("latest_signal_date") or "latest available")
        st.warning(f"Using the last successful ETF snapshot from `{resolved}` while a fresh rebuild is unavailable.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ETF Profile", _display_profile_label(etf_profile_label))
    m2.metric("Signal Date", str(snapshot.get("latest_signal_date", "-")))
    m3.metric("Gross Exposure", f"{float(snapshot.get('max_gross_exposure_pct', 0.0)):.1%}")
    m4.metric("Active ETFs", len(dict(snapshot.get("latest_target_weights") or {})))
    if snapshot.get("allocator_state"):
        st.caption(f"Allocator state: `{snapshot.get('allocator_state')}`")

    target_df = snapshot.get("target_df")
    if isinstance(target_df, pd.DataFrame) and not target_df.empty:
        _render_target_allocation_with_tail_controls(
            title="Current Target Allocation",
            df=target_df,
            planning_capital=planning_capital,
            key_prefix="etf_live_target",
            default_min_display_pct=1.0,
        )

    rebalance_df = snapshot.get("rebalance_df")
    if isinstance(rebalance_df, pd.DataFrame) and not rebalance_df.empty:
        _render_rebalance_with_tail_controls(
            title="Rebalance Delta vs Previous Signal",
            df=rebalance_df,
            planning_capital=planning_capital,
            key_prefix="etf_live_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.info("No allocation changes vs the previous ETF signal.")


def _render_etf_simulator(etf_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "🎮 ETF Paper Allocator",
        "Track the frozen ETF strategy as a paper allocation book using the same target schedule as Live Screener and Backtest.",
    )
    st.caption(
        "This simulator stores adopted target weights only. It does not run the stock paper-trader engine."
    )
    st.caption("Use Adopt Latest only after reviewing the target and rebalance delta below.")
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh ETF Recommendation", type="primary", key="refresh_etf_sim_snapshot")
    if refresh_requested:
        _build_etf_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        with st.spinner("Refreshing ETF simulator recommendation..."):
            snapshot, _ = _resolve_live_snapshot(
                session_key="etf_sim_snapshot",
                session_label_key="etf_sim_snapshot_label",
                snapshot_file=ETF_LIVE_SNAPSHOT_FILE,
                profile_label=etf_profile_label,
                expected_end_date=expected_end_date,
                builder=lambda: _build_etf_live_snapshot(
                    config_path=ETF_FROZEN_BENCHMARKS[etf_profile_label],
                    end_date=expected_end_date,
                ),
                force_refresh=refresh_requested,
            )
    except Exception as exc:
        st.error(f"ETF benchmark recommendation unavailable: {exc}")
        return

    state = _load_etf_paper_state()
    current_holdings = {
        str(sym): float(weight)
        for sym, weight in dict(state.get("holdings") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    latest_weights = {
        str(sym): float(weight)
        for sym, weight in dict(snapshot.get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stored Profile", _display_profile_label(str(state.get("profile_label") or ETF_FROZEN_DEFAULT_LABEL)))
    m2.metric("Adopted Signal", str(state.get("adopted_signal_date") or "None"))
    m3.metric("Current Gross", f"{sum(current_holdings.values()):.1%}")
    m4.metric("Target Gross", f"{sum(latest_weights.values()):.1%}")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Adopt Latest ETF Allocation", type="primary", key="adopt_latest_etf_alloc"):
            _save_etf_paper_state(
                {
                    "profile_label": etf_profile_label,
                    "adopted_signal_date": snapshot.get("latest_signal_date"),
                    "holdings": latest_weights,
                }
            )
            st.success("ETF simulator allocation updated.")
            st.rerun()
    with action_col2:
        if st.button("Reset ETF Simulator State", key="reset_etf_sim_state"):
            _save_etf_paper_state(
                {
                    "profile_label": etf_profile_label,
                    "adopted_signal_date": None,
                    "holdings": {},
                }
            )
            st.success("ETF simulator state reset.")
            st.rerun()

    current_rows = [
        {"Symbol": sym, "Weight %": float(weight) * 100.0}
        for sym, weight in sorted(current_holdings.items())
    ]
    target_rows = [
        {"Symbol": sym, "Weight %": float(weight) * 100.0}
        for sym, weight in sorted(latest_weights.items())
    ]
    close_map = _extract_close_map_from_snapshot(snapshot)
    for row in current_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    for row in target_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    delta_rows: List[Dict[str, Any]] = []
    for sym in sorted(set(current_holdings.keys()) | set(latest_weights.keys())):
        curr = float(current_holdings.get(sym, 0.0) or 0.0)
        tgt = float(latest_weights.get(sym, 0.0) or 0.0)
        delta = tgt - curr
        if abs(delta) <= 1e-9:
            continue
        delta_rows.append(
            {
                "Symbol": sym,
                "Current %": curr * 100.0,
                "Target %": tgt * 100.0,
                "Delta %": delta * 100.0,
                "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                "Last Close": float(close_map.get(str(sym), float("nan"))),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

    c1, c2 = st.columns(2)
    with c1:
        if current_rows:
            _render_target_allocation_with_tail_controls(
                title="Stored ETF Allocation",
                df=pd.DataFrame(current_rows),
                planning_capital=planning_capital,
                key_prefix="etf_sim_current",
                default_min_display_pct=1.0,
            )
        else:
            st.subheader("Stored ETF Allocation")
            st.info("No ETF allocation stored yet.")
    with c2:
        if target_rows:
            _render_target_allocation_with_tail_controls(
                title="Latest ETF Target",
                df=pd.DataFrame(target_rows),
                planning_capital=planning_capital,
                key_prefix="etf_sim_target",
                default_min_display_pct=1.0,
            )
        else:
            st.subheader("Latest ETF Target")
            st.warning("Latest ETF target is empty.")

    if not delta_df.empty:
        _render_rebalance_with_tail_controls(
            title="Required Rebalance",
            df=delta_df,
            planning_capital=planning_capital,
            key_prefix="etf_sim_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.subheader("Required Rebalance")
        st.success("Stored ETF allocation already matches the latest target.")


def _render_etf_benchmark_lab(default_end_date: str, *, etf_profile_label: str) -> None:
    st.markdown("### 🏦 ETF Benchmark Lab")
    st.caption(
        "Run the frozen ETF benchmark outside the generic stock-strategy engine. "
        "This is the current validated winner and the clean benchmark path."
    )
    st.caption("Blind Holdout is the validation view. Full Sample is the long-run context view.")
    st.info("This ETF workspace is not the 30%+ CAGR lane. That result belongs to the Stock Benchmark workspace.")
    _render_benchmark_execution_note(_current_benchmark_planning_capital())
    bench_m1, bench_m2, bench_m3 = st.columns(3)
    bench_m1.metric("Frozen Holdout CAGR", "22.04%")
    bench_m2.metric("Frozen Holdout Max DD", "25.45%")
    bench_m3.metric("Frozen Config", "Residual Defensive Calmar")

    st.info(f"Using ETF profile from the sidebar: **{_display_profile_label(etf_profile_label)}**")
    etf_eval_mode = st.radio(
        "ETF Evaluation Mode",
        ["Blind Holdout", "Full Sample"],
        horizontal=True,
        key="etf_benchmark_eval_mode",
    )

    if etf_eval_mode == "Blind Holdout":
        c1, c2, c3 = st.columns(3)
        with c1:
            etf_start = st.date_input(
                "ETF Start",
                value=pd.Timestamp("2011-01-01").date(),
                key="etf_holdout_start_base",
            )
        with c2:
            etf_train_end = st.date_input(
                "Train End",
                value=pd.Timestamp("2020-12-31").date(),
                key="etf_holdout_train_end",
            )
        with c3:
            etf_holdout_end = st.date_input(
                "Holdout End",
                value=pd.Timestamp(default_end_date).date(),
                key="etf_holdout_end",
            )
        etf_holdout_start = (pd.Timestamp(etf_train_end) + pd.Timedelta(days=1)).date()
        st.caption(f"Holdout starts automatically on `{etf_holdout_start.isoformat()}`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            etf_start = st.date_input(
                "ETF Start",
                value=pd.Timestamp("2011-01-01").date(),
                key="etf_full_start",
            )
        with c2:
            etf_holdout_end = st.date_input(
                "ETF End",
                value=pd.Timestamp(default_end_date).date(),
                key="etf_full_end",
            )
        etf_train_end = None
        etf_holdout_start = None

    run_etf_benchmark_btn = st.button(
        "Run ETF Benchmark",
        type="primary",
        key="run_etf_benchmark",
    )

    if run_etf_benchmark_btn:
        try:
            config_path = ETF_FROZEN_BENCHMARKS[etf_profile_label]
            start_str = pd.Timestamp(etf_start).date().isoformat()
            end_str = pd.Timestamp(etf_holdout_end).date().isoformat()
            train_end_str = pd.Timestamp(etf_train_end).date().isoformat() if etf_train_end is not None else None
            holdout_start_str = (
                pd.Timestamp(etf_holdout_start).date().isoformat() if etf_holdout_start is not None else None
            )
            if pd.Timestamp(start_str) >= pd.Timestamp(end_str):
                raise ValueError("ETF benchmark start date must be before end date.")
            if train_end_str and holdout_start_str:
                if pd.Timestamp(start_str) > pd.Timestamp(train_end_str):
                    raise ValueError("ETF benchmark train start must be on or before the train end date.")
                if pd.Timestamp(holdout_start_str) > pd.Timestamp(end_str):
                    raise ValueError("ETF benchmark holdout start must be on or before the holdout end date.")
            with st.spinner("Running ETF benchmark..."):
                etf_result = _run_etf_benchmark(
                    config_path=config_path,
                    start_date=start_str,
                    end_date=end_str,
                    train_end_date=train_end_str,
                    holdout_start_date=holdout_start_str,
                )
            etf_result["profile_label"] = etf_profile_label
            etf_result["evaluation_mode"] = etf_eval_mode
            st.session_state.etf_benchmark_result = etf_result
        except Exception as e:
            st.error(f"ETF benchmark run failed: {e}")

    etf_result = st.session_state.get("etf_benchmark_result")
    if not isinstance(etf_result, dict):
        return

    st.markdown("---")
    config_label = os.path.basename(str(etf_result.get("config_path", "")))
    st.caption(
        f"ETF benchmark result: `{etf_result.get('profile_label', ETF_FROZEN_DEFAULT_LABEL)}` "
        f"using `{config_label}`"
    )
    if "holdout" in etf_result and "train" in etf_result:
        train = dict(etf_result.get("train") or {})
        holdout = dict(etf_result.get("holdout") or {})
        col_train, col_hold = st.columns(2)
        with col_train:
            st.subheader("Train")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("CAGR", f"{float(train.get('cagr_pct', 0.0)):.2f}%")
            t2.metric("Max DD", f"{float(train.get('max_dd_pct', 0.0)):.2f}%")
            t3.metric("Calmar", f"{float(train.get('calmar', float('nan'))):.2f}")
            t4.metric("Trades", int(train.get("total_trades", 0) or 0))
        with col_hold:
            st.subheader("Holdout")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("CAGR", f"{float(holdout.get('cagr_pct', 0.0)):.2f}%")
            h2.metric("Max DD", f"{float(holdout.get('max_dd_pct', 0.0)):.2f}%")
            h3.metric("Calmar", f"{float(holdout.get('calmar', float('nan'))):.2f}")
            h4.metric("Trades", int(holdout.get("total_trades", 0) or 0))
            st.caption(
                f"Same-day open entries: {int(holdout.get('same_day_open_entries', 0) or 0)} | "
                f"Max gross: {float(holdout.get('max_gross_exposure_pct', 0.0) or 0.0):.1%}"
            )
        holdout_curve = normalize_equity_curve_df((etf_result.get("holdout_run") or {}).get("equity_curve", []))
        if not holdout_curve.empty:
            st.line_chart(holdout_curve.set_index("Date")["Equity"])
    else:
        full = dict(etf_result.get("full") or {})
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("CAGR", f"{float(full.get('cagr_pct', 0.0)):.2f}%")
        m2.metric("Max DD", f"{float(full.get('max_dd_pct', 0.0)):.2f}%")
        m3.metric("Calmar", f"{float(full.get('calmar', float('nan'))):.2f}")
        m4.metric("Trades", int(full.get("total_trades", 0) or 0))
        m5.metric("Turnover", f"{float(full.get('avg_annual_turnover_pct', float('nan'))):.1f}%")
        st.caption(
            f"Same-day open entries: {int(full.get('same_day_open_entries', 0) or 0)} | "
            f"Max gross: {float(full.get('max_gross_exposure_pct', 0.0) or 0.0):.1%}"
        )
        full_curve = normalize_equity_curve_df((etf_result.get("full_run") or {}).get("equity_curve", []))
        if not full_curve.empty:
            st.line_chart(full_curve.set_index("Date")["Equity"])

    payload_json = json.dumps(etf_result, indent=2, default=str)
    export_meta = _persist_streamlit_export("etf_benchmark_result.json", payload_json)
    st.download_button(
        label="📥 Export ETF Benchmark Result (JSON)",
        data=payload_json.encode("utf-8"),
        file_name="etf_benchmark_result.json",
        mime="application/json",
        key="dl_etf_benchmark_json",
    )
    _render_streamlit_export_status(export_meta)


def _load_stock_benchmark_helpers():
    from scripts.evaluate_smid_pullback_holdout import (
        DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
        DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
        DEFAULT_MIN_UNIVERSE_COVERAGE,
        _coverage_gate_failures,
        _coverage_ratio,
        build_smid_pullback_context,
        evaluate_smid_pullback_configs_on_context,
    )
    from scripts.run_smid_pullback_walkforward import (
        _extract_feature_arrays,
        _build_smid_pullback_scores,
        _load_config,
        _run_window,
        _slice_prices,
    )
    from scripts.run_factor_walkforward import _daily_membership_price_coverage

    return {
        "default_min_universe_coverage": DEFAULT_MIN_UNIVERSE_COVERAGE,
        "default_min_daily_membership_coverage": DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE,
        "default_min_daily_membership_coverage_p10": DEFAULT_MIN_DAILY_MEMBERSHIP_COVERAGE_P10,
        "coverage_gate_failures": _coverage_gate_failures,
        "coverage_ratio": _coverage_ratio,
        "build_context": build_smid_pullback_context,
        "evaluate_on_context": evaluate_smid_pullback_configs_on_context,
        "extract_features": _extract_feature_arrays,
        "build_scores": _build_smid_pullback_scores,
        "load_config": _load_config,
        "run_window": _run_window,
        "slice_prices": _slice_prices,
        "daily_membership_coverage": _daily_membership_price_coverage,
    }


def _stock_window_metrics(run: Mapping[str, Any]) -> Dict[str, float | int]:
    cagr_pct = float(_safe_float(run.get("cagr"), 0.0) * 100.0)
    dd_pct = float(_safe_float(run.get("max_drawdown_pct"), 0.0) * 100.0)
    audit = dict(run.get("audit_report") or {})
    return {
        "cagr_pct": cagr_pct,
        "max_dd_pct": dd_pct,
        "calmar": float(cagr_pct / dd_pct) if dd_pct > 0 else float("nan"),
        "avg_annual_turnover_pct": float(_safe_float(run.get("avg_annual_turnover_pct"), float("nan"))),
        "total_trades": int(run.get("total_trades", 0) or 0),
        "final_value": float(_safe_float(run.get("final_value"), 0.0)),
        "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
        "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
    }


@st.cache_data(ttl=900, show_spinner=False)
def _run_stock_benchmark(
    *,
    config_path: str,
    start_date: str,
    end_date: str,
    train_end_date: Optional[str] = None,
    holdout_start_date: Optional[str] = None,
    min_universe_coverage: Optional[float] = None,
    min_daily_membership_coverage: Optional[float] = None,
    min_daily_membership_coverage_p10: Optional[float] = None,
) -> Dict[str, Any]:
    helpers = _load_stock_benchmark_helpers()
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    days = max(900, int((end_ts - start_ts).days) + 320)
    min_universe = float(
        helpers["default_min_universe_coverage"]
        if min_universe_coverage is None
        else min_universe_coverage
    )
    min_daily = float(
        helpers["default_min_daily_membership_coverage"]
        if min_daily_membership_coverage is None
        else min_daily_membership_coverage
    )
    min_daily_p10 = float(
        helpers["default_min_daily_membership_coverage_p10"]
        if min_daily_membership_coverage_p10 is None
        else min_daily_membership_coverage_p10
    )

    context = helpers["build_context"](
        universe="RUSSELL3000",
        start_date=str(start_ts.date().isoformat()),
        end_date=str(end_ts.date().isoformat()),
        days=days,
        config_paths=[config_path],
    )
    coverage_ratio = helpers["coverage_ratio"](
        list(context.get("requested_symbols") or []),
        list(context.get("loaded_symbols") or []),
    )
    coverage_stats = dict(context.get("coverage_stats") or {})
    coverage_failures = helpers["coverage_gate_failures"](
        coverage_ratio=float(coverage_ratio),
        coverage_stats=coverage_stats,
        min_universe_coverage=min_universe,
        min_daily_membership_coverage=min_daily,
        min_daily_membership_coverage_p10=min_daily_p10,
    )
    if coverage_failures:
        raise RuntimeError("Stock benchmark coverage gate failed: " + "; ".join(str(x) for x in coverage_failures))

    if train_end_date and holdout_start_date:
        payload = helpers["evaluate_on_context"](
            context,
            config_paths=[config_path],
            universe="RUSSELL3000",
            train_start_date=str(start_date),
            train_end_date=str(train_end_date),
            holdout_start_date=str(holdout_start_date),
            holdout_end_date=str(end_date),
            min_universe_coverage=min_universe,
            min_daily_membership_coverage=min_daily,
            min_daily_membership_coverage_p10=min_daily_p10,
            include_runs=True,
        )
        if not payload.get("coverage_gate_passed", False):
            raise RuntimeError("Stock benchmark coverage gate failed.")
        reports = list(payload.get("reports") or [])
        if not reports:
            raise RuntimeError("Stock benchmark evaluator produced no reports.")
        report = dict(reports[0])
        return {
            "config_name": str(report.get("strategy_name", "") or os.path.basename(config_path)),
            "config_path": str(Path(config_path).resolve()),
            "start_date": str(start_date),
            "end_date": str(end_date),
            "train_end_date": str(train_end_date),
            "holdout_start_date": str(holdout_start_date),
            "coverage_gate_passed": True,
            "coverage_gate_failures": [],
            "coverage_ratio": float(report.get("coverage_ratio", coverage_ratio)),
            "daily_membership_price_coverage": report.get("daily_membership_price_coverage"),
            "requested_symbols": int(report.get("requested_symbols", 0) or 0),
            "loaded_symbols": int(report.get("loaded_symbols", 0) or 0),
            "prepared_symbols": int(report.get("prepared_symbols", 0) or 0),
            "active_scored_symbols": int(report.get("active_scored_symbols", 0) or 0),
            "train": report.get("train"),
            "holdout": report.get("holdout"),
            "train_run": report.get("train_run"),
            "holdout_run": report.get("holdout_run"),
        }

    cfg = helpers["load_config"](Path(config_path).resolve())
    scores = helpers["build_scores"](context["features"], cfg)
    if scores.empty:
        raise RuntimeError("Stock benchmark score builder produced no active symbols.")
    price_frames = helpers["slice_prices"](context["features"], ["open", "close"], list(scores.columns))
    full_run = helpers["run_window"](
        cfg=cfg,
        prices_close=price_frames["close"],
        prices_open=price_frames["open"],
        scores=scores,
        global_data=dict(context.get("global_data") or {}),
        start_date=str(start_date),
        end_date=str(end_date),
    )
    return {
        "config_name": str(cfg.get("name", "") or os.path.basename(config_path)),
        "config_path": str(Path(config_path).resolve()),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "coverage_gate_passed": True,
        "coverage_gate_failures": [],
        "coverage_ratio": float(coverage_ratio),
        "daily_membership_price_coverage": coverage_stats,
        "requested_symbols": int(len(list(context.get("requested_symbols") or []))),
        "loaded_symbols": int(len(list(context.get("loaded_symbols") or []))),
        "prepared_symbols": int(len(getattr(context.get("prepared"), "enriched", {}) or {})),
        "active_scored_symbols": int(len(scores.columns)),
        "full": _stock_window_metrics(full_run),
        "full_run": full_run,
    }


@st.cache_data(ttl=900, show_spinner=False)
def _load_hybrid_benchmark_helpers():
    from scripts.evaluate_hybrid_benchmark_holdout import _blend_equity_series, _metrics_from_equity

    return {
        "blend_equity_series": _blend_equity_series,
        "metrics_from_equity": _metrics_from_equity,
    }


def _hybrid_profile_spec(profile_label: str) -> Dict[str, Any]:
    raw = dict(HYBRID_BENCHMARK_CANDIDATES[profile_label])
    etf_weight = float(raw.get("etf_weight", 0.5) or 0.5)
    etf_weight = max(0.0, min(etf_weight, 1.0))
    raw["etf_weight"] = etf_weight
    raw["stock_weight"] = float(1.0 - etf_weight)
    return raw


def _equity_curve_series(run: Mapping[str, Any]) -> pd.Series:
    equity_curve = normalize_equity_curve_df((run or {}).get("equity_curve", []))
    if equity_curve.empty:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime(equity_curve["Date"], errors="coerce")
    vals = pd.to_numeric(equity_curve["Equity"], errors="coerce")
    ser = pd.Series(vals.values, index=idx, dtype="float64").dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser


def _series_to_equity_curve(ser: pd.Series) -> List[Dict[str, Any]]:
    if ser.empty:
        return []
    out: List[Dict[str, Any]] = []
    for dt, val in ser.items():
        ts = pd.Timestamp(dt).tz_localize(None)
        if pd.isna(ts) or pd.isna(val):
            continue
        out.append({"Date": ts.date().isoformat(), "Equity": float(val)})
    return out


def _blend_weight_maps(
    etf_weights: Mapping[str, Any],
    stock_weights: Mapping[str, Any],
    *,
    etf_weight: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    stock_weight = float(1.0 - etf_weight)
    for sym, weight in dict(etf_weights or {}).items():
        val = float(etf_weight) * float(_safe_float(weight, 0.0))
        if val > 0.0:
            out[str(sym)] = out.get(str(sym), 0.0) + val
    for sym, weight in dict(stock_weights or {}).items():
        val = stock_weight * float(_safe_float(weight, 0.0))
        if val > 0.0:
            out[str(sym)] = out.get(str(sym), 0.0) + val
    return {sym: float(weight) for sym, weight in out.items() if float(weight) > 0.0}


def _describe_signal_pair(etf_signal_date: Any, stock_signal_date: Any) -> str:
    etf_str = str(etf_signal_date or "")
    stock_str = str(stock_signal_date or "")
    if etf_str and stock_str and etf_str == stock_str:
        return etf_str
    if etf_str and stock_str:
        return f"ETF {etf_str} | Stock {stock_str}"
    return etf_str or stock_str or "-"


def _hybrid_window_summary(
    *,
    metrics: Mapping[str, Any],
    etf_metrics: Mapping[str, Any],
    stock_metrics: Mapping[str, Any],
    etf_weight: float,
) -> Dict[str, Any]:
    etf_turn = float(_safe_float(etf_metrics.get("avg_annual_turnover_pct"), float("nan")))
    stock_turn = float(_safe_float(stock_metrics.get("avg_annual_turnover_pct"), float("nan")))
    if math.isnan(etf_turn) and math.isnan(stock_turn):
        blended_turn = float("nan")
    elif math.isnan(etf_turn):
        blended_turn = float(1.0 - etf_weight) * stock_turn
    elif math.isnan(stock_turn):
        blended_turn = float(etf_weight) * etf_turn
    else:
        blended_turn = float(etf_weight) * etf_turn + float(1.0 - etf_weight) * stock_turn

    etf_gross = float(_safe_float(etf_metrics.get("max_gross_exposure_pct"), 1.0))
    stock_gross = float(_safe_float(stock_metrics.get("max_gross_exposure_pct"), 1.0))
    blended_gross = min(1.0, float(etf_weight) * etf_gross + float(1.0 - etf_weight) * stock_gross)

    out = dict(metrics or {})
    out["avg_annual_turnover_pct"] = float(blended_turn)
    out["total_trades"] = int(etf_metrics.get("total_trades", 0) or 0) + int(stock_metrics.get("total_trades", 0) or 0)
    out["same_day_open_entries"] = int(etf_metrics.get("same_day_open_entries", 0) or 0) + int(
        stock_metrics.get("same_day_open_entries", 0) or 0
    )
    out["max_gross_exposure_pct"] = float(blended_gross)
    return out


@st.cache_data(ttl=900, show_spinner=False)
def _run_hybrid_benchmark(
    *,
    profile_label: str,
    start_date: str,
    end_date: str,
    train_end_date: Optional[str] = None,
    holdout_start_date: Optional[str] = None,
) -> Dict[str, Any]:
    spec = _hybrid_profile_spec(profile_label)
    helpers = _load_hybrid_benchmark_helpers()
    etf_profile_label = str(spec["etf_profile_label"])
    stock_profile_label = str(spec["stock_profile_label"])
    etf_weight = float(spec["etf_weight"])

    etf_result = _run_etf_benchmark(
        config_path=ETF_FROZEN_BENCHMARKS[etf_profile_label],
        start_date=str(start_date),
        end_date=str(end_date),
        train_end_date=train_end_date,
        holdout_start_date=holdout_start_date,
    )
    stock_result = _run_stock_benchmark(
        config_path=STOCK_BENCHMARK_CANDIDATES[stock_profile_label],
        start_date=str(start_date),
        end_date=str(end_date),
        train_end_date=train_end_date,
        holdout_start_date=holdout_start_date,
    )

    payload: Dict[str, Any] = {
        "profile_label": profile_label,
        "etf_profile_label": etf_profile_label,
        "stock_profile_label": stock_profile_label,
        "etf_weight": float(etf_weight),
        "stock_weight": float(spec["stock_weight"]),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "etf": etf_result,
        "stock": stock_result,
    }

    if train_end_date and holdout_start_date:
        etf_train_eq = _equity_curve_series(etf_result.get("train_run") or {})
        etf_holdout_eq = _equity_curve_series(etf_result.get("holdout_run") or {})
        stock_train_eq = _equity_curve_series(stock_result.get("train_run") or {})
        stock_holdout_eq = _equity_curve_series(stock_result.get("holdout_run") or {})
        hybrid_train_eq = helpers["blend_equity_series"](etf_train_eq, stock_train_eq, etf_weight=etf_weight)
        hybrid_holdout_eq = helpers["blend_equity_series"](etf_holdout_eq, stock_holdout_eq, etf_weight=etf_weight)
        payload["train_end_date"] = str(train_end_date)
        payload["holdout_start_date"] = str(holdout_start_date)
        payload["train"] = _hybrid_window_summary(
            metrics=helpers["metrics_from_equity"](hybrid_train_eq),
            etf_metrics=dict(etf_result.get("train") or {}),
            stock_metrics=dict(stock_result.get("train") or {}),
            etf_weight=etf_weight,
        )
        payload["holdout"] = _hybrid_window_summary(
            metrics=helpers["metrics_from_equity"](hybrid_holdout_eq),
            etf_metrics=dict(etf_result.get("holdout") or {}),
            stock_metrics=dict(stock_result.get("holdout") or {}),
            etf_weight=etf_weight,
        )
        payload["train_run"] = {"equity_curve": _series_to_equity_curve(hybrid_train_eq)}
        payload["holdout_run"] = {"equity_curve": _series_to_equity_curve(hybrid_holdout_eq)}
        return payload

    etf_full_eq = _equity_curve_series(etf_result.get("full_run") or {})
    stock_full_eq = _equity_curve_series(stock_result.get("full_run") or {})
    hybrid_full_eq = helpers["blend_equity_series"](etf_full_eq, stock_full_eq, etf_weight=etf_weight)
    payload["full"] = _hybrid_window_summary(
        metrics=helpers["metrics_from_equity"](hybrid_full_eq),
        etf_metrics=dict(etf_result.get("full") or {}),
        stock_metrics=dict(stock_result.get("full") or {}),
        etf_weight=etf_weight,
    )
    payload["full_run"] = {"equity_curve": _series_to_equity_curve(hybrid_full_eq)}
    return payload


def _build_stock_benchmark_live_context(
    *,
    helpers: Mapping[str, Any],
    cfg: Mapping[str, Any],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    window_days: int,
    membership_union_days: int = 45,
) -> Dict[str, Any]:
    # The live screener only needs the recent PIT Russell membership set,
    # not the full historical union used by walk-forward research.
    recent_start_ts = max(start_ts, (end_ts - pd.Timedelta(days=int(membership_union_days))).normalize())
    requested_symbols, symbol_source = get_universe_symbols_pit_window_with_meta(
        "RUSSELL3000",
        recent_start_ts.date().isoformat(),
        end_ts.date().isoformat(),
    )
    quality_report: Dict[str, Any] = {}
    data = fetch_data_pack(
        list(requested_symbols),
        days=int(window_days),
        backtest_mode=True,
        quality_report=quality_report,
    ) or {}
    loaded_symbols = sorted(data.keys())
    global_data = fetch_data_pack(["SPY", "VIX", "HYG", "LQD"], days=int(window_days), backtest_mode=True) or {}
    prepared = prepare_backtest_data(
        data,
        loaded_symbols,
        start_date=start_ts.date().isoformat(),
        global_data=global_data,
    )
    membership_by_day, membership_source = build_russell3000_membership_by_day(
        list(prepared.all_dates),
        allow_missing_days=False,
    )
    if not membership_by_day or len(membership_by_day) != len(prepared.all_dates):
        raise RuntimeError(f"Failed to build Russell 3000 PIT membership timeline (source={membership_source}).")
    features = helpers["extract_features"](
        prepared,
        membership_by_day=membership_by_day,
        cfg={"price_filter_mode": str(cfg.get("price_filter_mode", "adjusted") or "adjusted")},
    )
    coverage_stats = helpers["daily_membership_coverage"](features)
    return {
        "requested_symbols": list(requested_symbols),
        "symbol_source": str(symbol_source),
        "data": data,
        "loaded_symbols": list(loaded_symbols),
        "global_data": global_data,
        "prepared": prepared,
        "membership_source": str(membership_source),
        "features": features,
        "coverage_stats": coverage_stats,
        "quality_report": quality_report,
        "recent_membership_start_date": recent_start_ts.date().isoformat(),
    }


@st.cache_data(ttl=900, show_spinner=False)
def _build_stock_benchmark_live_snapshot(
    *,
    config_path: str,
    end_date: Optional[str] = None,
    lookback_days: int = 520,
) -> Dict[str, Any]:
    helpers = _load_stock_benchmark_helpers()
    cfg = helpers["load_config"](Path(config_path).resolve())
    requested_end_ts = pd.Timestamp(end_date or pd.Timestamp.utcnow().tz_localize(None).date().isoformat()).normalize()
    attempt_windows = [int(lookback_days)]
    if int(lookback_days) < 720:
        attempt_windows.append(720)
    attempt_errors: List[str] = []

    for window_days in attempt_windows:
        for day_back in range(0, 8):
            end_ts = (requested_end_ts - pd.Timedelta(days=int(day_back))).normalize()
            start_ts = (end_ts - pd.Timedelta(days=int(window_days))).normalize()
            context = _build_stock_benchmark_live_context(
                helpers=helpers,
                cfg=cfg,
                start_ts=start_ts,
                end_ts=end_ts,
                window_days=int(window_days),
            )
            scores = helpers["build_scores"](context["features"], cfg)
            if scores.empty:
                attempt_errors.append(f"{end_ts.date().isoformat()}: no active scored symbols")
                continue

            price_frames = helpers["slice_prices"](context["features"], ["open", "close"], list(scores.columns))
            full_run = helpers["run_window"](
                cfg=cfg,
                prices_close=price_frames["close"],
                prices_open=price_frames["open"],
                scores=scores,
                global_data=dict(context.get("global_data") or {}),
                start_date=start_ts.date().isoformat(),
                end_date=end_ts.date().isoformat(),
            )
            rebalance_log = [dict(item) for item in list(full_run.get("rebalance_log") or [])]
            if not rebalance_log:
                attempt_errors.append(f"{end_ts.date().isoformat()}: no rebalance events")
                continue

            non_empty_indices = [
                idx
                for idx, entry in enumerate(rebalance_log)
                if any(float(_safe_float(weight, 0.0)) > 0.0 for weight in dict(entry.get("weights") or {}).values())
            ]
            if not non_empty_indices:
                attempt_errors.append(f"{end_ts.date().isoformat()}: rebalance log had no active target weights")
                continue

            latest_idx = int(non_empty_indices[-1])
            previous_idx = int(non_empty_indices[-2]) if len(non_empty_indices) > 1 else -1
            latest_log = rebalance_log[latest_idx]
            previous_log = rebalance_log[previous_idx] if previous_idx >= 0 else {}
            latest_signal_dt = pd.Timestamp(latest_log.get("signal_date") or latest_log.get("date")).tz_localize(None).normalize()
            previous_signal_dt = (
                pd.Timestamp(previous_log.get("signal_date") or previous_log.get("date")).tz_localize(None).normalize()
                if previous_log
                else None
            )
            latest_weights = {
                str(sym): float(weight)
                for sym, weight in dict(latest_log.get("weights") or {}).items()
                if _safe_float(weight, 0.0) > 0.0
            }
            previous_weights = {
                str(sym): float(weight)
                for sym, weight in dict(previous_log.get("weights") or {}).items()
                if _safe_float(weight, 0.0) > 0.0
            }
            if not latest_weights:
                attempt_errors.append(f"{end_ts.date().isoformat()}: latest qualifying signal had no weights")
                continue

            close_px = price_frames["close"]
            latest_close_row = {}
            if latest_signal_dt in close_px.index:
                latest_close_row = pd.to_numeric(close_px.loc[latest_signal_dt], errors="coerce").dropna().to_dict()

            coverage_stats = dict(context.get("coverage_stats") or {})
            coverage_ratio = helpers["coverage_ratio"](
                list(context.get("requested_symbols") or []),
                list(context.get("loaded_symbols") or []),
            )
            coverage_failures = helpers["coverage_gate_failures"](
                coverage_ratio=float(coverage_ratio),
                coverage_stats=coverage_stats,
                min_universe_coverage=float(helpers["default_min_universe_coverage"]),
                min_daily_membership_coverage=float(helpers["default_min_daily_membership_coverage"]),
                min_daily_membership_coverage_p10=float(helpers["default_min_daily_membership_coverage_p10"]),
            )

            target_rows: List[Dict[str, Any]] = []
            rebalance_rows: List[Dict[str, Any]] = []
            for sym in sorted(set(latest_weights.keys()) | set(previous_weights.keys())):
                latest_wt = float(latest_weights.get(sym, 0.0) or 0.0)
                prev_wt = float(previous_weights.get(sym, 0.0) or 0.0)
                delta = latest_wt - prev_wt
                last_close = latest_close_row.get(sym)
                if latest_wt > 0.0:
                    target_rows.append(
                        {
                            "Symbol": str(sym),
                            "Target Weight": latest_wt,
                            "Target %": latest_wt * 100.0,
                            "Last Close": float(last_close) if last_close is not None else float("nan"),
                            "Notional per $100k": latest_wt * 100000.0,
                        }
                    )
                if abs(delta) > 1e-9:
                    rebalance_rows.append(
                        {
                            "Symbol": str(sym),
                            "Prev %": prev_wt * 100.0,
                            "Target %": latest_wt * 100.0,
                            "Delta %": delta * 100.0,
                            "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                            "Last Close": float(last_close) if last_close is not None else float("nan"),
                        }
                    )
            target_df = pd.DataFrame(target_rows).sort_values("Target Weight", ascending=False)
            rebalance_df = pd.DataFrame(rebalance_rows)
            if not rebalance_df.empty:
                rebalance_df = rebalance_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

            return {
                "config_name": str(cfg.get("name", "") or os.path.basename(config_path)),
                "config_path": str(Path(config_path).resolve()),
                "configured_target_count": int(cfg.get("target_count", 0) or 0),
                "configured_turnover_budget": float(_safe_float(cfg.get("turnover_budget"), 0.0)),
                "configured_max_position_weight": float(_safe_float(cfg.get("max_position_weight"), 0.0)),
                "latest_signal_date": latest_signal_dt.date().isoformat(),
                "previous_signal_date": previous_signal_dt.date().isoformat() if previous_signal_dt is not None else None,
                "latest_target_weights": latest_weights,
                "previous_target_weights": previous_weights,
                "target_df": target_df,
                "rebalance_df": rebalance_df,
                "signal_count": int(len(rebalance_log)),
                "max_gross_exposure_pct": float(sum(float(v) for v in latest_weights.values())),
                "requested_symbols": int(len(list(context.get("requested_symbols") or []))),
                "loaded_symbols": int(len(list(context.get("loaded_symbols") or []))),
                "active_scored_symbols": int(len(scores.columns)),
                "coverage_ratio": float(coverage_ratio),
                "daily_membership_price_coverage": coverage_stats,
                "coverage_gate_failures": coverage_failures,
                "symbol_source": str(context.get("symbol_source", "")),
                "membership_source": str(context.get("membership_source", "")),
                "recent_membership_start_date": str(context.get("recent_membership_start_date", "")),
                "requested_end_date": requested_end_ts.date().isoformat(),
                "resolved_end_date": end_ts.date().isoformat(),
                "used_end_date_fallback": bool(day_back > 0),
                "used_lookback_fallback": bool(window_days != int(lookback_days)),
                "built_at_utc": _utc_now_iso(),
            }

    detail = "; ".join(attempt_errors[-6:]) if attempt_errors else "no fallback attempts recorded"
    raise RuntimeError("Stock benchmark live snapshot unavailable. " + detail)


def _load_stock_benchmark_paper_state() -> Dict[str, Any]:
    default_state = {
        "profile_label": STOCK_BENCHMARK_DEFAULT_LABEL,
        "adopted_signal_date": None,
        "holdings": {},
        "updated_at_utc": None,
    }
    payload = _read_json_payload(STOCK_BENCHMARK_PAPER_STATE_FILE, default_state)
    if not isinstance(payload, dict):
        return dict(default_state)
    state = dict(default_state)
    state.update(payload)
    if not isinstance(state.get("holdings"), dict):
        state["holdings"] = {}
    if str(state.get("profile_label") or "") not in STOCK_BENCHMARK_CANDIDATES:
        state["profile_label"] = STOCK_BENCHMARK_DEFAULT_LABEL
    return state


def _save_stock_benchmark_paper_state(state: Mapping[str, Any]) -> None:
    payload = {
        "profile_label": str(state.get("profile_label", STOCK_BENCHMARK_DEFAULT_LABEL) or STOCK_BENCHMARK_DEFAULT_LABEL),
        "adopted_signal_date": state.get("adopted_signal_date"),
        "holdings": {
            str(sym): float(weight)
            for sym, weight in dict(state.get("holdings") or {}).items()
            if _safe_float(weight, 0.0) > 0.0
        },
        "updated_at_utc": _utc_now_iso(),
    }
    _write_json_payload(STOCK_BENCHMARK_PAPER_STATE_FILE, payload)


@st.cache_data(ttl=900, show_spinner=False)
def _build_hybrid_benchmark_live_snapshot(*, profile_label: str, end_date: Optional[str] = None) -> Dict[str, Any]:
    spec = _hybrid_profile_spec(profile_label)
    etf_profile_label = str(spec["etf_profile_label"])
    stock_profile_label = str(spec["stock_profile_label"])
    etf_weight = float(spec["etf_weight"])
    stock_weight = float(spec["stock_weight"])
    requested_end_date = str(end_date or _latest_completed_market_session_date())

    stock_snapshot = _build_stock_benchmark_live_snapshot(
        config_path=STOCK_BENCHMARK_CANDIDATES[stock_profile_label],
        end_date=requested_end_date,
    )
    sync_end_date = str(stock_snapshot.get("resolved_end_date") or requested_end_date)
    etf_snapshot = _build_etf_live_snapshot(
        config_path=ETF_FROZEN_BENCHMARKS[etf_profile_label],
        end_date=sync_end_date,
    )
    latest_signal_detail = _describe_signal_pair(
        etf_snapshot.get("latest_signal_date"),
        stock_snapshot.get("latest_signal_date"),
    )
    previous_signal_detail = _describe_signal_pair(
        etf_snapshot.get("previous_signal_date"),
        stock_snapshot.get("previous_signal_date"),
    )

    latest_weights = _blend_weight_maps(
        etf_snapshot.get("latest_target_weights") or {},
        stock_snapshot.get("latest_target_weights") or {},
        etf_weight=etf_weight,
    )
    previous_weights = _blend_weight_maps(
        etf_snapshot.get("previous_target_weights") or {},
        stock_snapshot.get("previous_target_weights") or {},
        etf_weight=etf_weight,
    )

    close_map: Dict[str, float] = {}
    for df in (etf_snapshot.get("target_df"), stock_snapshot.get("target_df")):
        if isinstance(df, pd.DataFrame) and not df.empty and {"Symbol", "Last Close"}.issubset(df.columns):
            for row in df[["Symbol", "Last Close"]].to_dict("records"):
                close_map[str(row["Symbol"])] = float(_safe_float(row.get("Last Close"), float("nan")))

    target_rows: List[Dict[str, Any]] = []
    rebalance_rows: List[Dict[str, Any]] = []
    for source_name, source_label, source_weight, current_weights, previous_source_weights in (
        ("ETF", etf_profile_label, etf_weight, dict(etf_snapshot.get("latest_target_weights") or {}), dict(etf_snapshot.get("previous_target_weights") or {})),
        (
            "Stock",
            stock_profile_label,
            stock_weight,
            dict(stock_snapshot.get("latest_target_weights") or {}),
            dict(stock_snapshot.get("previous_target_weights") or {}),
        ),
    ):
        symbols = sorted(set(current_weights.keys()) | set(previous_source_weights.keys()))
        for sym in symbols:
            target_wt = float(source_weight) * float(_safe_float(current_weights.get(sym), 0.0))
            prev_wt = float(source_weight) * float(_safe_float(previous_source_weights.get(sym), 0.0))
            delta = target_wt - prev_wt
            if target_wt > 0.0:
                target_rows.append(
                    {
                        "Sleeve": source_name,
                        "Source Profile": source_label,
                        "Symbol": str(sym),
                        "Target Weight": target_wt,
                        "Target %": target_wt * 100.0,
                        "Last Close": float(close_map.get(str(sym), float("nan"))),
                        "Notional per $100k": target_wt * 100000.0,
                    }
                )
            if abs(delta) > 1e-9:
                rebalance_rows.append(
                    {
                        "Sleeve": source_name,
                        "Source Profile": source_label,
                        "Symbol": str(sym),
                        "Prev %": prev_wt * 100.0,
                        "Target %": target_wt * 100.0,
                        "Delta %": delta * 100.0,
                        "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                        "Last Close": float(close_map.get(str(sym), float("nan"))),
                    }
                )

    target_df = pd.DataFrame(target_rows).sort_values(["Sleeve", "Target Weight"], ascending=[True, False])
    rebalance_df = pd.DataFrame(rebalance_rows)
    if not rebalance_df.empty:
        rebalance_df = rebalance_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

    return {
        "profile_label": profile_label,
        "etf_profile_label": etf_profile_label,
        "stock_profile_label": stock_profile_label,
        "etf_weight": float(etf_weight),
        "stock_weight": float(stock_weight),
        "latest_signal_date": sync_end_date,
        "previous_signal_date": str(stock_snapshot.get("previous_signal_date") or etf_snapshot.get("previous_signal_date") or ""),
        "latest_signal_detail": latest_signal_detail,
        "previous_signal_detail": previous_signal_detail,
        "latest_target_weights": latest_weights,
        "previous_target_weights": previous_weights,
        "target_df": target_df,
        "rebalance_df": rebalance_df,
        "max_gross_exposure_pct": float(sum(float(v) for v in latest_weights.values())),
        "active_etfs": int(len(dict(etf_snapshot.get("latest_target_weights") or {}))),
        "active_stocks": int(len(dict(stock_snapshot.get("latest_target_weights") or {}))),
        "etf_snapshot": etf_snapshot,
        "stock_snapshot": stock_snapshot,
        "requested_end_date": requested_end_date,
        "resolved_end_date": sync_end_date,
        "built_at_utc": _utc_now_iso(),
    }


def _load_hybrid_benchmark_paper_state() -> Dict[str, Any]:
    default_state = {
        "profile_label": HYBRID_BENCHMARK_DEFAULT_LABEL,
        "adopted_signal_date": None,
        "holdings": {},
        "updated_at_utc": None,
    }
    payload = _read_json_payload(HYBRID_BENCHMARK_PAPER_STATE_FILE, default_state)
    if not isinstance(payload, dict):
        return dict(default_state)
    state = dict(default_state)
    state.update(payload)
    if not isinstance(state.get("holdings"), dict):
        state["holdings"] = {}
    if str(state.get("profile_label") or "") not in HYBRID_BENCHMARK_CANDIDATES:
        state["profile_label"] = HYBRID_BENCHMARK_DEFAULT_LABEL
    return state


def _save_hybrid_benchmark_paper_state(state: Mapping[str, Any]) -> None:
    payload = {
        "profile_label": str(state.get("profile_label", HYBRID_BENCHMARK_DEFAULT_LABEL) or HYBRID_BENCHMARK_DEFAULT_LABEL),
        "adopted_signal_date": state.get("adopted_signal_date"),
        "holdings": {
            str(sym): float(weight)
            for sym, weight in dict(state.get("holdings") or {}).items()
            if _safe_float(weight, 0.0) > 0.0
        },
        "updated_at_utc": _utc_now_iso(),
    }
    _write_json_payload(HYBRID_BENCHMARK_PAPER_STATE_FILE, payload)


def _render_hybrid_benchmark_live_screener(hybrid_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "🧩 Hybrid Benchmark",
        "Review the currently wired legacy ETF + SMID hybrid implementation and compare it against the promoted v2 benchmark reference.",
    )
    st.caption(
        "This live workflow currently combines only the ETF anchor and the validated SMID benchmark. "
        "Signals remain close-to-next-day only."
    )
    st.caption("Use this view to review the legacy blended ETF and stock allocations before the next trading session.")
    _render_hybrid_benchmark_reference_notice()
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh Hybrid Snapshot", type="primary", key="refresh_hybrid_live_snapshot")
    if refresh_requested:
        _build_etf_live_snapshot.clear()
        _build_stock_benchmark_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        try:
            with st.spinner("Building hybrid benchmark snapshot..."):
                snapshot, snapshot_source = _resolve_live_snapshot(
                    session_key="hybrid_live_snapshot",
                    session_label_key="hybrid_live_snapshot_label",
                    snapshot_file=HYBRID_LIVE_SNAPSHOT_FILE,
                    profile_label=hybrid_profile_label,
                    expected_end_date=expected_end_date,
                    builder=lambda: _build_hybrid_benchmark_live_snapshot(
                        profile_label=hybrid_profile_label,
                        end_date=expected_end_date,
                    ),
                    force_refresh=refresh_requested,
                )
        except Exception as exc:
            st.error(f"Hybrid benchmark snapshot unavailable: {exc}")
            st.info("Try Refresh again after market data updates, or open Stock Benchmark to inspect the stock sleeve directly.")
            return
    except Exception as exc:
        st.error(f"Hybrid benchmark snapshot unavailable: {exc}")
        return
    if snapshot_source in {"session", "disk"}:
        st.caption(f"Loaded cached hybrid snapshot for `{expected_end_date}`.")
    elif snapshot_source in {"stale-session", "stale-disk"}:
        resolved = str(snapshot.get("resolved_end_date") or snapshot.get("latest_signal_date") or "latest available")
        st.warning(f"Using the last successful hybrid snapshot from `{resolved}` while a fresh rebuild is unavailable.")

    effective_signal_date = str(snapshot.get("resolved_end_date") or snapshot.get("latest_signal_date", "-"))
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Hybrid Profile", _display_profile_label(hybrid_profile_label))
    m2.metric("Signal Date", effective_signal_date)
    m3.metric("Gross Exposure", f"{float(snapshot.get('max_gross_exposure_pct', 0.0)):.1%}")
    etf_live_weights = {
        str(sym): float(weight)
        for sym, weight in dict(dict(snapshot.get("etf_snapshot") or {}).get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    stock_live_weights = {
        str(sym): float(weight)
        for sym, weight in dict(dict(snapshot.get("stock_snapshot") or {}).get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    m4.metric("ETF Names", sum(1 for weight in etf_live_weights.values() if float(weight) >= 0.01))
    m5.metric("Stock Names >=2%", sum(1 for weight in stock_live_weights.values() if float(weight) >= 0.02))
    st.caption(
        f"Blend: {float(snapshot.get('etf_weight', 0.0)):.0%} ETF / "
        f"{float(snapshot.get('stock_weight', 0.0)):.0%} Stock | "
        f"ETF `{snapshot.get('etf_profile_label')}` | "
        f"Stock `{snapshot.get('stock_profile_label')}`"
    )
    st.caption(
        f"Stock sleeve config: target count {int(dict(snapshot.get('stock_snapshot') or {}).get('configured_target_count', 0) or 0)} | "
        f"turnover budget {float(dict(snapshot.get('stock_snapshot') or {}).get('configured_turnover_budget', 0.0) or 0.0):.1%} | "
        f"max position {float(dict(snapshot.get('stock_snapshot') or {}).get('configured_max_position_weight', 0.0) or 0.0):.1%}"
    )
    signal_detail = str(snapshot.get("latest_signal_detail") or "").strip()
    if signal_detail and signal_detail != effective_signal_date:
        st.caption(f"Signal detail: {signal_detail}")
    _render_snapshot_resolution_note(dict(snapshot.get("stock_snapshot") or {}), label="Stock sleeve")

    stock_cov = dict(dict(snapshot.get("stock_snapshot") or {}).get("daily_membership_price_coverage") or {})
    st.caption(
        f"Stock coverage inside hybrid: "
        f"{float(dict(snapshot.get('stock_snapshot') or {}).get('coverage_ratio', 0.0)):.1%} union | "
        f"{float(stock_cov.get('mean', 0.0)):.1%} mean daily PIT"
    )

    target_df = snapshot.get("target_df")
    if isinstance(target_df, pd.DataFrame) and not target_df.empty:
        _render_target_allocation_with_tail_controls(
            title="Current Target Allocation",
            df=target_df,
            planning_capital=planning_capital,
            key_prefix="hybrid_live_target",
            default_min_display_pct=2.0,
        )

    rebalance_df = snapshot.get("rebalance_df")
    if isinstance(rebalance_df, pd.DataFrame) and not rebalance_df.empty:
        _render_rebalance_with_tail_controls(
            title="Rebalance Delta vs Previous Signal",
            df=rebalance_df,
            planning_capital=planning_capital,
            key_prefix="hybrid_live_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.info("No allocation changes vs the previous hybrid signal.")


def _render_hybrid_benchmark_simulator(hybrid_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "🎮 Hybrid Benchmark Paper Allocator",
        "Track the currently wired legacy ETF + SMID hybrid implementation using the same target schedule as Live Screener and Backtest.",
    )
    st.caption("Use Adopt Latest only after reviewing both ETF and stock sleeve changes below.")
    _render_hybrid_benchmark_reference_notice()
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh Hybrid Recommendation", type="primary", key="refresh_hybrid_sim_snapshot")
    if refresh_requested:
        _build_etf_live_snapshot.clear()
        _build_stock_benchmark_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        try:
            with st.spinner("Refreshing hybrid benchmark recommendation..."):
                snapshot, _ = _resolve_live_snapshot(
                    session_key="hybrid_sim_snapshot",
                    session_label_key="hybrid_sim_snapshot_label",
                    snapshot_file=HYBRID_LIVE_SNAPSHOT_FILE,
                    profile_label=hybrid_profile_label,
                    expected_end_date=expected_end_date,
                    builder=lambda: _build_hybrid_benchmark_live_snapshot(
                        profile_label=hybrid_profile_label,
                        end_date=expected_end_date,
                    ),
                    force_refresh=refresh_requested,
                )
        except Exception as exc:
            st.error(f"Hybrid benchmark recommendation unavailable: {exc}")
            st.info("Try Refresh again after market data updates, or inspect the ETF and stock benchmark workspaces separately.")
            return
    except Exception as exc:
        st.error(f"Hybrid benchmark recommendation unavailable: {exc}")
        return
    snapshot_payload = dict(snapshot or {})

    state = _load_hybrid_benchmark_paper_state()
    current_holdings = {
        str(sym): float(weight)
        for sym, weight in dict(state.get("holdings") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    latest_weights = {
        str(sym): float(weight)
        for sym, weight in dict(snapshot_payload.get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }

    effective_signal_date = str(snapshot_payload.get("resolved_end_date") or snapshot_payload.get("latest_signal_date") or "None")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stored Profile", _display_profile_label(str(state.get("profile_label") or HYBRID_BENCHMARK_DEFAULT_LABEL)))
    m2.metric("Adopted Signal", str(state.get("adopted_signal_date") or "None"))
    m3.metric("Current Gross", f"{sum(current_holdings.values()):.1%}")
    m4.metric("Target Gross", f"{sum(latest_weights.values()):.1%}")

    _render_snapshot_resolution_note(dict(snapshot_payload.get("stock_snapshot") or {}), label="Stock sleeve")
    signal_detail = str(snapshot_payload.get("latest_signal_detail") or "").strip()
    if signal_detail and signal_detail != effective_signal_date:
        st.caption(f"Signal detail: {signal_detail}")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Adopt Latest Hybrid Allocation",
            type="primary",
            key="adopt_latest_hybrid_alloc",
            disabled=not bool(latest_weights),
        ):
            _save_hybrid_benchmark_paper_state(
                {
                    "profile_label": hybrid_profile_label,
                    "adopted_signal_date": effective_signal_date,
                    "holdings": latest_weights,
                }
            )
            st.success("Hybrid benchmark simulator allocation updated.")
            st.rerun()
    with c2:
        if st.button("Reset Hybrid Simulator State", key="reset_hybrid_sim_state"):
            _save_hybrid_benchmark_paper_state(
                {
                    "profile_label": hybrid_profile_label,
                    "adopted_signal_date": None,
                    "holdings": {},
                }
            )
            st.success("Hybrid simulator state reset.")
            st.rerun()

    current_rows = [{"Symbol": sym, "Weight %": float(weight) * 100.0} for sym, weight in sorted(current_holdings.items())]
    target_rows = [{"Symbol": sym, "Weight %": float(weight) * 100.0} for sym, weight in sorted(latest_weights.items())]
    close_map = _extract_close_map_from_snapshot(snapshot_payload)
    for row in current_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    for row in target_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    delta_rows: List[Dict[str, Any]] = []
    for sym in sorted(set(current_holdings.keys()) | set(latest_weights.keys())):
        curr = float(current_holdings.get(sym, 0.0) or 0.0)
        tgt = float(latest_weights.get(sym, 0.0) or 0.0)
        delta = tgt - curr
        if abs(delta) <= 1e-9:
            continue
        delta_rows.append(
            {
                "Symbol": sym,
                "Current %": curr * 100.0,
                "Target %": tgt * 100.0,
                "Delta %": delta * 100.0,
                "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                "Last Close": float(close_map.get(str(sym), float("nan"))),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

    left, right = st.columns(2)
    with left:
        st.subheader("Stored Hybrid Allocation")
        if current_rows:
            _render_target_allocation_with_tail_controls(
                title="Stored Hybrid Allocation",
                df=pd.DataFrame(current_rows),
                planning_capital=planning_capital,
                key_prefix="hybrid_sim_current",
                default_min_display_pct=2.0,
            )
        else:
            st.info("No hybrid allocation stored yet.")
    with right:
        if target_rows:
            _render_target_allocation_with_tail_controls(
                title="Latest Hybrid Target",
                df=pd.DataFrame(target_rows),
                planning_capital=planning_capital,
                key_prefix="hybrid_sim_target",
                default_min_display_pct=2.0,
            )
        else:
            st.info("No hybrid target allocation is loaded.")

    if not delta_df.empty:
        _render_rebalance_with_tail_controls(
            title="Required Rebalance",
            df=delta_df,
            planning_capital=planning_capital,
            key_prefix="hybrid_sim_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.subheader("Required Rebalance")
        st.success("Stored hybrid allocation already matches the latest target.")


def _render_stock_benchmark_live_screener(stock_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "📘 Stock Benchmark",
        "Review the latest target allocation from the current SMID pullback research leader on a PIT Russell 3000 universe.",
    )
    st.caption(
        "This path is research-first. The frozen ETF benchmark remains the production baseline until the stock sleeve clears broader validation and stress tests."
    )
    st.caption("Use this view to review the current stock list, coverage quality, and rebalance changes.")
    st.caption(
        "The stock sleeve targets a concentrated book, but turnover budgeting phases entries and exits. "
        "Use the concentration snapshot above the table to judge effective holdings rather than raw row count."
    )
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh Stock Snapshot", type="primary", key="refresh_stock_live_snapshot")
    if refresh_requested:
        _build_stock_benchmark_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        try:
            with st.spinner("Building stock benchmark snapshot..."):
                snapshot, snapshot_source = _resolve_live_snapshot(
                    session_key="stock_live_snapshot",
                    session_label_key="stock_live_snapshot_label",
                    snapshot_file=STOCK_LIVE_SNAPSHOT_FILE,
                    profile_label=stock_profile_label,
                    expected_end_date=expected_end_date,
                    builder=lambda: _build_stock_benchmark_live_snapshot(
                        config_path=STOCK_BENCHMARK_CANDIDATES[stock_profile_label],
                        end_date=expected_end_date,
                    ),
                    force_refresh=refresh_requested,
                )
        except Exception as exc:
            st.error(f"Stock benchmark snapshot unavailable: {exc}")
            st.info("Try Refresh again after market data updates. If the issue persists, use Backtest to verify the benchmark on a fixed window.")
            return
    except Exception as exc:
        st.error(f"Stock benchmark snapshot unavailable: {exc}")
        return
    if snapshot_source in {"session", "disk"}:
        st.caption(f"Loaded cached stock snapshot for `{expected_end_date}`.")
    elif snapshot_source in {"stale-session", "stale-disk"}:
        resolved = str(snapshot.get("resolved_end_date") or snapshot.get("latest_signal_date") or "latest available")
        st.warning(f"Using the last successful stock snapshot from `{resolved}` while a fresh rebuild is unavailable.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock Profile", _display_profile_label(stock_profile_label))
    m2.metric("Signal Date", str(snapshot.get("latest_signal_date", "-")))
    m3.metric("Gross Exposure", f"{float(snapshot.get('max_gross_exposure_pct', 0.0)):.1%}")
    latest_weights = {
        str(sym): float(weight)
        for sym, weight in dict(snapshot.get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    m4.metric("Core Names >=2%", sum(1 for weight in latest_weights.values() if float(weight) >= 0.02))
    st.caption(
        f"Configured core book: target count {int(snapshot.get('configured_target_count', 0) or 0)} | "
        f"turnover budget {float(snapshot.get('configured_turnover_budget', 0.0) or 0.0):.1%} | "
        f"max position {float(snapshot.get('configured_max_position_weight', 0.0) or 0.0):.1%}"
    )
    st.caption(
        f"Coverage: {float(snapshot.get('coverage_ratio', 0.0)):.1%} union | "
        f"{float(dict(snapshot.get('daily_membership_price_coverage') or {}).get('mean', 0.0)):.1%} mean daily PIT"
    )
    _render_snapshot_resolution_note(snapshot, label="Stock benchmark")
    coverage_failures = list(snapshot.get("coverage_gate_failures") or [])
    if coverage_failures:
        st.warning("Coverage gate warnings: " + "; ".join(str(x) for x in coverage_failures))

    target_df = snapshot.get("target_df")
    if isinstance(target_df, pd.DataFrame) and not target_df.empty:
        _render_target_allocation_with_tail_controls(
            title="Current Target Allocation",
            df=target_df,
            planning_capital=planning_capital,
            key_prefix="stock_live_target",
            default_min_display_pct=2.0,
        )

    rebalance_df = snapshot.get("rebalance_df")
    if isinstance(rebalance_df, pd.DataFrame) and not rebalance_df.empty:
        _render_rebalance_with_tail_controls(
            title="Rebalance Delta vs Previous Signal",
            df=rebalance_df,
            planning_capital=planning_capital,
            key_prefix="stock_live_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.info("No allocation changes vs the previous stock signal.")


def _render_stock_benchmark_simulator(stock_profile_label: str) -> None:
    planning_capital = _current_benchmark_planning_capital()
    expected_end_date = _latest_completed_market_session_date()
    render_mode_header(
        "🎮 Stock Benchmark Paper Allocator",
        "Track the current SMID pullback candidate as a paper allocation book using the same target schedule as Live Screener and Backtest.",
    )
    st.caption("Use this simulator for research paper trading only. The ETF baseline remains the production anchor.")
    st.caption(
        "Position counts can temporarily exceed the configured target while older names are being phased out. "
        "The latest target panel highlights meaningful exposure versus dust positions."
    )
    _render_benchmark_execution_note(planning_capital)
    refresh_requested = st.button("Refresh Stock Recommendation", type="primary", key="refresh_stock_sim_snapshot")
    if refresh_requested:
        _build_stock_benchmark_live_snapshot.clear()
        _build_hybrid_benchmark_live_snapshot.clear()
    try:
        try:
            with st.spinner("Refreshing stock benchmark recommendation..."):
                snapshot, _ = _resolve_live_snapshot(
                    session_key="stock_sim_snapshot",
                    session_label_key="stock_sim_snapshot_label",
                    snapshot_file=STOCK_LIVE_SNAPSHOT_FILE,
                    profile_label=stock_profile_label,
                    expected_end_date=expected_end_date,
                    builder=lambda: _build_stock_benchmark_live_snapshot(
                        config_path=STOCK_BENCHMARK_CANDIDATES[stock_profile_label],
                        end_date=expected_end_date,
                    ),
                    force_refresh=refresh_requested,
                )
        except Exception as exc:
            st.error(f"Stock benchmark recommendation unavailable: {exc}")
            st.info("Try Refresh again after market data updates, or use the Backtest workspace for a fixed historical run.")
            return
    except Exception as exc:
        st.error(f"Stock benchmark recommendation unavailable: {exc}")
        return
    snapshot_payload = dict(snapshot or {})

    state = _load_stock_benchmark_paper_state()
    current_holdings = {
        str(sym): float(weight)
        for sym, weight in dict(state.get("holdings") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }
    latest_weights = {
        str(sym): float(weight)
        for sym, weight in dict(snapshot_payload.get("latest_target_weights") or {}).items()
        if _safe_float(weight, 0.0) > 0.0
    }

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stored Profile", _display_profile_label(str(state.get("profile_label") or STOCK_BENCHMARK_DEFAULT_LABEL)))
    m2.metric("Adopted Signal", str(state.get("adopted_signal_date") or "None"))
    m3.metric("Current Gross", f"{sum(current_holdings.values()):.1%}")
    m4.metric("Target Gross", f"{sum(latest_weights.values()):.1%}")

    _render_snapshot_resolution_note(snapshot_payload, label="Stock benchmark")

    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Adopt Latest Stock Allocation",
            type="primary",
            key="adopt_latest_stock_alloc",
            disabled=not bool(latest_weights),
        ):
            _save_stock_benchmark_paper_state(
                {
                    "profile_label": stock_profile_label,
                    "adopted_signal_date": snapshot_payload.get("latest_signal_date"),
                    "holdings": latest_weights,
                }
            )
            st.success("Stock benchmark simulator allocation updated.")
            st.rerun()
    with c2:
        if st.button("Reset Stock Simulator State", key="reset_stock_sim_state"):
            _save_stock_benchmark_paper_state(
                {
                    "profile_label": stock_profile_label,
                    "adopted_signal_date": None,
                    "holdings": {},
                }
            )
            st.success("Stock benchmark simulator state reset.")
            st.rerun()

    current_rows = [{"Symbol": sym, "Weight %": float(weight) * 100.0} for sym, weight in sorted(current_holdings.items())]
    target_rows = [{"Symbol": sym, "Weight %": float(weight) * 100.0} for sym, weight in sorted(latest_weights.items())]
    close_map = _extract_close_map_from_snapshot(snapshot_payload)
    for row in current_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    for row in target_rows:
        row["Last Close"] = float(close_map.get(str(row["Symbol"]), float("nan")))
        row["Target Weight"] = float(_safe_float(row.get("Weight %"), 0.0)) / 100.0
    delta_rows: List[Dict[str, Any]] = []
    for sym in sorted(set(current_holdings.keys()) | set(latest_weights.keys())):
        curr = float(current_holdings.get(sym, 0.0) or 0.0)
        tgt = float(latest_weights.get(sym, 0.0) or 0.0)
        delta = tgt - curr
        if abs(delta) <= 1e-9:
            continue
        delta_rows.append(
            {
                "Symbol": sym,
                "Current %": curr * 100.0,
                "Target %": tgt * 100.0,
                "Delta %": delta * 100.0,
                "Action": "Buy / Increase" if delta > 0 else "Sell / Reduce",
                "Last Close": float(close_map.get(str(sym), float("nan"))),
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = delta_df.sort_values("Delta %", ascending=False, key=lambda s: s.abs())

    left, right = st.columns(2)
    with left:
        st.subheader("Stored Stock Allocation")
        if current_rows:
            _render_target_allocation_with_tail_controls(
                title="Stored Stock Allocation",
                df=pd.DataFrame(current_rows),
                planning_capital=planning_capital,
                key_prefix="stock_sim_current",
                default_min_display_pct=2.0,
            )
        else:
            st.info("No stock allocation stored yet.")
    with right:
        if target_rows:
            _render_target_allocation_with_tail_controls(
                title="Latest Stock Target",
                df=pd.DataFrame(target_rows),
                planning_capital=planning_capital,
                key_prefix="stock_sim_target",
                default_min_display_pct=2.0,
            )
        else:
            st.info("No stock target allocation is loaded.")

    if not delta_df.empty:
        _render_rebalance_with_tail_controls(
            title="Required Rebalance",
            df=delta_df,
            planning_capital=planning_capital,
            key_prefix="stock_sim_rebalance",
            default_min_delta_pct=0.5,
        )
    else:
        st.subheader("Required Rebalance")
        st.success("Stored stock allocation already matches the latest target.")


def _render_stock_benchmark_lab(default_end_date: str, *, stock_profile_label: str) -> None:
    st.markdown("### 📘 Stock Benchmark Lab")
    st.caption(
        "Run the current stock benchmark candidates outside the generic stock-strategy engine. "
        "This is the concentrated SMID pullback lane that beat the frozen ETF benchmark in PIT Russell holdout testing."
    )
    st.caption("Blind Holdout is the truth-testing view. Full Sample is descriptive only.")
    st.info("This is the cash-only benchmark lane that produced the 30%+ full-window CAGR result.")
    _render_benchmark_execution_note(_current_benchmark_planning_capital())
    m1, m2, m3 = st.columns(3)
    m1.metric("Validated Holdout CAGR", "36.00%")
    m2.metric("Validated Holdout Max DD", "17.05%")
    m3.metric("Lead Config", "SMID Pullback R3000 TC4 TB003")

    st.info(f"Using stock profile from the sidebar: **{_display_profile_label(stock_profile_label)}**")
    eval_mode = st.radio(
        "Stock Evaluation Mode",
        ["Blind Holdout", "Full Sample"],
        horizontal=True,
        key="stock_benchmark_eval_mode",
    )

    if eval_mode == "Blind Holdout":
        c1, c2, c3 = st.columns(3)
        with c1:
            stock_start = st.date_input(
                "Stock Start",
                value=pd.Timestamp("2016-01-01").date(),
                key="stock_holdout_start_base",
            )
        with c2:
            stock_train_end = st.date_input(
                "Train End",
                value=pd.Timestamp("2020-12-31").date(),
                key="stock_holdout_train_end",
            )
        with c3:
            stock_holdout_end = st.date_input(
                "Holdout End",
                value=pd.Timestamp(default_end_date).date(),
                key="stock_holdout_end",
            )
        stock_holdout_start = (pd.Timestamp(stock_train_end) + pd.Timedelta(days=1)).date()
        st.caption(f"Holdout starts automatically on `{stock_holdout_start.isoformat()}`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            stock_start = st.date_input(
                "Stock Start",
                value=pd.Timestamp("2016-01-01").date(),
                key="stock_full_start",
            )
        with c2:
            stock_holdout_end = st.date_input(
                "Stock End",
                value=pd.Timestamp(default_end_date).date(),
                key="stock_full_end",
            )
        stock_train_end = None
        stock_holdout_start = None

    run_btn = st.button("Run Stock Benchmark", type="primary", key="run_stock_benchmark")
    if run_btn:
        try:
            config_path = STOCK_BENCHMARK_CANDIDATES[stock_profile_label]
            start_str = pd.Timestamp(stock_start).date().isoformat()
            end_str = pd.Timestamp(stock_holdout_end).date().isoformat()
            train_end_str = pd.Timestamp(stock_train_end).date().isoformat() if stock_train_end is not None else None
            holdout_start_str = (
                pd.Timestamp(stock_holdout_start).date().isoformat() if stock_holdout_start is not None else None
            )
            with st.spinner("Running stock benchmark candidate..."):
                stock_result = _run_stock_benchmark(
                    config_path=config_path,
                    start_date=start_str,
                    end_date=end_str,
                    train_end_date=train_end_str,
                    holdout_start_date=holdout_start_str,
                )
            stock_result["profile_label"] = stock_profile_label
            stock_result["evaluation_mode"] = eval_mode
            st.session_state.stock_benchmark_result = stock_result
        except Exception as e:
            st.error(f"Stock benchmark run failed: {e}")

    result = st.session_state.get("stock_benchmark_result")
    if not isinstance(result, dict):
        return

    st.markdown("---")
    config_label = os.path.basename(str(result.get("config_path", "")))
    st.caption(
        f"Stock benchmark result: `{result.get('profile_label', STOCK_BENCHMARK_DEFAULT_LABEL)}` "
        f"using `{config_label}`"
    )
    coverage = float(result.get("coverage_ratio", 0.0) or 0.0)
    daily_cov = float(dict(result.get("daily_membership_price_coverage") or {}).get("mean", 0.0) or 0.0)
    st.caption(
        f"Union coverage: {coverage:.1%} | Daily PIT mean coverage: {daily_cov:.1%} | "
        f"Requested: {int(result.get('requested_symbols', 0) or 0):,} | "
        f"Loaded: {int(result.get('loaded_symbols', 0) or 0):,} | "
        f"Active scored: {int(result.get('active_scored_symbols', 0) or 0):,}"
    )

    if "holdout" in result and "train" in result:
        train = dict(result.get("train") or {})
        holdout = dict(result.get("holdout") or {})
        left, right = st.columns(2)
        with left:
            st.subheader("Train")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("CAGR", f"{float(train.get('cagr_pct', 0.0)):.2f}%")
            t2.metric("Max DD", f"{float(train.get('max_dd_pct', 0.0)):.2f}%")
            t3.metric("Calmar", f"{float(train.get('calmar', float('nan'))):.2f}")
            t4.metric("Trades", int(train.get("total_trades", 0) or 0))
        with right:
            st.subheader("Holdout")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("CAGR", f"{float(holdout.get('cagr_pct', 0.0)):.2f}%")
            h2.metric("Max DD", f"{float(holdout.get('max_dd_pct', 0.0)):.2f}%")
            h3.metric("Calmar", f"{float(holdout.get('calmar', float('nan'))):.2f}")
            h4.metric("Trades", int(holdout.get("total_trades", 0) or 0))
            st.caption(
                f"Same-day open entries: {int(holdout.get('same_day_open_entries', 0) or 0)} | "
                f"Max gross: {float(holdout.get('max_gross_exposure_pct', 0.0) or 0.0):.1%}"
            )
        holdout_curve = normalize_equity_curve_df((result.get("holdout_run") or {}).get("equity_curve", []))
        if not holdout_curve.empty:
            st.line_chart(holdout_curve.set_index("Date")["Equity"])
    else:
        full = dict(result.get("full") or {})
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("CAGR", f"{float(full.get('cagr_pct', 0.0)):.2f}%")
        f2.metric("Max DD", f"{float(full.get('max_dd_pct', 0.0)):.2f}%")
        f3.metric("Calmar", f"{float(full.get('calmar', float('nan'))):.2f}")
        f4.metric("Trades", int(full.get("total_trades", 0) or 0))
        f5.metric("Turnover", f"{float(full.get('avg_annual_turnover_pct', float('nan'))):.1f}%")
        st.caption(
            f"Same-day open entries: {int(full.get('same_day_open_entries', 0) or 0)} | "
            f"Max gross: {float(full.get('max_gross_exposure_pct', 0.0) or 0.0):.1%}"
        )
        full_curve = normalize_equity_curve_df((result.get("full_run") or {}).get("equity_curve", []))
        if not full_curve.empty:
            st.line_chart(full_curve.set_index("Date")["Equity"])

    payload_json = json.dumps(result, indent=2, default=str)
    export_meta = _persist_streamlit_export("stock_benchmark_result.json", payload_json)
    st.download_button(
        label="📥 Export Stock Benchmark Result (JSON)",
        data=payload_json.encode("utf-8"),
        file_name="stock_benchmark_result.json",
        mime="application/json",
        key="dl_stock_benchmark_json",
    )
    _render_streamlit_export_status(export_meta)


def _render_hybrid_benchmark_lab(default_end_date: str, *, hybrid_profile_label: str) -> None:
    st.markdown("### 🧩 Hybrid Benchmark Lab")
    st.caption(
        "Run the currently wired legacy ETF + SMID benchmark blends outside the generic stock-strategy engine."
    )
    st.caption("Blind Holdout is the validation view for the legacy fixed blend. Full Sample is the long-run context view.")
    _render_hybrid_benchmark_reference_notice()
    _render_benchmark_execution_note(_current_benchmark_planning_capital())
    hybrid_catalog = _load_hybrid_benchmark_catalog()
    hybrid_promoted = dict(hybrid_catalog.get("promoted") or {})
    hybrid_promoted_validated = dict(hybrid_promoted.get("validated") or {})
    hybrid_legacy_holdout = dict(hybrid_catalog.get("legacy_default_holdout") or {})
    h1, h2, h3 = st.columns(3)
    h1.metric("Promoted v2 Holdout CAGR", _fmt_pct(hybrid_promoted_validated.get("holdout_cagr_pct")))
    h2.metric("Promoted v2 Holdout Max DD", _fmt_pct(hybrid_promoted_validated.get("holdout_dd_pct")))
    h3.metric("Selected Legacy Blend", _display_profile_label(hybrid_profile_label))
    if hybrid_legacy_holdout:
        st.caption(
            f"Selected GUI default legacy reference: {_fmt_pct(hybrid_legacy_holdout.get('cagr_pct'))} CAGR / "
            f"{_fmt_pct(hybrid_legacy_holdout.get('max_dd_pct'))} max DD."
        )

    st.info(f"Using hybrid profile from the sidebar: **{_display_profile_label(hybrid_profile_label)}**")
    st.info(
        "Run Hybrid Benchmark currently evaluates the legacy ETF + SMID blend selected above. "
        "It does not yet include the B17 sleeve used by the promoted v2 benchmark."
    )
    eval_mode = st.radio(
        "Hybrid Evaluation Mode",
        ["Blind Holdout", "Full Sample"],
        horizontal=True,
        key="hybrid_benchmark_eval_mode",
    )

    if eval_mode == "Blind Holdout":
        c1, c2, c3 = st.columns(3)
        with c1:
            hybrid_start = st.date_input(
                "Hybrid Start",
                value=pd.Timestamp("2016-01-01").date(),
                key="hybrid_holdout_start_base",
            )
        with c2:
            hybrid_train_end = st.date_input(
                "Train End",
                value=pd.Timestamp("2020-12-31").date(),
                key="hybrid_holdout_train_end",
            )
        with c3:
            hybrid_holdout_end = st.date_input(
                "Holdout End",
                value=pd.Timestamp(default_end_date).date(),
                key="hybrid_holdout_end",
            )
        hybrid_holdout_start = (pd.Timestamp(hybrid_train_end) + pd.Timedelta(days=1)).date()
        st.caption(f"Holdout starts automatically on `{hybrid_holdout_start.isoformat()}`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            hybrid_start = st.date_input(
                "Hybrid Start",
                value=pd.Timestamp("2016-01-01").date(),
                key="hybrid_full_start",
            )
        with c2:
            hybrid_holdout_end = st.date_input(
                "Hybrid End",
                value=pd.Timestamp(default_end_date).date(),
                key="hybrid_full_end",
            )
        hybrid_train_end = None
        hybrid_holdout_start = None

    run_btn = st.button("Run Hybrid Benchmark", type="primary", key="run_hybrid_benchmark")
    if run_btn:
        try:
            start_str = pd.Timestamp(hybrid_start).date().isoformat()
            end_str = pd.Timestamp(hybrid_holdout_end).date().isoformat()
            train_end_str = pd.Timestamp(hybrid_train_end).date().isoformat() if hybrid_train_end is not None else None
            holdout_start_str = (
                pd.Timestamp(hybrid_holdout_start).date().isoformat() if hybrid_holdout_start is not None else None
            )
            with st.spinner("Running hybrid benchmark candidate..."):
                hybrid_result = _run_hybrid_benchmark(
                    profile_label=hybrid_profile_label,
                    start_date=start_str,
                    end_date=end_str,
                    train_end_date=train_end_str,
                    holdout_start_date=holdout_start_str,
                )
            hybrid_result["evaluation_mode"] = eval_mode
            st.session_state.hybrid_benchmark_result = hybrid_result
        except Exception as e:
            st.error(f"Hybrid benchmark run failed: {e}")

    result = st.session_state.get("hybrid_benchmark_result")
    if not isinstance(result, dict):
        return

    st.markdown("---")
    st.caption(
        f"Hybrid benchmark result: `{result.get('profile_label', HYBRID_BENCHMARK_DEFAULT_LABEL)}` | "
        f"ETF `{result.get('etf_profile_label', ETF_FROZEN_DEFAULT_LABEL)}` | "
        f"Stock `{result.get('stock_profile_label', STOCK_BENCHMARK_DEFAULT_LABEL)}`"
    )
    st.caption(
        f"Blend weights: {float(result.get('etf_weight', 0.0)):.0%} ETF / "
        f"{float(result.get('stock_weight', 0.0)):.0%} Stock"
    )

    if "holdout" in result and "train" in result:
        train = dict(result.get("train") or {})
        holdout = dict(result.get("holdout") or {})
        left, right = st.columns(2)
        with left:
            st.subheader("Train")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("CAGR", f"{float(train.get('cagr_pct', 0.0)):.2f}%")
            t2.metric("Max DD", f"{float(train.get('max_dd_pct', 0.0)):.2f}%")
            t3.metric("Calmar", f"{float(train.get('calmar', float('nan'))):.2f}")
            t4.metric("Approx Trades", int(train.get("total_trades", 0) or 0))
        with right:
            st.subheader("Holdout")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("CAGR", f"{float(holdout.get('cagr_pct', 0.0)):.2f}%")
            h2.metric("Max DD", f"{float(holdout.get('max_dd_pct', 0.0)):.2f}%")
            h3.metric("Calmar", f"{float(holdout.get('calmar', float('nan'))):.2f}")
            h4.metric("Approx Trades", int(holdout.get("total_trades", 0) or 0))
            st.caption(
                f"Approx turnover: {float(holdout.get('avg_annual_turnover_pct', float('nan'))):.1f}% | "
                f"Same-day open entries: {int(holdout.get('same_day_open_entries', 0) or 0)}"
            )
        holdout_curve = normalize_equity_curve_df((result.get("holdout_run") or {}).get("equity_curve", []))
        if not holdout_curve.empty:
            st.line_chart(holdout_curve.set_index("Date")["Equity"])
    else:
        full = dict(result.get("full") or {})
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("CAGR", f"{float(full.get('cagr_pct', 0.0)):.2f}%")
        f2.metric("Max DD", f"{float(full.get('max_dd_pct', 0.0)):.2f}%")
        f3.metric("Calmar", f"{float(full.get('calmar', float('nan'))):.2f}")
        f4.metric("Approx Trades", int(full.get("total_trades", 0) or 0))
        f5.metric("Approx Turnover", f"{float(full.get('avg_annual_turnover_pct', float('nan'))):.1f}%")
        st.caption(f"Same-day open entries: {int(full.get('same_day_open_entries', 0) or 0)}")
        full_curve = normalize_equity_curve_df((result.get("full_run") or {}).get("equity_curve", []))
        if not full_curve.empty:
            st.line_chart(full_curve.set_index("Date")["Equity"])

    comp_train = pd.DataFrame(
        [
            {"Sleeve": "ETF", **dict(result.get("etf", {}).get("train") or {})},
            {"Sleeve": "Stock", **dict(result.get("stock", {}).get("train") or {})},
        ]
    )
    comp_holdout = pd.DataFrame(
        [
            {"Sleeve": "ETF", **dict(result.get("etf", {}).get("holdout") or {})},
            {"Sleeve": "Stock", **dict(result.get("stock", {}).get("holdout") or {})},
        ]
    )
    if "holdout" in result and not comp_holdout.empty:
        st.subheader("Component Comparison")
        comp_df = pd.DataFrame(
            [
                {
                    "Sleeve": "ETF",
                    "Train CAGR %": float(dict(result.get("etf", {}).get("train") or {}).get("cagr_pct", float("nan"))),
                    "Train Max DD %": float(dict(result.get("etf", {}).get("train") or {}).get("max_dd_pct", float("nan"))),
                    "Holdout CAGR %": float(dict(result.get("etf", {}).get("holdout") or {}).get("cagr_pct", float("nan"))),
                    "Holdout Max DD %": float(dict(result.get("etf", {}).get("holdout") or {}).get("max_dd_pct", float("nan"))),
                },
                {
                    "Sleeve": "Stock",
                    "Train CAGR %": float(dict(result.get("stock", {}).get("train") or {}).get("cagr_pct", float("nan"))),
                    "Train Max DD %": float(dict(result.get("stock", {}).get("train") or {}).get("max_dd_pct", float("nan"))),
                    "Holdout CAGR %": float(dict(result.get("stock", {}).get("holdout") or {}).get("cagr_pct", float("nan"))),
                    "Holdout Max DD %": float(dict(result.get("stock", {}).get("holdout") or {}).get("max_dd_pct", float("nan"))),
                },
            ]
        )
        st.dataframe(
            comp_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Train CAGR %": st.column_config.NumberColumn(format="%.2f"),
                "Train Max DD %": st.column_config.NumberColumn(format="%.2f"),
                "Holdout CAGR %": st.column_config.NumberColumn(format="%.2f"),
                "Holdout Max DD %": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    payload_json = json.dumps(result, indent=2, default=str)
    export_meta = _persist_streamlit_export("hybrid_benchmark_result.json", payload_json)
    st.download_button(
        label="📥 Export Hybrid Benchmark Result (JSON)",
        data=payload_json.encode("utf-8"),
        file_name="hybrid_benchmark_result.json",
        mime="application/json",
        key="dl_hybrid_benchmark_json",
    )
    _render_streamlit_export_status(export_meta)


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


def _duration_years(duration_label: str | None) -> int | None:
    if duration_label is None:
        return None
    key = str(duration_label or "").strip().lower()
    mapping = {
        "1 year": 1,
        "5 years": 5,
        "10 years": 10,
        "20 years": 20,
    }
    return mapping.get(key)


def _min_coverage_threshold(
    universe_name: str,
    *,
    accuracy_mode: str | None = None,
    duration_label: str | None = None,
) -> float:
    u = str(universe_name or "").upper()
    mode = str(accuracy_mode or _accuracy_mode()).strip().lower()
    years = _duration_years(duration_label)
    if u in {"RUSSELL3000", "RUSSELL 3000"}:
        if mode == "block":
            # Balanced strict defaults: require high quality without making long-horizon
            # PIT backtests impossible due delist/rename data constraints.
            if years is not None and years <= 1:
                default = "0.75"
            elif years is not None and years <= 5:
                default = "0.55"
            elif years is not None and years <= 10:
                default = "0.50"
            else:
                default = "0.45"
        elif mode == "warn":
            if years is not None and years <= 1:
                default = "0.70"
            elif years is not None and years <= 5:
                default = "0.50"
            elif years is not None and years <= 10:
                default = "0.45"
            else:
                default = "0.40"
        else:
            default = "0.10"
        raw = os.getenv("APEX_BACKTEST_MIN_COVERAGE_RUSSELL", default)
    else:
        default = "0.80" if mode == "block" else "0.65"
        raw = os.getenv("APEX_BACKTEST_MIN_COVERAGE", default)
    try:
        val = float(raw or 0.0)
    except Exception:
        val = float(default)
    return max(0.10, min(val, 1.00))


def _accuracy_thresholds(
    universe_name: str,
    *,
    accuracy_mode: str | None = None,
    duration_label: str | None = None,
) -> Dict[str, float]:
    mode = str(accuracy_mode or _accuracy_mode()).strip().lower()
    years = _duration_years(duration_label)
    is_russell = str(universe_name or "").upper() in {"RUSSELL3000", "RUSSELL 3000"}

    min_cov_required = _min_coverage_threshold(
        universe_name,
        accuracy_mode=mode,
        duration_label=duration_label,
    )

    if mode == "off":
        return {
            "min_coverage": min_cov_required,
            "min_recent_coverage": 0.0,
            "max_incomplete_ratio": 1.0,
            "max_stale_ratio": 1.0,
        }

    if mode == "block":
        if is_russell:
            if years is not None and years <= 1:
                recent_default = "0.75"
                stale_default = "0.25"
            elif years is not None and years <= 5:
                recent_default = "0.55"
                stale_default = "0.40"
            elif years is not None and years <= 10:
                recent_default = "0.50"
                stale_default = "0.45"
            else:
                recent_default = "0.45"
                stale_default = "0.50"
        else:
            recent_default = "0.75"
            stale_default = "0.25"
        incomplete_default = "0.10"
        min_recent_cov_required = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MIN_RECENT_COVERAGE_STRICT", recent_default) or recent_default),
            ),
        )
        max_incomplete_ratio = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MAX_INCOMPLETE_RATIO_STRICT", incomplete_default) or incomplete_default),
            ),
        )
        max_stale_ratio = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MAX_STALE_RATIO_STRICT", stale_default) or stale_default),
            ),
        )
    else:
        # warn mode
        if is_russell:
            if years is not None and years <= 1:
                recent_default = "0.65"
                stale_default = "0.35"
            elif years is not None and years <= 5:
                recent_default = "0.50"
                stale_default = "0.55"
            elif years is not None and years <= 10:
                recent_default = "0.45"
                stale_default = "0.60"
            else:
                recent_default = "0.40"
                stale_default = "0.65"
        else:
            recent_default = "0.60"
            stale_default = "0.40"
        incomplete_default = "0.20"
        min_recent_cov_required = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MIN_RECENT_COVERAGE_WARN", recent_default) or recent_default),
            ),
        )
        max_incomplete_ratio = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MAX_INCOMPLETE_RATIO_WARN", incomplete_default) or incomplete_default),
            ),
        )
        max_stale_ratio = max(
            0.0,
            min(
                1.0,
                float(os.getenv("APEX_BACKTEST_MAX_STALE_RATIO_WARN", stale_default) or stale_default),
            ),
        )

    return {
        "min_coverage": float(min_cov_required),
        "min_recent_coverage": float(min_recent_cov_required),
        "max_incomplete_ratio": float(max_incomplete_ratio),
        "max_stale_ratio": float(max_stale_ratio),
    }


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
    st.caption("Single-strategy swing trading workspace")
    st.markdown("---")
    st.info("Apex Swing production profile active")
    mode = st.radio("Select Mode", ["Live Screener", "Backtest", "Simulator"])
    primary_strategy = PRIMARY_STRATEGY_OPTIONS[0]

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
        st.caption("Russell min coverage: `APEX_BACKTEST_MIN_COVERAGE_RUSSELL` (defaults are duration-aware in block/warn mode)")
        st.caption("Run validator: `./.venv/bin/python tools/validate_pit_universe.py --strict`")

    selected_etf_profile_label = ETF_FROZEN_DEFAULT_LABEL
    selected_hybrid_profile_label = HYBRID_BENCHMARK_DEFAULT_LABEL
    selected_stock_benchmark_label = STOCK_BENCHMARK_DEFAULT_LABEL

    strategies_list = load_strategy_configs()
    strategies_map = {s.get("name", f"Strategy {i+1}"): s for i, s in enumerate(strategies_list)}
    selected_strategies = strategies_list[:1]
    active_strategy = dict(selected_strategies[0]) if selected_strategies else {}

    st.markdown("### 🏁 Current Focus")
    if active_strategy:
        st.markdown(
            "- **Workspace:** Apex Swing\n"
            "- **Mandate:** cash-only, next-day executable swing trading\n"
            "- **Style:** Minervini/Qullamaggie hybrid with EP, VCP, and low-cheat entries\n"
            "- **Objective:** maximize realistic long-run CAGR while keeping trade count low and drawdown controlled"
        )
    else:
        st.error("No production strategy config is available. Restore `config/superperformance_winner.json`.")

    st.markdown("### 💵 Order Planning")
    st.number_input(
        "Planning Capital ($)",
        min_value=1000.0,
        step=10000.0,
        value=float(_safe_float(st.session_state.get("benchmark_planning_capital"), 100000.0)),
        key="benchmark_planning_capital",
        help="Planning tables convert target weights into estimated dollars and shares using this account size. Backtest logic is unchanged.",
    )
    st.caption("Share estimates use the latest close for planning. Modeled fills remain next-session open plus configured slippage.")

    st.markdown("### 🧭 Quick Start")
    st.caption("1. Use Live Screener after the close to build the next-session watchlist.")
    st.caption("2. Use Backtest for strict point-in-time validation and exportable results.")
    st.caption("3. Use Simulator to practice the nightly scan, next-open entry, and exit workflow.")

    st.markdown("### 📘 Apex Swing Blueprint")
    if active_strategy:
        rs_gate = float(active_strategy.get("rs_gate_min", 85) or 85)
        growth_gate = float(active_strategy.get("fundamental_growth_min_pct", 10) or 10)
        min_score = float(active_strategy.get("min_entry_score", 0) or 0)
        max_stop = float(active_strategy.get("max_stop_pct", 0.06) or 0.06) * 100.0
        time_stop_days = int(active_strategy.get("time_stop_days", 7) or 7)
        risk_trade = float(active_strategy.get("risk_per_trade", 0.01) or 0.01) * 100.0
        max_positions = int(active_strategy.get("max_positions", 3) or 3)
        entry_mode = str(active_strategy.get("entry_mode", "both") or "both").upper()
        st.caption(f"Active profile: `{active_strategy.get('name', 'Apex Swing')}`")
        st.markdown(
            f"""
**Gate Logic**
- Core trend gate stays strict for breakout entries.
- RS gate: **>= {rs_gate:.0f}**
- Fundamental growth: **>= {growth_gate:.0f}%**

**Entry Logic**
- Archetype union: **{entry_mode}**
- Execution model: **after close signal, next-session order**
- Composite score floor: **>= {min_score:.1f}**

**Risk & Exit**
- Max positions: **{max_positions}**
- Initial stop cap: **{max_stop:.1f}%**
- Risk per trade: **{risk_trade:.1f}%**
- Global dead-money clock: **{time_stop_days} bars**
"""
        )


# --- 1. LIVE SCREENER ---
if mode == "Live Screener":
    if primary_strategy == "ETF Benchmark":
        _render_etf_live_screener(selected_etf_profile_label)
        st.stop()
    if primary_strategy == "Hybrid Benchmark":
        _render_hybrid_benchmark_live_screener(selected_hybrid_profile_label)
        st.stop()
    if primary_strategy == "Stock Benchmark":
        _render_stock_benchmark_live_screener(selected_stock_benchmark_label)
        st.stop()
    render_mode_header(
        "🚀 Daily Opportunity Scanner",
        "Scan the Apex Swing production engine after the close and build a next-session execution list with point-in-time universe alignment.",
    )
    if "scan_results" not in st.session_state:
        st.session_state.scan_results = None

    col1, col2 = st.columns([1, 4])
    with col1:
        universe = st.selectbox(
            "Universe",
            UNIVERSE_OPTIONS,
            index=UNIVERSE_OPTIONS.index(DEFAULT_UNIVERSE),
        )
        run_btn = st.button(
            "RUN SCAN",
            type="primary",
            disabled=not bool(selected_strategies),
            help=None if selected_strategies else "Select at least one strategy in the sidebar.",
        )
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
            live_as_of = datetime.now(ZoneInfo("UTC")).date().isoformat()
            try:
                symbols, live_universe_source = get_universe_symbols_pit_with_meta("RUSSELL3000", live_as_of)
            except RuntimeError as e:
                st.error(
                    "Point-in-time Russell 3000 membership data is required for this scan. "
                    "Configure `RUSSELL3000_PIT_DIR` or `RUSSELL3000_PIT_MEMBERSHIP_CSV`."
                )
                st.caption(f"Resolver detail: {e}")
                progress_bar.empty()
                status_msg.empty()
                timer_msg.empty()
                st.stop()
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
            scan_error_count = 0
            scan_error_examples: list[str] = []
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
                    except Exception as e:
                        scan_error_count += 1
                        if len(scan_error_examples) < 8:
                            scan_error_examples.append(f"{sym}: {str(e)}")

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
            if scan_error_count > 0:
                st.warning(
                    f"{scan_error_count} symbol evaluations raised exceptions and were skipped."
                )
                with st.expander("🔎 Scan Diagnostics (sample errors)"):
                    st.text("\n".join(scan_error_examples))

    if st.session_state.scan_results is not None:
        # --- DATA PREP ---
        df = st.session_state.scan_results.copy()

        if df.empty:
            st.info("No signals found today.")
        else:
            status_col = df.get("Status")
            status_text = status_col.astype(str) if status_col is not None else pd.Series([], dtype="string")
            tradable_n = int(status_text.str.contains("TRADABLE", na=False).sum())
            pending_n = int(status_text.str.contains("PENDING", na=False).sum())
            wait_n = int(status_text.str.contains("WAIT", na=False).sum())
            rejected_n = int(status_text.str.contains("REJECTED|SKIP", na=False).sum())
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Scanned", f"{len(df):,}")
            k2.metric("Tradable", f"{tradable_n:,}")
            k3.metric("Pending/Held", f"{pending_n:,}")
            k4.metric("Watchlist", f"{wait_n:,}")
            k5.metric("Rejected", f"{rejected_n:,}")

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
                        width="stretch",
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
                        width="stretch",
                        hide_index=True
                    )
                else:
                    st.info("No Alpha Targets. Market quiet or slots full.")

            with col2:
                st.subheader("🔄 Smart Swaps (Advisory)")
                if swap_recommendations:
                    st.dataframe(pd.DataFrame(swap_recommendations), width="stretch", hide_index=True)
                    if best_buy is None:
                        st.caption("Strategy: Raise Cash (No Buys Available)")
                    else:
                        st.caption("Strategy: Swap to Upgrade")
                else:
                    st.success("🛡️ Portfolio Optimized")
        
                st.subheader("🔭 Watchtower")
                if not watchtower.empty:
                    st.dataframe(watchtower[["Symbol", "Score", "Status"]], width="stretch", hide_index=True)

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
    render_mode_header(
        "📈 Apex Swing Backtest Lab",
        "Run disciplined historical validations for Apex Swing with accuracy gating, PIT coverage checks, and exportable diagnostics.",
    )

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    bt_end_date = today.date().isoformat()
    if primary_strategy == "ETF Benchmark":
        _render_etf_benchmark_lab(bt_end_date, etf_profile_label=selected_etf_profile_label)
        st.stop()
    if primary_strategy == "Hybrid Benchmark":
        _render_hybrid_benchmark_lab(bt_end_date, hybrid_profile_label=selected_hybrid_profile_label)
        st.stop()
    if primary_strategy == "Stock Benchmark":
        _render_stock_benchmark_lab(bt_end_date, stock_profile_label=selected_stock_benchmark_label)
        st.stop()
    
    col_uni, col_dur = st.columns([1, 3])
    with col_uni:
        bt_universe = st.selectbox(
            "Universe",
            UNIVERSE_OPTIONS,
            index=UNIVERSE_OPTIONS.index(DEFAULT_UNIVERSE),
        )
    
    with col_dur:
        st.write("Duration:")
        c1, c2, c3, c4, c5 = st.columns(5)
        dur_map = {"1 Year": 252, "5 Years": 1260, "10 Years": 2520, "20 Years": 5040, "Max": 10000}
        if "bt_duration" not in st.session_state: st.session_state.bt_duration = "10 Years"
        
        for label in dur_map:
            if c1.button(label) if label=="1 Year" else c2.button(label) if label=="5 Years" else c3.button(label) if label=="10 Years" else c4.button(label) if label=="20 Years" else c5.button(label):
                st.session_state.bt_duration = label
    
    bt_duration = st.session_state.bt_duration
    days = dur_map.get(bt_duration, 1260)
    year_map = {"1 Year": 1, "5 Years": 5, "10 Years": 10, "20 Years": 20}
    bt_years = year_map.get(bt_duration)
    bt_start_ts = (today - pd.DateOffset(years=bt_years)).normalize() if bt_years else None
    bt_start_date = bt_start_ts.date().isoformat() if bt_start_ts is not None else None

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
    thresholds_hint = _accuracy_thresholds(
        bt_universe,
        accuracy_mode=run_accuracy_mode,
        duration_label=bt_duration,
    )
    st.caption(
        f"Run accuracy mode: `{run_accuracy_mode}` | "
        f"Coverage >= {thresholds_hint['min_coverage']:.0%}, "
        f"Recent >= {thresholds_hint['min_recent_coverage']:.0%}, "
        f"Stale <= {thresholds_hint['max_stale_ratio']:.0%}"
    )

    st.markdown("### 🧪 Apex Swing Validation Lab")
    st.caption(
        "Validate the single production Apex Swing profile across strict point-in-time universes, exportable results, and repeatability checks."
    )

    run_backtest_btn = st.button(
        "🚀 RUN BACKTEST",
        type="primary",
        disabled=not bool(selected_strategies),
        help=None if selected_strategies else "Select at least one strategy in the sidebar.",
    )
    if run_backtest_btn:
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
                    "source_hits": {},
                }

                def _merge_quality(rep: dict | None) -> None:
                    if not rep:
                        return
                    quality_union["requested"].update(rep.get("requested_symbols") or [])
                    quality_union["loaded"].update(rep.get("loaded_symbols") or [])
                    quality_union["missing"].update(rep.get("missing_symbols") or [])
                    quality_union["incomplete"].update(rep.get("incomplete_symbols") or [])
                    quality_union["stale"].update(rep.get("stale_symbols") or [])
                    src = rep.get("source_hits") or {}
                    if isinstance(src, dict):
                        dst = quality_union["source_hits"]
                        for k, v in src.items():
                            key = str(k or "").strip()
                            if not key:
                                continue
                            try:
                                inc = int(v or 0)
                            except Exception:
                                inc = 0
                            if inc <= 0:
                                continue
                            dst[key] = int(dst.get(key, 0) or 0) + inc

                strict_full_lookback = False
                if not cache_hit:
                    stage_msg.caption("Stage: Resolving universe membership...")
                    if bt_universe in ("Russell 3000", "RUSSELL3000"):
                        try:
                            symbols, universe_source = get_universe_symbols_pit_window_with_meta(
                                "RUSSELL3000",
                                bt_start_date,
                                bt_end_date,
                            )
                        except RuntimeError as e:
                            st.error(
                                "Point-in-time Russell 3000 membership data is required for this backtest. "
                                "Set `RUSSELL3000_PIT_DIR` or `RUSSELL3000_PIT_MEMBERSHIP_CSV`."
                            )
                            st.caption(f"Resolver detail: {e}")
                            st.session_state.backtest_results = {}
                            symbols = []
                            universe_source = "unavailable"
                    else:
                        symbols = get_index_symbols(bt_universe)
                        universe_source = "current_index"
                    expected_symbols = list(symbols or [])
                    expected_symbol_count = len(expected_symbols)
                    if not symbols:
                        st.error(f"No symbols loaded for universe: {bt_universe}")
                        st.session_state.backtest_results = {}
                        symbols = []

                    accuracy_mode = run_accuracy_mode
                    accuracy_active = accuracy_mode in {"warn", "block"}
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
                    # Verified/strict runs prioritize freshness over cache-only speed by default.
                    if accuracy_mode == "block" and not _env_flag("APEX_BACKTEST_STRICT_CACHE_FIRST", "0"):
                        cache_only_first = False
                    try:
                        min_cache_coverage = float(
                            os.getenv("APEX_BACKTEST_CACHE_MIN_COVERAGE", "0.80") or "0.80"
                        )
                    except Exception:
                        min_cache_coverage = 0.80
                    min_cache_coverage = max(0.50, min(min_cache_coverage, 1.00))
                    force_refresh_for_accuracy = _env_flag("APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY", "1")
                    force_fresh_refresh = (
                        accuracy_mode == "block"
                        and _env_flag("APEX_BACKTEST_FORCE_FRESH_REFRESH", "1")
                    )
                    incremental_refresh_enabled = _env_flag("APEX_BACKTEST_INCREMENTAL_REFRESH", "1")
                    max_end_lag_days = int(os.getenv("APEX_BACKTEST_MAX_END_LAG_DAYS", "7") or "7")
                    max_end_lag_days = max(0, max_end_lag_days)
                    end_scope_symbols: set[str] = set()
                    if bt_universe in ("Russell 3000", "RUSSELL3000") and bt_end_date:
                        try:
                            end_members, _end_source = get_universe_symbols_pit_with_meta("RUSSELL3000", bt_end_date)
                        except Exception:
                            end_members = []
                        if end_members:
                            end_scope_symbols = {str(s).upper() for s in end_members if str(s).strip()}
                    strict_full_lookback = (
                        accuracy_mode == "block"
                        and _env_flag("APEX_BACKTEST_ENFORCE_FULL_LOOKBACK", "0")
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
                            missing_symbols = [s for s in symbols if s not in data]
                            refresh_symbols: list[str] = []
                            if initial_cov < min_cache_coverage:
                                refresh_symbols.extend(missing_symbols)

                            strict_recent_refresh_needed = False
                            strict_recent_cov = None
                            strict_recent_missing: list[str] = []
                            strict_recent_stale: list[str] = []
                            if accuracy_mode == "block" and end_scope_symbols:
                                thresholds_for_refresh = _accuracy_thresholds(
                                    bt_universe,
                                    accuracy_mode=accuracy_mode,
                                    duration_label=bt_duration,
                                )
                                min_recent_cov_for_refresh = float(
                                    thresholds_for_refresh.get("min_recent_coverage", 0.0)
                                )
                                now_et = datetime.now(ZoneInfo("America/New_York")).date()
                                fresh_end = 0
                                for sym in end_scope_symbols:
                                    df_sym = data.get(sym)
                                    if df_sym is None or df_sym.empty:
                                        strict_recent_missing.append(sym)
                                        continue
                                    try:
                                        last_dt = pd.Timestamp(df_sym.index.max()).tz_localize(None).date()
                                    except Exception:
                                        strict_recent_missing.append(sym)
                                        continue
                                    if (now_et - last_dt).days <= max_end_lag_days:
                                        fresh_end += 1
                                    else:
                                        strict_recent_stale.append(sym)

                                strict_recent_cov = (
                                    fresh_end / float(len(end_scope_symbols))
                                    if end_scope_symbols
                                    else 0.0
                                )
                                strict_recent_refresh_needed = (
                                    strict_recent_cov < min_recent_cov_for_refresh
                                )
                                if strict_recent_refresh_needed:
                                    refresh_symbols.extend(strict_recent_missing)
                                    refresh_symbols.extend(strict_recent_stale)
                                    st.warning(
                                        "Strict mode freshness check: recent end-of-window coverage "
                                        f"{strict_recent_cov:.2%} is below {min_recent_cov_for_refresh:.0%}. "
                                        f"Refreshing {len(strict_recent_missing) + len(strict_recent_stale)} "
                                        "end-scope symbols (missing/stale)."
                                    )

                            # Deduplicate while preserving order.
                            refresh_symbols = list(dict.fromkeys(refresh_symbols))
                            allow_incremental_refresh = (
                                bool(refresh_symbols)
                                and incremental_refresh_enabled
                                and (
                                    market_open
                                    or allow_refresh_when_closed
                                    or initial_cov <= 0.05
                                    or strict_recent_refresh_needed
                                )
                            )

                            if allow_incremental_refresh:
                                if refresh_symbols and len(refresh_symbols) > refresh_cap:
                                    skipped = len(refresh_symbols) - refresh_cap
                                    refresh_symbols = refresh_symbols[:refresh_cap]
                                    st.warning(
                                        f"Refresh cap active: fetching first {len(refresh_symbols)} "
                                        f"symbols this run ({skipped} deferred)."
                                    )

                                if initial_cov < min_cache_coverage:
                                    st.warning(
                                        f"Cache coverage {initial_cov:.1%} below threshold "
                                        f"({min_cache_coverage:.0%}); refreshing {len(refresh_symbols)} symbols."
                                    )
                                elif strict_recent_refresh_needed:
                                    st.warning(
                                        "Cache coverage passed, but strict recent-coverage freshness failed; "
                                        f"refreshing {len(refresh_symbols)} symbols."
                                    )

                                if refresh_symbols:
                                    stage_msg.caption("Stage: Refreshing symbols incrementally...")
                                    quality_refresh = {}
                                    fresh = fetch_data_pack(
                                        refresh_symbols,
                                        days=fetch_days,
                                        backtest_mode=False,
                                        force_fresh=force_fresh_refresh,
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
                            elif refresh_symbols:
                                st.warning(
                                    f"Refresh needed for {len(refresh_symbols)} symbols, but network refresh is skipped "
                                    "while market is closed. Set `APEX_BACKTEST_REFRESH_WHEN_CLOSED=1` to override."
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
                        if accuracy_mode == "block":
                            st.warning(
                                f"Cached prepared data ends on {loaded_end_dt.date().isoformat()} "
                                f"({end_lag} days stale). Refreshing for verified accuracy."
                            )
                            cache_hit = False
                            prepared = None
                        else:
                            st.warning(
                                f"Cached prepared data ends on {loaded_end_dt.date().isoformat()} "
                                f"({end_lag} days stale). Using cached data in warn mode. "
                                "Enable Verified run to force a fresh rebuild."
                            )

                if not cache_hit:
                    # Re-enter with fresh data for stale cache case.
                    stage_msg.caption("Stage: Refreshing stale prepared dataset...")
                    if bt_universe in ("Russell 3000", "RUSSELL3000"):
                        try:
                            symbols, universe_source = get_universe_symbols_pit_window_with_meta(
                                "RUSSELL3000",
                                bt_start_date,
                                bt_end_date,
                            )
                        except RuntimeError as e:
                            st.error(
                                "Point-in-time Russell 3000 membership is required for accurate backtests. "
                                "Configure `RUSSELL3000_PIT_DIR` or `RUSSELL3000_PIT_MEMBERSHIP_CSV`."
                            )
                            st.caption(f"Resolver detail: {e}")
                            st.session_state.backtest_results = {}
                            symbols = []
                            universe_source = "unavailable"
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
                thresholds = _accuracy_thresholds(
                    bt_universe,
                    accuracy_mode=accuracy_mode,
                    duration_label=bt_duration,
                )
                min_cov_required = float(thresholds.get("min_coverage", 0.0))
                if expected_symbol_count > 0:
                    tested_cov = tested_symbols / float(max(1, expected_symbol_count))
                    st.caption(
                        f"Universe coverage used in run: {tested_symbols}/{expected_symbol_count} "
                        f"({tested_cov:.2%})"
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
                    try:
                        end_members, end_source = get_universe_symbols_pit_with_meta("RUSSELL3000", bt_end_date)
                        if end_members:
                            recent_scope = {str(s).upper() for s in end_members if str(s).strip()}
                            st.caption(
                                f"Recent coverage scope: end-of-window PIT membership "
                                f"({len(recent_scope)} symbols, source `{end_source}`)."
                            )
                    except RuntimeError as e:
                        st.caption(f"Recent coverage scope unavailable (PIT resolver: {e}).")
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

                source_hits = quality_union.get("source_hits") or {}
                if source_hits:
                    parts = [f"{k}={v}" for k, v in sorted(source_hits.items())]
                    st.caption(f"Data source mix this run: {', '.join(parts)}")

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

                min_recent_cov_required = float(thresholds.get("min_recent_coverage", 0.0))
                max_incomplete_ratio = float(thresholds.get("max_incomplete_ratio", 1.0))
                max_stale_ratio = float(thresholds.get("max_stale_ratio", 1.0))

                incomplete_base = max(1, len(loaded_set) + len(incomplete_set))
                incomplete_ratio = len(incomplete_set) / float(incomplete_base)
                loaded_upper = {str(s).upper() for s in loaded_set if str(s).strip()}
                stale_upper = {str(s).upper() for s in stale_set if str(s).strip()}
                if recent_scope:
                    recent_scope_upper = {str(s).upper() for s in recent_scope if str(s).strip()}
                else:
                    recent_scope_upper = set()
                if recent_scope_upper:
                    loaded_scope_upper = loaded_upper & recent_scope_upper
                    stale_scope_upper = stale_upper & loaded_scope_upper
                    stale_ratio_base = len(loaded_scope_upper)
                    stale_ratio_count = len(stale_scope_upper)
                    stale_ratio = (
                        stale_ratio_count / float(stale_ratio_base)
                        if stale_ratio_base > 0
                        else 0.0
                    )
                    stale_ratio_label = (
                        f"{stale_ratio_count}/{stale_ratio_base} "
                        f"end-scope loaded symbols ({stale_ratio:.1%})"
                    )
                    stale_basis = "end-scope loaded symbols"
                else:
                    stale_ratio_base = len(loaded_upper)
                    stale_ratio_count = len(stale_upper)
                    stale_ratio = (
                        stale_ratio_count / float(max(1, stale_ratio_base))
                    )
                    stale_ratio_label = (
                        f"{stale_ratio_count}/{stale_ratio_base} "
                        f"loaded symbols ({stale_ratio:.1%})"
                    )
                    stale_basis = "loaded symbols"
                spy_ok = bool(global_data.get("SPY") is not None and not global_data.get("SPY").empty)
                vix_ok = bool(global_data.get("VIX") is not None and not global_data.get("VIX").empty)
                recent_cov_pass = (fresh_total <= 0) or (fresh_cov >= min_recent_cov_required)
                incomplete_pass = incomplete_ratio <= max_incomplete_ratio
                stale_pass = stale_ratio <= max_stale_ratio
                coverage_pass = not (expected_symbol_count > 0 and tested_cov < min_cov_required)
                lock_blockers = []
                if not coverage_pass:
                    lock_blockers.append(
                        f"coverage {tested_cov:.2%} below required {min_cov_required:.0%}"
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
                        f"stale-data ratio {stale_ratio:.1%} above {max_stale_ratio:.0%} ({stale_basis})"
                    )
                if not spy_ok:
                    lock_blockers.append("SPY market context missing")

                scorecard_rows = [
                    {
                        "Metric": "Universe coverage",
                        "Value": f"{tested_symbols}/{expected_symbol_count} ({tested_cov:.2%})",
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
                        "Value": stale_ratio_label,
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
                st.dataframe(pd.DataFrame(scorecard_rows), width="stretch", hide_index=True)

                quality_gate_failed = len(lock_blockers) > 0
                if quality_gate_failed:
                    blocker_msg = "; ".join(lock_blockers)
                    if accuracy_mode == "block":
                        show_blocked_results = _env_flag("APEX_BACKTEST_SHOW_BLOCKED_RESULTS", "1")
                        if show_blocked_results:
                            st.warning(
                                "Accuracy gate failed for strict mode: "
                                f"{blocker_msg}. Showing provisional performance only "
                                "(not baseline-eligible)."
                            )
                            _drop_backtest_cache(cache_key)
                        else:
                            st.error(
                                "Accuracy gate blocked this run: "
                                f"{blocker_msg}. Results withheld to avoid biased metrics."
                            )
                            if not coverage_pass:
                                st.caption(
                                    "Coverage gaps usually indicate data-provider constraints in PIT windows "
                                    "(commonly delisted/renamed symbols)."
                                )
                            _drop_backtest_cache(cache_key)
                            st.session_state.backtest_results = {}
                            prepared = None
                    elif accuracy_mode == "warn":
                        st.warning(
                            "Accuracy warning: run allowed in best-effort mode. "
                            f"Quality blockers: {blocker_msg}."
                        )
                        export_missing = _env_flag(
                            "APEX_EXPORT_MISSING_SYMBOLS_ON_LOW_COVERAGE",
                            "1",
                        )
                        if export_missing and expected_symbols and not coverage_pass:
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
                        # Force a rebuild on next run when critical freshness/coverage checks fail.
                        if (not coverage_pass) or (not recent_cov_pass) or (not stale_pass):
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
                        run_requires_pit_membership = False
                        membership_source = "none"
                        if bt_universe in ("Russell 3000", "RUSSELL3000"):
                            run_requires_pit_membership = bool(
                                require_pit_universe and run_accuracy_mode == "block"
                            )
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
                                timeline_msg = (
                                    "PIT timeline could not be constructed for this run."
                                )
                                if run_requires_pit_membership:
                                    st.error(
                                        f"{timeline_msg} Verified mode requires complete day-level PIT membership; "
                                        "run aborted."
                                    )
                                else:
                                    st.warning(
                                        f"{timeline_msg} Falling back to static start-window membership."
                                    )
                        start_time = time.time()
                        status_text.text(f"Running {len(run_strategies)} strategy simulation(s)...")
                        raw_results = []
                        if (
                            run_requires_pit_membership
                            and bt_universe in ("Russell 3000", "RUSSELL3000")
                            and universe_membership_by_day is None
                        ):
                            status_text.text("Backtest aborted: missing day-level PIT membership.")
                        else:
                            try:
                                raw_results = run_backtest(
                                    run_strategies,
                                    prepared,
                                    start_cash=100000.0,
                                    start_date=bt_start_date,
                                    global_data=global_data,
                                    universe_membership_by_day=universe_membership_by_day,
                                    require_pit_membership=run_requires_pit_membership,
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
                        "stale": int(stale_ratio_count),
                        "stale_basis": stale_basis,
                        "stale_ratio": float(stale_ratio),
                        "source_hits": dict(source_hits),
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
                first_trade_date = res.get("first_trade_date")
                active_cagr = res.get("active_period_cagr")
                active_days = int(res.get("active_period_days") or 0)
                if active_cagr is not None and pd.notna(active_cagr):
                    st.caption(
                        f"Active-Period CAGR: {float(active_cagr):.1%} "
                        f"(first trade: {first_trade_date or 'n/a'}, days: {active_days})"
                    )
                elif first_trade_date:
                    st.caption(f"First trade date: {first_trade_date}")

                audit = res.get("audit_report") or {}
                if isinstance(audit, dict) and audit:
                    with st.expander("Execution Audit (Instrumentation-Only)", expanded=False):
                        a1, a2, a3 = st.columns(3)
                        a1.metric("Same-Day Open Entries", int(audit.get("same_day_open_entries", 0) or 0))
                        a2.metric("Stale Position-Days", int(audit.get("stale_position_days", 0) or 0))
                        a3.metric(
                            "Max Gross Exposure",
                            f"{float(audit.get('max_gross_exposure_pct', 0.0) or 0.0):.1%}",
                        )
                        st.caption(
                            f"Max gross date: {audit.get('max_gross_exposure_date', 'N/A')} | "
                            f"Same-day symbols: {int(audit.get('same_day_open_symbol_count', 0) or 0)} | "
                            f"Stale symbols: {int(audit.get('stale_position_symbol_count', 0) or 0)}"
                        )
                        daily_rows = audit.get("gross_exposure_daily") or []
                        if isinstance(daily_rows, list) and daily_rows:
                            tail = daily_rows[-10:]
                            st.dataframe(pd.DataFrame(tail), width="stretch", hide_index=True)

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
                        "locked_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
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
                    st.dataframe(pd.DataFrame(checks), width="stretch", hide_index=True)

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
                        export_meta = _persist_streamlit_bytes(
                            f"{safe_name}_backtest.csv",
                            csv_data,
                        )
                        st.download_button(
                            label="📥 Export Result (CSV)",
                            data=csv_data,
                            file_name=f"{safe_name}_backtest.csv",
                            mime="text/csv",
                            key=f"dl_{safe_name}_{i}"  # Ensure 'i' comes from the loop variable
                        )
                        _render_streamlit_export_status(export_meta)
                    except Exception as e:
                        st.error(f"⚠️ Export failed for {strategy_name}: {e}")
                else:
                    st.warning(f"⚠️ No equity data for {strategy_name}")

# --- 3. SIMULATOR (PRO MODE) ---
elif mode == "Simulator":
    if primary_strategy == "ETF Benchmark":
        _render_etf_simulator(selected_etf_profile_label)
        st.stop()
    if primary_strategy == "Hybrid Benchmark":
        _render_hybrid_benchmark_simulator(selected_hybrid_profile_label)
        st.stop()
    if primary_strategy == "Stock Benchmark":
        _render_stock_benchmark_simulator(selected_stock_benchmark_label)
        st.stop()
    render_mode_header(
        "🎮 Apex Swing Paper Trader",
        "Practice the intended workflow: after-close scan, next-open entries, and disciplined swing management.",
    )
    sim_universe = st.selectbox(
        "Universe",
        UNIVERSE_OPTIONS,
        index=UNIVERSE_OPTIONS.index(DEFAULT_UNIVERSE),
        key="sim_universe",
    )
    pt = PaperTrader(configs=selected_strategies if selected_strategies else None)
    state = pt.state
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equity", f"${state['equity']:,.2f}")
    m2.metric("Cash", f"${state['cash']:,.2f}")
    pnl_val = state['equity'] - 100000.0
    m3.metric("Total PnL", f"${pnl_val:,.2f}", delta=f"{pnl_val/100000*100:.2f}%")
    m4.metric("Positions", len(state['positions']))
    
    if "sim_reset_confirm_armed" not in st.session_state:
        st.session_state.sim_reset_confirm_armed = False

    if st.button(
        "🔭 PHASE 1: Scan for New Entries (Evening/Market Close)",
        type="primary",
        disabled=not bool(selected_strategies),
        help=None if selected_strategies else "Select at least one strategy in the sidebar.",
    ):
        status = st.status("🚀 Initializing Simulation...", expanded=True)
        try:
            status.write("1️⃣ Verifying Strategies...")
            if not getattr(pt, 'strategies', None):
                status.update(label="❌ No Strategies Loaded!", state="error")
                st.error("PaperTrader has 0 loaded strategies. Check config.")
                st.stop()
            
            status.write(f"2️⃣ Resolving {sim_universe} constituents...")
            if sim_universe in ("Russell 3000", "RUSSELL3000"):
                sim_as_of = datetime.now(ZoneInfo("UTC")).date().isoformat()
                try:
                    symbols, sim_universe_source = get_universe_symbols_pit_with_meta(
                        "RUSSELL3000",
                        sim_as_of,
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Point-in-time Russell 3000 membership data is required for simulation: {e}"
                    ) from e
            else:
                symbols = get_universe_symbols(sim_universe)
                sim_universe_source = "current_index"
            symbols = list(symbols or [])
            if not symbols:
                raise RuntimeError(f"No symbols loaded for {sim_universe}.")
            status.write(
                f"3️⃣ Executing Scan & Governor ({sim_universe}, {len(symbols):,} symbols, source: {sim_universe_source})..."
            )
            base_days = 400
            data_pack = fetch_data_pack(
                symbols,
                days=base_days,
                max_workers=_recommended_fetch_workers(len(symbols), cache_only=False),
                inject_live=True,
                max_lag_days=0,
            )
            g_data = fetch_data_pack(
                ["SPY", "$VIX", "VIX"],
                days=600,
                max_workers=_recommended_fetch_workers(3, cache_only=False),
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
                universe_label=f"{sim_universe} ({sim_universe_source})",
            )
            orders = scan_result.get("orders", []) if scan_result else []
            logs = scan_result.get("logs", []) if scan_result else []
            queued_count = scan_result.get("count", len(orders)) if scan_result else 0
            status_text.empty()
            progress_bar.empty()
            
            status.write(f"4️⃣ Scan Complete. Orders Queued: {queued_count}")
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
        if not st.session_state.sim_reset_confirm_armed:
            if st.button("⚠️ Emergency System Reset (Wipe All State)", key="sim_reset_arm_btn"):
                st.session_state.sim_reset_confirm_armed = True
                st.rerun()
        else:
            st.error("Confirm full simulator reset. This will wipe positions, pending orders, and trade history.")
            x1, x2 = st.columns(2)
            with x1:
                if st.button("✅ Confirm Wipe", key="sim_reset_confirm_btn", type="primary"):
                    pt.reset_account()
                    st.session_state.sim_reset_confirm_armed = False
                    st.success("Simulator state reset.")
                    st.rerun()
            with x2:
                if st.button("Cancel", key="sim_reset_cancel_btn"):
                    st.session_state.sim_reset_confirm_armed = False
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

            if c_cols[8].button("SELL", key=f"sell_{sym}", width="stretch"):
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
            if c_cols[5].button("CANCEL", key=f"cancel_{sym}_{idx}", width="stretch"):
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
                        st.dataframe(initial_buys, width="stretch")
                st.dataframe(history_df.tail(20), width="stretch")
                csv_data = history_df.to_csv(index=False).encode("utf-8")
                export_meta = _persist_streamlit_bytes("sim_trade_history.csv", csv_data)
                st.download_button(
                    label="📥 Export Performance Audit (CSV)",
                    data=csv_data,
                    file_name="sim_trade_history.csv",
                    mime="text/csv",
                )
                _render_streamlit_export_status(export_meta)
    else:
        st.caption("No closed trades yet.")
