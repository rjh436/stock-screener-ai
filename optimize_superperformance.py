import os
import sys
import json
import random
import time
import pandas as pd
import numpy as np
import concurrent.futures
import multiprocessing
import pickle
import gc
import subprocess
from pathlib import Path

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

# --- IMPORTS ---
try:
    from execution.engine import prepare_backtest_data, run_backtest
    from data.loader import fetch_data_pack
    from data.universe import (
        get_universe_symbols,
        get_universe_symbols_pit_window_with_meta,
        build_russell3000_membership_by_day,
    )
    from strategies.superperformance import SuperperformanceStrategy
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
RESULTS_FILE = "superperformance_results.csv"
BEST_GENOME_FILE = "config/superperformance_winner.json"
LAST_METRICS_FILE = "config/superperformance_last_metrics.json"
POPULATION_SIZE = int(os.getenv("APEX_POPULATION_SIZE", "30") or "30")
GENERATIONS = int(os.getenv("APEX_GENERATIONS", "6") or "6")
_DEFAULT_END_TS = pd.Timestamp.today().normalize()
_DEFAULT_END_DATE = _DEFAULT_END_TS.strftime("%Y-%m-%d")
_DEFAULT_START_DATE = (_DEFAULT_END_TS - pd.DateOffset(years=20)).strftime("%Y-%m-%d")
START_DATE = str(os.getenv("APEX_START_DATE", _DEFAULT_START_DATE) or _DEFAULT_START_DATE)
END_DATE = str(os.getenv("APEX_END_DATE", _DEFAULT_END_DATE) or _DEFAULT_END_DATE)
MAX_WORKERS = int(os.getenv("APEX_MAX_WORKERS", "10") or "10")
CHECKPOINT_FILE = "optimizer_checkpoint_sp.pkl" 
MAX_DD_CAP = float(os.getenv("APEX_MAX_DD_CAP", "30.0") or "30.0")
_DEFAULT_MP_START = "fork" if sys.platform == "darwin" else "spawn"
MP_START_METHOD = str(os.getenv("APEX_MP_START_METHOD", _DEFAULT_MP_START) or _DEFAULT_MP_START).strip().lower()
if MP_START_METHOD not in {"spawn", "fork", "forkserver"}:
    MP_START_METHOD = _DEFAULT_MP_START
_DEFAULT_WORKER_HARD_CAP = "6" if sys.platform == "darwin" else "0"
MAX_WORKERS_HARD_CAP = int(os.getenv("APEX_MAX_WORKERS_HARD_CAP", _DEFAULT_WORKER_HARD_CAP) or _DEFAULT_WORKER_HARD_CAP)
POOL_MAX_TASKS_PER_CHILD = int(os.getenv("APEX_POOL_MAX_TASKS_PER_CHILD", "4") or "4")
MEMORY_UTILIZATION = float(os.getenv("APEX_MEMORY_UTILIZATION", "0.72") or "0.72")
MEMORY_BUDGET_GB = float(os.getenv("APEX_MEMORY_BUDGET_GB", "0") or "0")
MEMORY_HEADROOM_GB = float(os.getenv("APEX_MEMORY_HEADROOM_GB", "6.0") or "6.0")
_DEFAULT_COPY_MULTIPLIER = "1.0"
WORKER_DATA_COPY_MULTIPLIER = float(
    os.getenv("APEX_WORKER_DATA_COPY_MULTIPLIER", _DEFAULT_COPY_MULTIPLIER) or _DEFAULT_COPY_MULTIPLIER
)
# Pruning can remove strategy-critical columns and silently produce zero-entry runs.
# Keep it opt-in via env override.
PRUNE_PREPARED_DF = str(os.getenv("APEX_PRUNE_PREPARED_DF", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RESUME_CHECKPOINT = str(os.getenv("APEX_RESUME_CHECKPOINT", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
OBJECTIVE_PROFILE = str(os.getenv("APEX_OBJECTIVE_PROFILE", "no_leverage") or "no_leverage").strip().lower()
if OBJECTIVE_PROFILE not in {"superperformance", "balanced", "defensive", "no_leverage"}:
    OBJECTIVE_PROFILE = "no_leverage"
if OBJECTIVE_PROFILE == "superperformance":
    _TARGET_DEFAULT = "35.0"
    _MIN_CAGR_DEFAULT = "20.0"
elif OBJECTIVE_PROFILE == "no_leverage":
    _TARGET_DEFAULT = "24.0"
    _MIN_CAGR_DEFAULT = "14.0"
else:
    _TARGET_DEFAULT = "25.0"
    _MIN_CAGR_DEFAULT = "10.0"
TARGET_CAGR = float(os.getenv("APEX_TARGET_CAGR", _TARGET_DEFAULT) or _TARGET_DEFAULT)
MIN_CAGR_FLOOR = float(os.getenv("APEX_MIN_CAGR_FLOOR", _MIN_CAGR_DEFAULT) or _MIN_CAGR_DEFAULT)
MIN_PF_FLOOR = float(os.getenv("APEX_MIN_PF_FLOOR", "1.20") or "1.20")
MIN_WINLOSS_RATIO = float(os.getenv("APEX_MIN_WINLOSS_RATIO", "2.50") or "2.50")
MIN_TRADES_FLOOR = int(os.getenv("APEX_MIN_TRADES_FLOOR", "50") or "50")
if OBJECTIVE_PROFILE == "no_leverage":
    _MIN_TRADES_HARD_DEFAULT = "40"
    _MIN_ACTIVE_YEARS_DEFAULT = "6.0"
else:
    _MIN_TRADES_HARD_DEFAULT = "25"
    _MIN_ACTIVE_YEARS_DEFAULT = "4.0"
MIN_TRADES_HARD_FLOOR = int(
    os.getenv("APEX_MIN_TRADES_HARD_FLOOR", _MIN_TRADES_HARD_DEFAULT) or _MIN_TRADES_HARD_DEFAULT
)
MIN_ACTIVE_YEARS = float(
    os.getenv("APEX_MIN_ACTIVE_YEARS", _MIN_ACTIVE_YEARS_DEFAULT) or _MIN_ACTIVE_YEARS_DEFAULT
)
# Legacy total-trade cap. Default off in favor of annualized trade velocity shaping.
MAX_TRADES_SOFT = int(os.getenv("APEX_MAX_TRADES_SOFT", "0") or "0")
MIN_TRADES_PER_YEAR_FLOOR = float(os.getenv("APEX_MIN_TRADES_PER_YEAR_FLOOR", "15.0") or "15.0")
TARGET_TRADES_PER_YEAR_MIN = float(os.getenv("APEX_TARGET_TRADES_PER_YEAR_MIN", "25.0") or "25.0")
TARGET_TRADES_PER_YEAR_MAX = float(os.getenv("APEX_TARGET_TRADES_PER_YEAR_MAX", "120.0") or "120.0")
MAX_TRADES_PER_YEAR_SOFT = float(os.getenv("APEX_MAX_TRADES_PER_YEAR_SOFT", "200.0") or "200.0")
TRADES_PER_YEAR_LOW_PENALTY_MULT = float(
    os.getenv("APEX_TRADES_PER_YEAR_LOW_PENALTY_MULT", "6.0") or "6.0"
)
TRADES_PER_YEAR_HIGH_PENALTY_MULT = float(
    os.getenv("APEX_TRADES_PER_YEAR_HIGH_PENALTY_MULT", "0.8") or "0.8"
)
TRADES_PER_YEAR_TARGET_BONUS = float(os.getenv("APEX_TRADES_PER_YEAR_TARGET_BONUS", "12.0") or "12.0")
OPTIMIZER_COST_BPS = float(os.getenv("APEX_OPTIMIZER_COST_BPS", "2.0") or "2.0")
OPTIMIZER_ENTRY_SLIPPAGE_BPS = float(
    os.getenv("APEX_OPTIMIZER_ENTRY_SLIPPAGE_BPS", "4.0") or "4.0"
)
OPTIMIZER_EXIT_SLIPPAGE_BPS = float(
    os.getenv("APEX_OPTIMIZER_EXIT_SLIPPAGE_BPS", "6.0") or "6.0"
)
OPTIMIZER_TRACE_REJECTS = str(
    os.getenv("APEX_OPTIMIZER_TRACE_REJECTS", "0") or "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
}
OPTIMIZER_TRACE_MAX_LINES = int(os.getenv("APEX_OPTIMIZER_TRACE_MAX_LINES", "5000") or "5000")
GENOME_TIMEOUT_MIN = float(os.getenv("APEX_GENOME_TIMEOUT_MIN", "30") or "30")
if GENOME_TIMEOUT_MIN < 0:
    GENOME_TIMEOUT_MIN = 0.0
IMMIGRANT_FRAC = float(os.getenv("APEX_IMMIGRANT_FRAC", "0.30") or "0.30")
ELITE_COUNT = int(os.getenv("APEX_ELITE_COUNT", "5") or "5")
RECENT_5Y_CAGR_FLOOR = float(os.getenv("APEX_RECENT_5Y_CAGR_FLOOR", "15.0") or "15.0")
RECENT_3Y_CAGR_FLOOR = float(os.getenv("APEX_RECENT_3Y_CAGR_FLOOR", "16.0") or "16.0")
RECENT_5Y_CAGR_TARGET = float(os.getenv("APEX_RECENT_5Y_CAGR_TARGET", "20.0") or "20.0")
RECENT_3Y_CAGR_TARGET = float(os.getenv("APEX_RECENT_3Y_CAGR_TARGET", "22.0") or "22.0")
RECENT_5Y_WEIGHT = float(os.getenv("APEX_RECENT_5Y_WEIGHT", "0.35") or "0.35")
RECENT_3Y_WEIGHT = float(os.getenv("APEX_RECENT_3Y_WEIGHT", "0.20") or "0.20")
RECENT_5Y_PENALTY_MULT = float(os.getenv("APEX_RECENT_5Y_PENALTY_MULT", "4.0") or "4.0")
RECENT_3Y_PENALTY_MULT = float(os.getenv("APEX_RECENT_3Y_PENALTY_MULT", "3.0") or "3.0")
RECENT_5Y_TARGET_BONUS = float(os.getenv("APEX_RECENT_5Y_TARGET_BONUS", "0.40") or "0.40")
RECENT_3Y_TARGET_BONUS = float(os.getenv("APEX_RECENT_3Y_TARGET_BONUS", "0.35") or "0.35")
WORST_12M_FLOOR_PCT = float(os.getenv("APEX_WORST_12M_FLOOR_PCT", "-20.0") or "-20.0")
WORST_12M_PENALTY_MULT = float(os.getenv("APEX_WORST_12M_PENALTY_MULT", "2.5") or "2.5")
WORST_24M_CAGR_FLOOR = float(os.getenv("APEX_WORST_24M_CAGR_FLOOR", "8.0") or "8.0")
WORST_24M_CAGR_PENALTY_MULT = float(os.getenv("APEX_WORST_24M_CAGR_PENALTY_MULT", "2.0") or "2.0")
MEDIAN_24M_CAGR_FLOOR = float(os.getenv("APEX_MEDIAN_24M_CAGR_FLOOR", "12.0") or "12.0")
MEDIAN_24M_CAGR_PENALTY_MULT = float(os.getenv("APEX_MEDIAN_24M_CAGR_PENALTY_MULT", "1.5") or "1.5")
MAX_SINGLE_YEAR_PNL_SHARE = float(os.getenv("APEX_MAX_SINGLE_YEAR_PNL_SHARE", "0.35") or "0.35")
YEAR_CONCENTRATION_PENALTY_MULT = float(os.getenv("APEX_YEAR_CONCENTRATION_PENALTY_MULT", "220.0") or "220.0")
MAX_TWO_YEAR_PNL_SHARE = float(os.getenv("APEX_MAX_TWO_YEAR_PNL_SHARE", "0.55") or "0.55")
TWO_YEAR_CONCENTRATION_PENALTY_MULT = float(os.getenv("APEX_TWO_YEAR_CONCENTRATION_PENALTY_MULT", "180.0") or "180.0")
NEGATIVE_YEAR_SOFT_CAP = int(os.getenv("APEX_NEGATIVE_YEAR_SOFT_CAP", "2") or "2")
NEGATIVE_YEAR_PENALTY = float(os.getenv("APEX_NEGATIVE_YEAR_PENALTY", "4.0") or "4.0")
STITCHED_OOS_TRAIN_YEARS = int(os.getenv("APEX_STITCHED_OOS_TRAIN_YEARS", "3") or "3")
STITCHED_OOS_TEST_YEARS = int(os.getenv("APEX_STITCHED_OOS_TEST_YEARS", "1") or "1")
STITCHED_OOS_CAGR_FLOOR = float(os.getenv("APEX_STITCHED_OOS_CAGR_FLOOR", "15.0") or "15.0")
STITCHED_OOS_TARGET = float(os.getenv("APEX_STITCHED_OOS_TARGET", "18.0") or "18.0")
STITCHED_OOS_WEIGHT = float(os.getenv("APEX_STITCHED_OOS_WEIGHT", "2.25") or "2.25")
STITCHED_OOS_PENALTY_MULT = float(os.getenv("APEX_STITCHED_OOS_PENALTY_MULT", "6.0") or "6.0")
STITCHED_OOS_TARGET_BONUS = float(os.getenv("APEX_STITCHED_OOS_TARGET_BONUS", "0.60") or "0.60")
CALMAR_WEIGHT = float(os.getenv("APEX_CALMAR_WEIGHT", "5.0") or "5.0")
CALMAR_FLOOR = float(os.getenv("APEX_CALMAR_FLOOR", "0.85") or "0.85")
CALMAR_TARGET = float(os.getenv("APEX_CALMAR_TARGET", "1.25") or "1.25")
CALMAR_FLOOR_PENALTY_MULT = float(os.getenv("APEX_CALMAR_FLOOR_PENALTY_MULT", "25.0") or "25.0")
CALMAR_TARGET_BONUS = float(os.getenv("APEX_CALMAR_TARGET_BONUS", "8.0") or "8.0")
DD_PENALTY_START_SOFT = float(os.getenv("APEX_DD_PENALTY_START_SOFT", "18.0") or "18.0")
DD_PENALTY_START_HARD = float(os.getenv("APEX_DD_PENALTY_START_HARD", "22.0") or "22.0")
DD_PENALTY_START_CLIFF = float(os.getenv("APEX_DD_PENALTY_START_CLIFF", "27.0") or "27.0")
DD_PENALTY_MULT_SOFT = float(os.getenv("APEX_DD_PENALTY_MULT_SOFT", "0.75") or "0.75")
DD_PENALTY_MULT_HARD = float(os.getenv("APEX_DD_PENALTY_MULT_HARD", "1.75") or "1.75")
DD_PENALTY_MULT_CLIFF = float(os.getenv("APEX_DD_PENALTY_MULT_CLIFF", "4.5") or "4.5")
FUNDAMENTAL_PARQUET_DIR = str(
    os.getenv("APEX_FUNDAMENTAL_PARQUET_DIR", os.path.join("data", "fundamentals", "edgar_income"))
)
MIN_FUNDAMENTAL_COVERAGE = float(os.getenv("APEX_MIN_FUNDAMENTAL_COVERAGE", "0.75") or "0.75")
ALLOW_LOW_FUND_COVERAGE = str(os.getenv("APEX_ALLOW_LOW_FUND_COVERAGE", "0") or "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
AUTO_LOAD_FUNDAMENTALS = str(os.getenv("APEX_AUTO_LOAD_FUNDAMENTALS", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIRE_PIT_UNIVERSE = str(os.getenv("APEX_REQUIRE_PIT_UNIVERSE", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
REQUIRE_PIT_DAY_MEMBERSHIP = str(os.getenv("APEX_REQUIRE_PIT_DAY_MEMBERSHIP", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}

# --- GENOME SPACE (Optimization Variables) ---
GENE_SPACE = {
    "require_trend": [True],
    "rs_gate_min": [85, 88, 90],
    "min_price": [4, 5, 8],
    "min_avg_volume_30": [50000, 100000, 200000],
    "fundamental_growth_min_pct": [10, 15, 20],
    "high_tight_flag_override_pct": [85, 90, 95],
    # Keep Gate+Archetype architecture in union mode; avoid optimizer collapsing to EP-only.
    "entry_mode": ["both"],
    "vcp_lookback_bars": [40, 60, 80],
    "vcp_extrema_order": [2, 3],
    "vcp_breakout_volume_mult": [1.5, 1.75, 2.0],
    "breakout_buffer": [0, 0.0005, 0.001, 0.002],
    "adr_min_pct": [2.0, 2.5, 3.0, 3.5],
    "prior_runup_min_pct": [10, 15, 20, 30],
    "ep_gap_pct": [8],
    "ep_vol_mult": [3, 4],
    "ep_close_near_high_min": [0.70, 0.75, 0.80, 0.85],
    "ep_entry_mode": ["close", "open"],
    "ep_max_stop_pct": [0.12, 0.15, 0.20],
    "score_mode": ['dual_core'],
    "technical_weight": [0.55, 0.65, 0.75],
    "fundamental_weight": [0.45, 0.35, 0.25],
    "min_entry_score": [30, 35, 45, 55],
    "max_stop_pct": [0.06, 0.08, 0.10, 0.12],
    "stop_limit_pct": [0.02, 0.03],
    "stop_loss_atr_bull": [3, 4, 5, 6],
    "stop_loss_atr_bear": [0.5, 0.75],
    "breakeven_at_pct": [0.20],
    "exit_sma_fast": ['sma50'],
    "exit_sma_slow": ['sma50'],
    "time_stop_days": [10, 12, 15, 20],
    "pyramid_threshold": [0.04, 0.06, 0.08, 0.10],
    "pyramid_fraction": [0.5, 0.67],
    "pyramid_max_adds": [1, 2, 3],
    "max_positions": [6, 8, 10, 12],
    "risk_per_trade": [0.015, 0.02, 0.025, 0.03],
    "max_pos_size_pct": [0.15, 0.20, 0.25, 0.30],
    "max_total_exposure_pct_bull": [0.8, 0.9, 1.0],
    "max_total_exposure_pct_bear": [0.0, 0.05, 0.10, 0.20, 0.30],
    "vcp_trigger_mode": ["close_confirmed", "setup"],
}

if OBJECTIVE_PROFILE == "no_leverage":
    GENE_SPACE.update(
        {
            "min_price": [6, 8, 10, 12],
            "min_avg_volume_30": [100000, 150000, 200000, 300000],
            "max_total_exposure_pct_bull": [0.8, 0.9, 1.0],
            "max_total_exposure_pct_bear": [0.0, 0.05],
            "max_positions": [4, 5, 6, 8],
            "risk_per_trade": [0.006, 0.008, 0.01, 0.012],
            "max_pos_size_pct": [0.12, 0.15, 0.18, 0.20, 0.22],
            "rs_gate_min": [80, 82, 85, 88, 90, 92],
            "fundamental_growth_min_pct": [10, 15, 20, 25],
            "min_entry_score": [25, 30, 35, 45, 55],
            "vcp_lookback_bars": [40, 60, 80],
            "breakout_buffer": [0.0, 0.0005, 0.001, 0.002],
            "adr_min_pct": [2.0, 2.5, 3.0],
            "prior_runup_min_pct": [10, 15, 20],
            "ep_gap_pct": [6, 8],
            "ep_close_near_high_min": [0.65, 0.70, 0.75, 0.80],
            "ep_entry_mode": ["close"],
            "ep_max_stop_pct": [0.10, 0.12, 0.15],
            "max_stop_pct": [0.05, 0.06, 0.08],
            "stop_limit_pct": [0.02, 0.03, 0.05],
            "stop_loss_atr_bull": [2.5, 3, 4, 5],
            "time_stop_days": [10, 12, 15, 20, 25],
            "pyramid_fraction": [0.5],
            "pyramid_max_adds": [1, 2],
            "trend_template_mode": ["classic"],
            "vcp_trigger_mode": ["setup", "close_confirmed"],
            "vcp_gap_chase_max_pct": [0.0, 0.01, 0.02, 0.03],
            "vcp_gap_chase_rs_min": [92, 95],
            "vcp_gap_chase_score_min": [60, 70],
        }
    )


def _edgar_partition_exists(symbol: str) -> bool:
    path = Path(FUNDAMENTAL_PARQUET_DIR) / f"ticker={str(symbol).upper()}" / "fundamentals.parquet"
    return path.exists()


def _fundamental_coverage(symbols):
    syms = [str(s).upper() for s in symbols or [] if str(s or "").strip()]
    if not syms:
        return 0.0, [], []
    covered = [s for s in syms if _edgar_partition_exists(s)]
    missing = [s for s in syms if s not in set(covered)]
    return (len(covered) / float(len(syms))), covered, missing


def _ensure_fundamental_coverage(symbols):
    coverage, covered, missing = _fundamental_coverage(symbols)
    print(
        f"📚 Fundamental coverage: {len(covered)}/{len(symbols)} ({coverage:.1%}) "
        f"from {FUNDAMENTAL_PARQUET_DIR}"
    )

    if coverage >= MIN_FUNDAMENTAL_COVERAGE:
        return

    if AUTO_LOAD_FUNDAMENTALS and missing:
        identity = str(os.getenv("SEC_EDGAR_IDENTITY", "") or "").strip()
        if identity:
            print(f"🧾 Loading missing fundamentals for {len(missing)} symbols from SEC EDGAR...")
            try:
                from data.fundamental_loader import LoaderConfig, refresh_fundamentals

                cfg = LoaderConfig(
                    output_dir=Path(FUNDAMENTAL_PARQUET_DIR),
                    identity=identity,
                    max_workers=int(os.getenv("APEX_FUND_MAX_WORKERS", "8") or "8"),
                    sec_rate_limit=int(os.getenv("APEX_FUND_RATE_LIMIT", "8") or "8"),
                    max_filings_per_ticker=int(os.getenv("APEX_FUND_MAX_FILINGS", "8") or "8"),
                    max_tasks_per_child=int(os.getenv("APEX_FUND_MAX_TASKS_PER_CHILD", "8") or "8"),
                    request_pause_sec=max(
                        0.0,
                        float(os.getenv("APEX_FUND_REQUEST_PAUSE_MS", "120") or "120") / 1000.0,
                    ),
                    overwrite=False,
                    verbose=True,
                )
                refresh_fundamentals(missing, cfg)
            except Exception as exc:
                print(f"⚠️ Fundamental auto-load failed: {exc}")
        else:
            print("⚠️ SEC_EDGAR_IDENTITY missing; skipping auto fundamental load.")

        coverage, covered, missing = _fundamental_coverage(symbols)
        print(
            f"📚 Fundamental coverage after load: {len(covered)}/{len(symbols)} ({coverage:.1%})"
        )

    if coverage < MIN_FUNDAMENTAL_COVERAGE and not ALLOW_LOW_FUND_COVERAGE:
        raise RuntimeError(
            f"Fundamental coverage too low: {coverage:.1%} < {MIN_FUNDAMENTAL_COVERAGE:.1%}. "
            "Set SEC_EDGAR_IDENTITY and rerun, or override with APEX_ALLOW_LOW_FUND_COVERAGE=1."
        )


def _bytes_to_gb(num_bytes):
    return float(num_bytes) / float(1024 ** 3)


def _system_total_ram_bytes():
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * phys_pages
        if total > 0:
            return int(total)
    except Exception:
        pass
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        total = int(out)
        if total > 0:
            return total
    except Exception:
        pass
    return 0


def _estimate_data_bytes(obj, seen=None):
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return 0
    seen.add(oid)

    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, pd.DataFrame):
        return int(obj.memory_usage(index=True, deep=True).sum())
    if isinstance(obj, pd.Series):
        return int(obj.memory_usage(index=True, deep=True))
    if isinstance(obj, dict):
        return int(sum(_estimate_data_bytes(v, seen) for v in obj.values()))
    if isinstance(obj, (list, tuple, set)):
        return int(sum(_estimate_data_bytes(v, seen) for v in obj))
    if hasattr(obj, "__dict__"):
        return int(sum(_estimate_data_bytes(v, seen) for v in vars(obj).values()))
    return 0


def _estimate_prepared_bytes(prepared_obj):
    if prepared_obj is None:
        return 0
    total = 0
    seen = set()
    enriched = getattr(prepared_obj, "enriched", {}) or {}
    for sym_data in enriched.values():
        df = getattr(sym_data, "df", None)
        if isinstance(df, pd.DataFrame):
            total += _estimate_data_bytes(df, seen)
        slots = getattr(type(sym_data), "__slots__", ())
        for attr in slots:
            if attr == "df":
                continue
            try:
                val = getattr(sym_data, attr)
            except Exception:
                continue
            if isinstance(val, np.ndarray):
                total += _estimate_data_bytes(val, seen)
    total += _estimate_data_bytes(getattr(prepared_obj, "all_dates", None), seen)
    return int(total)


def _resolve_effective_workers(requested_workers, prepared_bytes, global_bytes):
    cpu_cap = max(1, int(os.cpu_count() or 1))
    requested = max(1, min(int(requested_workers), cpu_cap))
    if MAX_WORKERS_HARD_CAP > 0:
        requested = min(requested, int(MAX_WORKERS_HARD_CAP))

    total_ram_bytes = _system_total_ram_bytes()
    if total_ram_bytes <= 0:
        return requested, {
            "cpu_cap": cpu_cap,
            "requested": requested,
            "effective": requested,
            "ram_total_gb": float("nan"),
            "ram_budget_gb": float("nan"),
            "dataset_gb": _bytes_to_gb(prepared_bytes + global_bytes),
            "per_worker_gb": float("nan"),
            "mem_limited_workers": requested,
        }

    util = min(max(float(MEMORY_UTILIZATION), 0.40), 0.95)
    budget_bytes = int(total_ram_bytes * util)
    if MEMORY_BUDGET_GB > 0:
        budget_bytes = min(budget_bytes, int(MEMORY_BUDGET_GB * (1024 ** 3)))

    dataset_bytes = max(1, int(prepared_bytes + global_bytes))
    reserve_bytes = int(max(1.0, float(MEMORY_HEADROOM_GB)) * (1024 ** 3))
    per_worker_bytes = max(
        int(dataset_bytes * max(0.15, float(WORKER_DATA_COPY_MULTIPLIER))),
        256 * 1024 * 1024,
    )
    coordinator_bytes = dataset_bytes + reserve_bytes
    remaining = max(0, budget_bytes - coordinator_bytes)
    mem_limited_workers = max(1, int(remaining // per_worker_bytes)) if per_worker_bytes > 0 else 1

    effective = max(1, min(requested, mem_limited_workers))
    return effective, {
        "cpu_cap": cpu_cap,
        "requested": requested,
        "effective": effective,
        "ram_total_gb": _bytes_to_gb(total_ram_bytes),
        "ram_budget_gb": _bytes_to_gb(budget_bytes),
        "dataset_gb": _bytes_to_gb(dataset_bytes),
        "per_worker_gb": _bytes_to_gb(per_worker_bytes),
        "mem_limited_workers": mem_limited_workers,
    }


def _prune_prepared_frames(prepared_obj):
    if prepared_obj is None or not getattr(prepared_obj, "enriched", None):
        return prepared_obj
    required_cols = {
        # Core OHLCV
        "open",
        "high",
        "low",
        "close",
        "volume",
        # Trend / volume fields used in strategy gate
        "sma10",
        "sma20",
        "sma50",
        "sma150",
        "sma200",
        "vol_ma30",
        "vol_ma50",
        "natr",
        "atr14",
        "clv",
        # RS / momentum proxies
        "rs_percentile",
        "relative_strength_percentile",
        "momentum_rank",
        "rs_rating",
        # 52w / price action override proxies
        "high_52w",
        "high52w",
        "price_action_percentile",
        "pa_percentile",
        "breakout_percentile",
        # Fundamental gate / dual-core scoring inputs
        "eps_growth_qoq",
        "eps_growth_yoy",
        "eps_yoy_growth_pct",
        "eps_yoy",
        "sales_growth_yoy",
        "revenue_yoy_growth_pct",
        "sales_yoy",
        "institutional_sponsorship",
        # Entry diagnostics and scoring
        "gap_pct",
        "vcp_tightness",
        # VCP candidate debug counters used in engine
        "range_pct_5",
        "range_pct_10",
        "range_pct_20",
        "range_pct_40",
    }
    dropped_total = 0
    kept_total = 0
    for s_data in prepared_obj.enriched.values():
        df = getattr(s_data, "df", None)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        current_cols = list(df.columns)
        drop_cols = [c for c in current_cols if c not in required_cols]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True, errors="ignore")
            dropped_total += len(drop_cols)
        kept_total += len(df.columns)
    print(
        f"🧹 Pruned per-symbol DataFrame columns: dropped={dropped_total}, "
        f"avg_kept={kept_total / max(len(prepared_obj.enriched), 1):.1f}"
    )
    return prepared_obj


# --- WORKER STATE (Initializer Pattern) ---
_worker_data = None
_worker_global_data = None
_worker_universe_membership_by_day = None

def save_checkpoint(generation, population, best_genome_so_far):
    try:
        checkpoint_data = {
            "generation": generation,
            "population": population,
            "best_genome": best_genome_so_far
        }
        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(checkpoint_data, f)
        print(f"💾 Checkpoint saved for Generation {generation}")
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "rb") as f:
                checkpoint_data = pickle.load(f)
            print(f"🚀 RESUMING from Generation {checkpoint_data['generation'] + 1}...")
            return checkpoint_data["generation"], checkpoint_data["population"], checkpoint_data.get("best_genome")
        except Exception as e:
            print(f"⚠️ Checkpoint found but failed to load: {e}")
            return None
    return None

def generate_random_genome():
    return {k: random.choice(v) for k, v in GENE_SPACE.items()}


def _coerce_gene_value(gene_key, raw_value):
    options = list(GENE_SPACE.get(gene_key, []))
    if not options:
        return raw_value
    if raw_value in options:
        return raw_value
    try:
        opt_num = [float(o) for o in options]
        raw_num = float(raw_value)
        nearest_idx = int(np.argmin([abs(o - raw_num) for o in opt_num]))
        return options[nearest_idx]
    except Exception:
        pass
    raw_text = str(raw_value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == raw_text:
            return opt
    return random.choice(options)


def _seed_population(population_size: int) -> list:
    seed_paths = [
        os.path.join("config", "superperformance_practical_no_leverage_optimized.json"),
        os.path.join("config", "superperformance_practical_eod_cash_v2.json"),
        os.path.join("config", "superperformance_practical_no_leverage_best_20260225_052809.json"),
        os.path.join("config", "superperformance_selective_candidate.json"),
    ]
    seeded = []
    seen = set()
    for path in seed_paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        genome = {}
        for key in GENE_SPACE.keys():
            if key in payload:
                genome[key] = _coerce_gene_value(key, payload.get(key))
            else:
                genome[key] = random.choice(GENE_SPACE[key])
        sig = _genome_signature(genome)
        if sig in seen:
            continue
        seen.add(sig)
        seeded.append(genome)
        if len(seeded) >= population_size:
            break
    return seeded

def mutate_genome(genome):
    new_genome = genome.copy()
    gene = random.choice(list(GENE_SPACE.keys()))
    new_genome[gene] = random.choice(GENE_SPACE[gene])
    return new_genome

def crossover(parent1, parent2):
    child = {}
    for key in GENE_SPACE.keys():
        child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
    return child


def _genome_signature(genome):
    try:
        return json.dumps(genome or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except Exception:
        return str(genome)


def _build_tasks_from_population(population, dd_cap):
    tasks = []
    seen = set()
    duplicates = 0
    for idx, genome in enumerate(population or []):
        sig = _genome_signature(genome)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        tasks.append((idx, genome, dd_cap))
    return tasks, duplicates

def compress_data(prepared_obj):
    print("🗜️  Compressing Data (float32)...")
    float_cols_cast = 0
    arrays_cast = 0
    ints_cast = 0
    for sym, s_data in prepared_obj.enriched.items():
        cols = s_data.df.select_dtypes(include=['float64']).columns
        if len(cols) > 0:
            s_data.df[cols] = s_data.df[cols].astype('float32')
            float_cols_cast += len(cols)
        try:
            for attr in getattr(type(s_data), "__slots__", ()):
                if attr == "df":
                    continue
                val = getattr(s_data, attr)
                if isinstance(val, np.ndarray) and val.dtype == np.float64:
                    setattr(s_data, attr, val.astype(np.float32, copy=False))
                    arrays_cast += 1
                elif isinstance(val, np.ndarray) and val.dtype == np.int64 and attr == "gidx":
                    setattr(s_data, attr, val.astype(np.int32, copy=False))
                    ints_cast += 1
        except Exception:
            pass
    print(
        f"🗜️  Downcast summary: float64 columns={float_cols_cast}, "
        f"float64 arrays={arrays_cast}, int64 arrays={ints_cast}"
    )
    return prepared_obj

def _init_worker(prepared_data_readonly, global_data_readonly, universe_membership_by_day=None):
    global _worker_data, _worker_global_data, _worker_universe_membership_by_day
    _worker_data = prepared_data_readonly
    _worker_global_data = global_data_readonly
    _worker_universe_membership_by_day = universe_membership_by_day


def _trade_asymmetry(trades_list):
    win_returns = []
    loss_returns = []
    for tr in trades_list or []:
        try:
            ret_pct = float(tr.get("Return %", 0.0) or 0.0)
        except Exception:
            ret_pct = 0.0
        if ret_pct > 0:
            win_returns.append(ret_pct)
        elif ret_pct < 0:
            loss_returns.append(abs(ret_pct))
    avg_win = float(np.mean(win_returns)) if win_returns else 0.0
    avg_loss = float(np.mean(loss_returns)) if loss_returns else 0.0
    if avg_loss > 0:
        ratio = avg_win / avg_loss
    else:
        ratio = float("inf") if avg_win > 0 else 0.0
    return avg_win, avg_loss, ratio


def _recent_cagr_pct(equity_curve, years):
    if not equity_curve:
        return float("nan")
    try:
        ec = pd.DataFrame(equity_curve)
    except Exception:
        return float("nan")
    if ec.empty or "Date" not in ec.columns or "Equity" not in ec.columns:
        return float("nan")
    try:
        ec["Date"] = pd.to_datetime(ec["Date"], errors="coerce")
        ec["Equity"] = pd.to_numeric(ec["Equity"], errors="coerce")
        ec = ec.dropna(subset=["Date", "Equity"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    except Exception:
        return float("nan")
    if len(ec) < 2:
        return float("nan")

    end_dt = ec["Date"].iloc[-1]
    start_cutoff = end_dt - pd.DateOffset(years=int(max(1, years)))
    leading = ec[ec["Date"] < start_cutoff].tail(1)
    window = pd.concat([leading, ec[ec["Date"] >= start_cutoff]], ignore_index=True)
    if len(window) < 2:
        return float("nan")

    start_eq = float(window["Equity"].iloc[0] or 0.0)
    end_eq = float(window["Equity"].iloc[-1] or 0.0)
    if start_eq <= 0 or end_eq <= 0:
        return float("nan")

    span_years = (window["Date"].iloc[-1] - window["Date"].iloc[0]).days / 365.25
    if span_years <= 0:
        return float("nan")
    return ((end_eq / start_eq) ** (1 / span_years) - 1.0) * 100.0


def _worst_rolling_12m_return_pct(equity_curve):
    if not equity_curve:
        return float("nan")
    try:
        ec = pd.DataFrame(equity_curve)
    except Exception:
        return float("nan")
    if ec.empty or "Date" not in ec.columns or "Equity" not in ec.columns:
        return float("nan")
    try:
        ec["Date"] = pd.to_datetime(ec["Date"], errors="coerce")
        ec["Equity"] = pd.to_numeric(ec["Equity"], errors="coerce")
        ec = ec.dropna(subset=["Date", "Equity"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    except Exception:
        return float("nan")
    if len(ec) < 252:
        return float("nan")

    equity = ec["Equity"].to_numpy(dtype=np.float64, copy=False)
    rolling_ret = (equity[252:] / equity[:-252]) - 1.0
    if rolling_ret.size == 0:
        return float("nan")
    return float(np.nanmin(rolling_ret) * 100.0)


def _equity_curve_df(equity_curve):
    if not equity_curve:
        return pd.DataFrame(columns=["Date", "Equity"])
    try:
        ec = pd.DataFrame(equity_curve)
    except Exception:
        return pd.DataFrame(columns=["Date", "Equity"])
    if ec.empty or "Date" not in ec.columns or "Equity" not in ec.columns:
        return pd.DataFrame(columns=["Date", "Equity"])
    try:
        ec["Date"] = pd.to_datetime(ec["Date"], errors="coerce")
        ec["Equity"] = pd.to_numeric(ec["Equity"], errors="coerce")
    except Exception:
        return pd.DataFrame(columns=["Date", "Equity"])
    ec = ec.dropna(subset=["Date", "Equity"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    if ec.empty:
        return pd.DataFrame(columns=["Date", "Equity"])
    return ec[["Date", "Equity"]].reset_index(drop=True)


def _calendar_year_returns_pct(equity_curve):
    ec = _equity_curve_df(equity_curve)
    if ec.empty:
        return {}
    ec["Year"] = ec["Date"].dt.year
    out = {}
    for year, grp in ec.groupby("Year"):
        if len(grp) < 2:
            continue
        start_eq = float(grp["Equity"].iloc[0] or 0.0)
        end_eq = float(grp["Equity"].iloc[-1] or 0.0)
        if start_eq <= 0 or end_eq <= 0:
            continue
        out[int(year)] = ((end_eq / start_eq) - 1.0) * 100.0
    return out


def _year_concentration_stats(equity_curve):
    yearly = _calendar_year_returns_pct(equity_curve)
    if not yearly:
        return {
            "max_positive_share": float("nan"),
            "top2_positive_share": float("nan"),
            "negative_years": 0,
            "total_years": 0,
            "worst_year_return_pct": float("nan"),
            "median_year_return_pct": float("nan"),
        }
    vals = list(yearly.values())
    positive = [max(0.0, float(v)) for v in vals]
    positive_total = float(sum(positive))
    top1_share = float("nan")
    top2_share = float("nan")
    if positive_total > 0:
        positive_sorted = sorted(positive, reverse=True)
        top1_share = float(positive_sorted[0] / positive_total)
        top2_share = float(sum(positive_sorted[:2]) / positive_total)
    negative_years = int(sum(1 for v in vals if float(v) < 0.0))
    return {
        "max_positive_share": top1_share,
        "top2_positive_share": top2_share,
        "negative_years": negative_years,
        "total_years": int(len(vals)),
        "worst_year_return_pct": float(min(vals)),
        "median_year_return_pct": float(np.median(vals)) if vals else float("nan"),
    }


def _rolling_window_cagr_stats(equity_curve, window_years=2, step_days=21):
    ec = _equity_curve_df(equity_curve)
    if ec.empty or len(ec) < 10:
        return float("nan"), float("nan"), 0

    dates = ec["Date"].to_numpy(dtype="datetime64[ns]")
    equity = ec["Equity"].to_numpy(dtype=np.float64, copy=False)
    if equity.size < 10:
        return float("nan"), float("nan"), 0

    window_days = max(90, int(round(float(window_years) * 365.25)))
    step = max(5, int(step_days))
    cagrs = []

    for start_idx in range(0, len(ec), step):
        start_eq = float(equity[start_idx])
        if not np.isfinite(start_eq) or start_eq <= 0:
            continue
        target_date = dates[start_idx] + np.timedelta64(window_days, "D")
        end_idx = int(np.searchsorted(dates, target_date, side="left"))
        if end_idx >= len(ec):
            break
        end_eq = float(equity[end_idx])
        if not np.isfinite(end_eq) or end_eq <= 0:
            continue
        cagr_pct = ((end_eq / start_eq) ** (1.0 / float(window_years)) - 1.0) * 100.0
        if np.isfinite(cagr_pct):
            cagrs.append(float(cagr_pct))

    if not cagrs:
        return float("nan"), float("nan"), 0
    return float(min(cagrs)), float(np.median(cagrs)), int(len(cagrs))


def _stitched_oos_cagr_pct(equity_curve, train_years=3, test_years=1):
    ec = _equity_curve_df(equity_curve)
    if ec.empty or len(ec) < 20:
        return float("nan"), 0, float("nan")
    if train_years < 1 or test_years < 1:
        return float("nan"), 0, float("nan")

    ec = ec.copy()
    dates = ec["Date"].to_numpy(dtype="datetime64[ns]")
    equity = ec["Equity"].to_numpy(dtype=np.float64, copy=False)

    start_date = pd.Timestamp(dates[0])
    end_date = pd.Timestamp(dates[-1])
    test_start = start_date + pd.DateOffset(years=int(train_years))
    if test_start >= end_date:
        return float("nan"), 0, float("nan")

    fold_returns = []
    while True:
        test_end = test_start + pd.DateOffset(years=int(test_years))
        if test_end > end_date:
            break
        test_start_np = np.datetime64(test_start.to_datetime64())
        test_end_np = np.datetime64(test_end.to_datetime64())

        start_idx = int(np.searchsorted(dates, test_start_np, side="right")) - 1
        end_idx = int(np.searchsorted(dates, test_end_np, side="right")) - 1
        if start_idx < 0 or end_idx <= start_idx or end_idx >= len(equity):
            test_start = test_start + pd.DateOffset(years=1)
            continue

        start_eq = float(equity[start_idx])
        end_eq = float(equity[end_idx])
        if start_eq > 0 and end_eq > 0 and np.isfinite(start_eq) and np.isfinite(end_eq):
            fold_returns.append((end_eq / start_eq) - 1.0)
        test_start = test_start + pd.DateOffset(years=1)

    if not fold_returns:
        return float("nan"), 0, float("nan")

    stitched_mult = 1.0
    for ret in fold_returns:
        stitched_mult *= (1.0 + float(ret))
    total_test_years = float(len(fold_returns) * int(test_years))
    if stitched_mult <= 0 or total_test_years <= 0:
        return float("nan"), len(fold_returns), float("nan")

    stitched_cagr_pct = ((stitched_mult ** (1.0 / total_test_years)) - 1.0) * 100.0
    worst_fold_ret_pct = float(min(fold_returns) * 100.0)
    return float(stitched_cagr_pct), int(len(fold_returns)), worst_fold_ret_pct


def evaluate_genome(genome_id_and_genome):
    try:
        genome_id, genome, max_dd_cap = genome_id_and_genome
    except Exception:
        genome_id, genome = genome_id_and_genome
        max_dd_cap = MAX_DD_CAP
    global _worker_data, _worker_global_data, _worker_universe_membership_by_day
    data = _worker_data
    global_data = _worker_global_data
    universe_membership_by_day = _worker_universe_membership_by_day
    
    if data is None:
        return {"id": genome_id, "score": -999, "error": "Init failed"}

    try:
        # Construct Strategy Config
        strategy_config = dict(genome)
        genome_adj = dict(strategy_config)
        strategy_config["name"] = f"Gen_{genome_id}"
        # Keep the optimizer in Gate+Archetype union mode.
        strategy_config["entry_mode"] = "both"
        genome_adj["entry_mode"] = "both"
        # Hard clamp: cash-only, no leverage.
        max_exposure = float(strategy_config.get("max_total_exposure_pct_bull", 1.0) or 1.0)
        max_exposure = min(max(max_exposure, 0.0), 1.0)
        strategy_config["max_total_exposure_pct_bull"] = max_exposure
        genome_adj["max_total_exposure_pct_bull"] = max_exposure
        bear_exposure = float(strategy_config.get("max_total_exposure_pct_bear", 0.0) or 0.0)
        bear_exposure = min(max(bear_exposure, 0.0), 1.0)
        strategy_config["max_total_exposure_pct_bear"] = bear_exposure
        genome_adj["max_total_exposure_pct_bear"] = bear_exposure
        max_positions = int(strategy_config.get("max_positions", 1) or 1)
        max_pos_size = float(strategy_config.get("max_pos_size_pct", 1.0) or 1.0)
        max_pos_size = min(max(max_pos_size, 0.0), 1.0)
        if max_positions > 0 and (max_positions * max_pos_size) > max_exposure:
            max_pos_size = max_exposure / max_positions
            strategy_config["max_pos_size_pct"] = max_pos_size
            genome_adj["max_pos_size_pct"] = max_pos_size
        risk_per_trade = float(strategy_config.get("risk_per_trade", 0.01) or 0.01)
        if not np.isfinite(risk_per_trade):
            risk_per_trade = 0.01
        max_stop_pct = float(strategy_config.get("max_stop_pct", 0.08) or 0.08)
        ep_max_stop_pct = float(strategy_config.get("ep_max_stop_pct", 0.12) or 0.12)
        if OBJECTIVE_PROFILE == "no_leverage":
            if max_exposure >= 0.95 and max_positions <= 4:
                max_pos_size = min(max_pos_size, 0.20)
            if max_stop_pct >= 0.08:
                risk_per_trade = min(risk_per_trade, 0.01)
                max_pos_size = min(max_pos_size, 0.18)
            elif max_stop_pct >= 0.06:
                risk_per_trade = min(risk_per_trade, 0.012)
            if ep_max_stop_pct >= 0.15:
                risk_per_trade = min(risk_per_trade, 0.01)
            strategy_config["risk_per_trade"] = risk_per_trade
            genome_adj["risk_per_trade"] = risk_per_trade
            strategy_config["max_pos_size_pct"] = max_pos_size
            genome_adj["max_pos_size_pct"] = max_pos_size
        # After-close scan, next-day stop order (EP overrides with same-day close)
        strategy_config["signal_mode"] = "after_close"
        entry_day_stop_mode = str(
            os.getenv(
                "APEX_ENTRY_DAY_STOP_MODE",
                strategy_config.get("entry_day_stop_mode", "close_confirmed"),
            )
            or "close_confirmed"
        ).strip().lower()
        if entry_day_stop_mode not in {"off", "none", "disabled", "gap_only", "gap", "close_confirmed", "close", "intraday_low", "conservative"}:
            entry_day_stop_mode = "close_confirmed"
        strategy_config["entry_day_stop_mode"] = entry_day_stop_mode
        genome_adj["entry_day_stop_mode"] = entry_day_stop_mode
        if np.isfinite(OPTIMIZER_COST_BPS) and OPTIMIZER_COST_BPS > 0:
            strategy_config["transaction_cost_bps"] = float(max(0.0, OPTIMIZER_COST_BPS))
            strategy_config["slippage_bps"] = float(
                max(
                    0.0,
                    (
                        max(0.0, OPTIMIZER_ENTRY_SLIPPAGE_BPS)
                        + max(0.0, OPTIMIZER_EXIT_SLIPPAGE_BPS)
                    )
                    / 2.0,
                )
            )
            strategy_config["entry_slippage_bps"] = float(max(0.0, OPTIMIZER_ENTRY_SLIPPAGE_BPS))
            strategy_config["exit_slippage_bps"] = float(max(0.0, OPTIMIZER_EXIT_SLIPPAGE_BPS))
            genome_adj["transaction_cost_bps"] = strategy_config["transaction_cost_bps"]
            genome_adj["slippage_bps"] = strategy_config["slippage_bps"]
            genome_adj["entry_slippage_bps"] = strategy_config["entry_slippage_bps"]
            genome_adj["exit_slippage_bps"] = strategy_config["exit_slippage_bps"]
        # Default to adaptive exposure in no-leverage mode to avoid sparse,
        # regime-skipping overfit profiles that sit in cash for long spans.
        default_exposure_mode = "exposure" if OBJECTIVE_PROFILE == "no_leverage" else "hybrid"
        exposure_mode = str(
            os.getenv("APEX_MARKET_EXPOSURE_MODE", strategy_config.get("market_exposure_mode", default_exposure_mode))
            or default_exposure_mode
        ).strip().lower()
        if exposure_mode not in {"filter", "hard", "hybrid", "scaled", "exposure"}:
            exposure_mode = "hybrid"
        strategy_config["market_exposure_mode"] = exposure_mode
        genome_adj["market_exposure_mode"] = exposure_mode
        default_bear_max_positions = "2" if OBJECTIVE_PROFILE == "no_leverage" else "1"
        strategy_config["bear_max_positions"] = int(
            os.getenv("APEX_BEAR_MAX_POSITIONS", default_bear_max_positions) or default_bear_max_positions
        )
        strategy_config["regime_filter"] = exposure_mode in {"filter", "hard"}
        strategy_config["regime_exit"] = exposure_mode in {"filter", "hard"}
        strategy_config["market_filter_mode"] = "sma200"
        strategy_config["regime_ma"] = "sma200"
        default_tl = "0"
        use_traffic_light = str(os.getenv("APEX_USE_TRAFFIC_LIGHT", default_tl) or default_tl).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        strategy_config["use_market_regime_traffic_light"] = use_traffic_light
        genome_adj["use_market_regime_traffic_light"] = use_traffic_light
        if OBJECTIVE_PROFILE == "no_leverage":
            bear_cash_mode = str(os.getenv("APEX_BEAR_CASH_MODE", "off") or "off").strip().lower()
            tl_block_exposure = str(os.getenv("APEX_TL_BLOCK_EXPOSURE", "0") or "0").strip().lower() in {
                "1",
                "true",
                "yes",
            }
            strategy_config["bear_cash_mode"] = bear_cash_mode
            strategy_config["traffic_light_block_exposure_mode"] = tl_block_exposure
            strategy_config["enforce_next_day_exit_execution"] = True
            genome_adj["bear_cash_mode"] = bear_cash_mode
            genome_adj["traffic_light_block_exposure_mode"] = tl_block_exposure
            genome_adj["enforce_next_day_exit_execution"] = True
        if "stop_loss_atr_bull" in strategy_config:
            strategy_config["stop_loss_atr"] = strategy_config.get("stop_loss_atr_bull")
        strategy_config["split_exit"] = False
        strategy_config["enable_partial_profit"] = False
        strategy_config["partial_profit_mode"] = "off"
        strategy_config["partial_profit_r"] = 0.0
        strategy_config["partial_profit_pct"] = 0.0
        strategy_config["partial_profit_fraction"] = 0.0
        strategy_config["profit_target"] = 0.0
        strategy_config["profit_target_pct"] = 0.0
        strategy_config["allow_profit_target_exit"] = False
        strategy_config["dead_money_days"] = int(strategy_config.get("time_stop_days", 5) or 5)
        strategy_config["dead_money_profit_pct"] = 0.01
        strategy_config["breakeven_profit_pct"] = 0.20
        strategy_config["sma50_trail_profit_pct"] = 0.40
        strategy_config["sma10_trail_profit_pct"] = 1.00
        strategy_config["exit_sma_fast"] = "sma50"
        strategy_config["exit_sma_slow"] = "sma50"
        strategy_config["move_stop_to_be"] = True
        strategy_config["pyramid_stop_to_avg_cost"] = False
        strategy_config["allow_margin"] = False
        score_mode = str(strategy_config.get("score_mode", "dual_core") or "dual_core").strip().lower()
        if score_mode not in {"dual_core", "momentum"}:
            score_mode = "dual_core"
        strategy_config["score_mode"] = score_mode
        genome_adj["score_mode"] = score_mode

        tech_w = float(strategy_config.get("technical_weight", 0.60) or 0.60)
        fund_w = float(strategy_config.get("fundamental_weight", 0.40) or 0.40)
        if not np.isfinite(tech_w):
            tech_w = 0.60
        if not np.isfinite(fund_w):
            fund_w = 0.40
        tech_w = max(0.0, tech_w)
        fund_w = max(0.0, fund_w)
        total_w = tech_w + fund_w
        if total_w <= 0.0:
            tech_w, fund_w = 0.60, 0.40
            total_w = 1.0
        tech_w /= total_w
        fund_w /= total_w
        strategy_config["technical_weight"] = tech_w
        strategy_config["fundamental_weight"] = fund_w
        genome_adj["technical_weight"] = tech_w
        genome_adj["fundamental_weight"] = fund_w

        min_entry_score = float(strategy_config.get("min_entry_score", 80.0) or 80.0)
        if not np.isfinite(min_entry_score):
            min_entry_score = 80.0
        strategy_config["min_entry_score"] = min_entry_score
        genome_adj["min_entry_score"] = min_entry_score
        strategy_config["batch_size"] = 100
        strategy_config["log_regime_skips"] = False
        # Entry-type sizing caps
        max_pos_cap = float(strategy_config.get("max_pos_size_pct", 0.20) or 0.20)
        if not np.isfinite(max_pos_cap):
            max_pos_cap = 0.20
        max_pos_cap = max(0.05, max_pos_cap)
        strategy_config["vcp_max_pos_size_pct"] = min(max_pos_cap, 0.75)
        strategy_config["ep_max_pos_size_pct"] = min(max_pos_cap, 0.50)
        
        # Instantiate Specific Strategy
        strat = SuperperformanceStrategy(strategy_config)
        
        # Run Backtest
        result = run_backtest(
            [strat], 
            data, 
            start_cash=100000.0, 
            start_date=START_DATE, 
            end_date=END_DATE,
            global_data=global_data,
            universe_membership_by_day=universe_membership_by_day,
            require_pit_membership=REQUIRE_PIT_DAY_MEMBERSHIP,
        )
        
        if not result: return {"id": genome_id, "score": 0, "error": "No result"}
        metrics = result[0] if isinstance(result, list) else result
        
        final_val = metrics.get("final_value", 100000)
        trades = metrics.get("total_trades", 0)
        active_period_years = float(metrics.get("active_period_years", 0.0) or 0.0)
        raw_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0)))
        max_dd = raw_dd * 100.0 if raw_dd < 1.0 else raw_dd
            
        # CAGR
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        years = max(years, 1.0)
        cagr_pct = ((final_val / 100000.0) ** (1/years) - 1) * 100
        recent_5y_cagr = _recent_cagr_pct(metrics.get("equity_curve", []), 5)
        recent_3y_cagr = _recent_cagr_pct(metrics.get("equity_curve", []), 3)
        worst_12m_return_pct = _worst_rolling_12m_return_pct(metrics.get("equity_curve", []))
        worst_24m_cagr, median_24m_cagr, rolling_24m_samples = _rolling_window_cagr_stats(
            metrics.get("equity_curve", []),
            window_years=2,
            step_days=21,
        )
        stitched_oos_cagr, stitched_oos_folds, stitched_oos_worst_fold = _stitched_oos_cagr_pct(
            metrics.get("equity_curve", []),
            train_years=STITCHED_OOS_TRAIN_YEARS,
            test_years=STITCHED_OOS_TEST_YEARS,
        )
        year_conc = _year_concentration_stats(metrics.get("equity_curve", []))
        max_year_share = float(year_conc.get("max_positive_share", float("nan")))
        top2_year_share = float(year_conc.get("top2_positive_share", float("nan")))
        negative_years = int(year_conc.get("negative_years", 0) or 0)
        total_years = int(year_conc.get("total_years", 0) or 0)

        # DEATH PENALTY: Disqualify high drawdown genomes
        if max_dd > float(max_dd_cap):
            return {
                "id": genome_id,
                "genome": genome,
                "score": -100.0,
                "cagr": cagr_pct,
                "cagr_5y": recent_5y_cagr,
                "cagr_3y": recent_3y_cagr,
                "stitched_oos_cagr": stitched_oos_cagr,
                "dd": max_dd,
                "trades": trades,
                "disqualified": True
            }
        
        trades_list = metrics.get("trades_list", []) or []
        audit_report = metrics.get("audit_report", {}) or {}
        try:
            same_day_open_entries = int(audit_report.get("same_day_open_entries", 0) or 0)
        except Exception:
            same_day_open_entries = 0
        try:
            max_gross_exposure_pct = float(audit_report.get("max_gross_exposure_pct", 0.0) or 0.0)
        except Exception:
            max_gross_exposure_pct = 0.0
        # Trade quality (PF) for fitness shaping.
        gross_wins = 0.0
        gross_losses = 0.0
        for tr in trades_list:
            try:
                pnl = float(tr.get("PnL", 0.0) or 0.0)
            except Exception:
                pnl = 0.0
            if pnl > 0:
                gross_wins += pnl
            elif pnl < 0:
                gross_losses += -pnl
        if gross_losses > 0:
            pf = gross_wins / gross_losses
        else:
            pf = 3.0 if gross_wins > 0 else 0.0
        avg_win_pct, avg_loss_pct, win_loss_ratio = _trade_asymmetry(trades_list)

        # FITNESS FUNCTION: CAGR-first with trade-count-adjusted quality.
        # Prevent high-PF / high-ratio small-sample genomes from dominating.
        dd_floor = max(max_dd, 1.0)
        calmar = cagr_pct / dd_floor if dd_floor > 0 else cagr_pct
        pf_clipped = max(0.0, min(float(pf), 5.0))
        ratio_clipped = max(0.0, min(float(win_loss_ratio), 8.0))

        cagr_curve = cagr_pct
        if cagr_pct > 12.0:
            cagr_curve += (cagr_pct - 12.0) * 0.90
        if cagr_pct > 20.0:
            cagr_curve += (cagr_pct - 20.0) * 1.50
        if cagr_pct > 30.0:
            cagr_curve += (cagr_pct - 30.0) * 0.75

        trades_floor = max(float(MIN_TRADES_FLOOR), 1.0)
        trade_factor = max(0.0, min(float(trades) / trades_floor, 1.0))
        trades_per_year = float(trades) / years if years > 0 else float(trades)
        quality_bonus = ((pf_clipped - 1.0) * 6.0) + ((ratio_clipped - 1.5) * 4.0)
        quality_bonus = max(-20.0, quality_bonus) * trade_factor

        score = cagr_curve * 6.0
        score += calmar * CALMAR_WEIGHT
        score += quality_bonus

        # Hard-bias away from low-return / poor-quality / low-sample profiles.
        if cagr_pct < MIN_CAGR_FLOOR:
            score -= (MIN_CAGR_FLOOR - cagr_pct) * 3.5
        if pf_clipped < MIN_PF_FLOOR:
            score -= (MIN_PF_FLOOR - pf_clipped) * 10.0
        if ratio_clipped < MIN_WINLOSS_RATIO:
            score -= (MIN_WINLOSS_RATIO - ratio_clipped) * 5.0

        if trades < MIN_TRADES_FLOOR:
            trade_deficit = (MIN_TRADES_FLOOR - float(trades)) / trades_floor
            score -= trade_deficit * 20.0
            if trades < int(MIN_TRADES_FLOOR * 0.60):
                score -= 6.0
        if trades < MIN_TRADES_HARD_FLOOR:
            score -= 35.0
        if active_period_years < MIN_ACTIVE_YEARS:
            score -= (MIN_ACTIVE_YEARS - active_period_years) * 8.0

        if calmar < CALMAR_FLOOR:
            score -= (CALMAR_FLOOR - calmar) * CALMAR_FLOOR_PENALTY_MULT
        if calmar > CALMAR_TARGET:
            score += (calmar - CALMAR_TARGET) * CALMAR_TARGET_BONUS

        dd_soft = max(0.0, max_dd - DD_PENALTY_START_SOFT)
        dd_hard = max(0.0, max_dd - DD_PENALTY_START_HARD)
        dd_cliff = max(0.0, max_dd - DD_PENALTY_START_CLIFF)
        score -= dd_soft * DD_PENALTY_MULT_SOFT
        score -= dd_hard * DD_PENALTY_MULT_HARD
        score -= dd_cliff * DD_PENALTY_MULT_CLIFF
        if trades_per_year < MIN_TRADES_PER_YEAR_FLOOR:
            score -= (MIN_TRADES_PER_YEAR_FLOOR - trades_per_year) * TRADES_PER_YEAR_LOW_PENALTY_MULT
        elif TARGET_TRADES_PER_YEAR_MIN <= trades_per_year <= TARGET_TRADES_PER_YEAR_MAX:
            score += TRADES_PER_YEAR_TARGET_BONUS
        elif trades_per_year > MAX_TRADES_PER_YEAR_SOFT:
            score -= (trades_per_year - MAX_TRADES_PER_YEAR_SOFT) * TRADES_PER_YEAR_HIGH_PENALTY_MULT
        if MAX_TRADES_SOFT > 0 and trades > MAX_TRADES_SOFT:
            score -= (trades - MAX_TRADES_SOFT) / 35.0
        if cagr_pct <= 0.0:
            score -= 25.0

        # Keep optimization aligned with practical execution constraints.
        if same_day_open_entries > 0:
            score -= min(160.0, 25.0 + (same_day_open_entries * 0.35))
        if max_gross_exposure_pct > 1.0:
            score -= (max_gross_exposure_pct - 1.0) * 500.0

        # Keep some recency awareness, but avoid hard-wiring optimization to a single market phase.
        if np.isfinite(recent_5y_cagr):
            score += recent_5y_cagr * RECENT_5Y_WEIGHT
            if recent_5y_cagr < RECENT_5Y_CAGR_FLOOR:
                score -= (RECENT_5Y_CAGR_FLOOR - recent_5y_cagr) * RECENT_5Y_PENALTY_MULT
            elif recent_5y_cagr > RECENT_5Y_CAGR_TARGET:
                score += (recent_5y_cagr - RECENT_5Y_CAGR_TARGET) * RECENT_5Y_TARGET_BONUS
        else:
            score -= 3.0

        if np.isfinite(recent_3y_cagr):
            score += recent_3y_cagr * RECENT_3Y_WEIGHT
            if recent_3y_cagr < RECENT_3Y_CAGR_FLOOR:
                score -= (RECENT_3Y_CAGR_FLOOR - recent_3y_cagr) * RECENT_3Y_PENALTY_MULT
            elif recent_3y_cagr > RECENT_3Y_CAGR_TARGET:
                score += (recent_3y_cagr - RECENT_3Y_CAGR_TARGET) * RECENT_3Y_TARGET_BONUS
        else:
            score -= 2.0

        if np.isfinite(worst_12m_return_pct) and worst_12m_return_pct < WORST_12M_FLOOR_PCT:
            score -= (WORST_12M_FLOOR_PCT - worst_12m_return_pct) * WORST_12M_PENALTY_MULT

        if np.isfinite(stitched_oos_cagr):
            score += stitched_oos_cagr * STITCHED_OOS_WEIGHT
            if stitched_oos_cagr < STITCHED_OOS_CAGR_FLOOR:
                score -= (STITCHED_OOS_CAGR_FLOOR - stitched_oos_cagr) * STITCHED_OOS_PENALTY_MULT
            elif stitched_oos_cagr > STITCHED_OOS_TARGET:
                score += (stitched_oos_cagr - STITCHED_OOS_TARGET) * STITCHED_OOS_TARGET_BONUS
        else:
            score -= 6.0

        if np.isfinite(worst_24m_cagr):
            if worst_24m_cagr < WORST_24M_CAGR_FLOOR:
                score -= (WORST_24M_CAGR_FLOOR - worst_24m_cagr) * WORST_24M_CAGR_PENALTY_MULT
        else:
            score -= 3.0

        if np.isfinite(median_24m_cagr):
            score += median_24m_cagr * 0.35
            if median_24m_cagr < MEDIAN_24M_CAGR_FLOOR:
                score -= (MEDIAN_24M_CAGR_FLOOR - median_24m_cagr) * MEDIAN_24M_CAGR_PENALTY_MULT
        else:
            score -= 3.0

        if np.isfinite(max_year_share) and max_year_share > MAX_SINGLE_YEAR_PNL_SHARE:
            score -= (max_year_share - MAX_SINGLE_YEAR_PNL_SHARE) * YEAR_CONCENTRATION_PENALTY_MULT
        if np.isfinite(top2_year_share) and top2_year_share > MAX_TWO_YEAR_PNL_SHARE:
            score -= (top2_year_share - MAX_TWO_YEAR_PNL_SHARE) * TWO_YEAR_CONCENTRATION_PENALTY_MULT
        if total_years >= 5 and negative_years > NEGATIVE_YEAR_SOFT_CAP:
            score -= (negative_years - NEGATIVE_YEAR_SOFT_CAP) * NEGATIVE_YEAR_PENALTY

        # Soft anti-overrestriction bias:
        # Keep broad-profile runs from collapsing into ultra-sparse screeners.
        # Skip this in no-leverage mode because strict quality filters are intentional.
        if OBJECTIVE_PROFILE != "no_leverage":
            if float(strategy_config.get("rs_gate_min", 85.0) or 85.0) > 90.0:
                score -= 5.0
            if float(strategy_config.get("fundamental_growth_min_pct", 20.0) or 20.0) > 20.0:
                score -= 4.0
            if float(strategy_config.get("ep_gap_pct", 8.0) or 8.0) > 8.0:
                score -= 4.0
            if float(strategy_config.get("min_avg_volume_30", 0.0) or 0.0) >= 500000.0:
                score -= 3.0
            min_entry_score_cfg = float(strategy_config.get("min_entry_score", 0.0) or 0.0)
            if min_entry_score_cfg > 55.0:
                score -= (min_entry_score_cfg - 55.0) * 1.5

        is_super_candidate = (
            cagr_pct >= TARGET_CAGR
            and pf_clipped >= max(1.20, MIN_PF_FLOOR)
            and ratio_clipped >= max(2.5, MIN_WINLOSS_RATIO)
            and trades >= MIN_TRADES_FLOOR
            and calmar >= CALMAR_FLOOR
            and max_dd <= DD_PENALTY_START_CLIFF
            and (not np.isfinite(stitched_oos_cagr) or stitched_oos_cagr >= STITCHED_OOS_CAGR_FLOOR)
        )
        if is_super_candidate:
            score += 10.0

        return {
            "id": genome_id,
            "genome": genome_adj,
            "score": score,
            "cagr": cagr_pct,
            "cagr_5y": recent_5y_cagr,
            "cagr_3y": recent_3y_cagr,
            "dd": max_dd,
            "calmar": calmar,
            "pf": pf,
            "trades": trades,
            "active_period_years": active_period_years,
            "worst_12m_return_pct": worst_12m_return_pct,
            "worst_24m_cagr": worst_24m_cagr,
            "median_24m_cagr": median_24m_cagr,
            "rolling_24m_samples": rolling_24m_samples,
            "stitched_oos_cagr": stitched_oos_cagr,
            "stitched_oos_folds": stitched_oos_folds,
            "stitched_oos_worst_fold": stitched_oos_worst_fold,
            "max_year_pnl_share": max_year_share,
            "top2_year_pnl_share": top2_year_share,
            "negative_years": negative_years,
            "total_years": total_years,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "win_loss_ratio": win_loss_ratio,
            "same_day_open_entries": same_day_open_entries,
            "max_gross_exposure_pct": max_gross_exposure_pct,
        }
    except Exception as e:
        return {"id": genome_id, "score": -999, "error": str(e)}
    finally:
        gc.collect()
if __name__ == "__main__":
    start_method_in_use = MP_START_METHOD
    try:
        multiprocessing.set_start_method(MP_START_METHOD, force=True)
    except Exception as exc:
        start_method_in_use = multiprocessing.get_start_method(allow_none=True) or MP_START_METHOD
        print(f"⚠️  Unable to force start method '{MP_START_METHOD}': {exc}. Using '{start_method_in_use}'.")

    print(f"🚀 PROJECT APEX: Strategic Nuclear Reset")
    print(
        f"HARDWARE: M3 Max | REQUESTED_WORKERS: {MAX_WORKERS} | "
        f"MP_START_METHOD: {start_method_in_use}"
    )
    print(
        "FRICTION MODEL: "
        f"transaction_cost_bps={OPTIMIZER_COST_BPS:.1f}, "
        f"entry_slippage_bps={OPTIMIZER_ENTRY_SLIPPAGE_BPS:.1f}, "
        f"exit_slippage_bps={OPTIMIZER_EXIT_SLIPPAGE_BPS:.1f}"
    )
    print(f"OBJECTIVE PROFILE: {OBJECTIVE_PROFILE} (cash-only, no leverage hard-clamped)")
    if "APEX_TRACE_KNOWN_WINNER_REJECTS" not in os.environ:
        os.environ["APEX_TRACE_KNOWN_WINNER_REJECTS"] = "1" if OPTIMIZER_TRACE_REJECTS else "0"
    if "APEX_TRACE_REJECTS_MAX_LINES" not in os.environ:
        os.environ["APEX_TRACE_REJECTS_MAX_LINES"] = str(max(0, int(OPTIMIZER_TRACE_MAX_LINES)))
    print(
        "TRACE LOGGER: "
        f"known_winner_rejects={'on' if os.environ.get('APEX_TRACE_KNOWN_WINNER_REJECTS') in {'1', 'true', 'yes'} else 'off'} "
        f"(max_lines={os.environ.get('APEX_TRACE_REJECTS_MAX_LINES', '0')})"
    )
    print(f"TIMEOUT GUARD: per-generation straggler cutoff={GENOME_TIMEOUT_MIN:.1f}m")
    print("SCHEDULER: Persistent pool + per-genome dynamic dispatch")
    
    print("...Loading Data...")
    universe_name = str(os.getenv("APEX_UNIVERSE", "RUSSELL3000") or "RUSSELL3000").strip().upper()
    if universe_name == "RUSSELL3000":
        symbols, universe_source = get_universe_symbols_pit_window_with_meta(
            universe_name,
            START_DATE,
            END_DATE,
        )
        print(f"...Universe source: {universe_source}")
        if REQUIRE_PIT_UNIVERSE:
            if universe_source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
                raise RuntimeError(
                    "PIT Russell 3000 universe is required, but PIT data is unavailable for the requested window. "
                    "Populate data/russell3000_membership or RUSSELL3000_PIT_MEMBERSHIP_CSV."
                )
            if not symbols:
                raise RuntimeError(
                    "PIT Russell 3000 universe is required, but zero symbols were returned for the requested window."
                )
    else:
        symbols = get_universe_symbols(universe_name) or []
    if not symbols and universe_name != "RUSSELL3000":
        print(f"⚠️ Universe '{universe_name}' returned no symbols. Falling back to RUSSELL3000.")
        universe_name = "RUSSELL3000"
        symbols, universe_source = get_universe_symbols_pit_window_with_meta(
            universe_name,
            START_DATE,
            END_DATE,
        )
        print(f"...Universe source: {universe_source}")
        if REQUIRE_PIT_UNIVERSE:
            if universe_source not in {"pit_snapshot", "pit_ranges", "pit_snapshot_window"}:
                raise RuntimeError(
                    "PIT Russell 3000 universe is required, but PIT data is unavailable for the requested window. "
                    "Populate data/russell3000_membership or RUSSELL3000_PIT_MEMBERSHIP_CSV."
                )
            if not symbols:
                raise RuntimeError(
                    "PIT Russell 3000 universe is required, but zero symbols were returned for the requested window."
                )
    universe_limit = int(os.getenv("APEX_UNIVERSE_LIMIT", "0") or "0")
    if universe_limit > 0:
        symbols = symbols[:universe_limit]
    print(f"...Universe: {universe_name} ({len(symbols)} symbols)")
    _ensure_fundamental_coverage(symbols)
    total_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
    print(
        f"...Optimization window: {START_DATE} -> {END_DATE} "
        f"({max(total_days, 0) / 365.25:.1f} years)"
    )
    if total_days < 3650:
        print("⚠️  Short optimization window detected (<10 years). Set APEX_START_DATE/APEX_END_DATE for full-cycle tuning.")
    trading_days = int((total_days / 365.25) * 252) + 400  # warmup buffer

    prepared = None
    cache_path = os.path.join(ROOT, "data", "cache_indicators.pkl")
    disable_cache = str(os.getenv("APEX_DISABLE_INDICATOR_CACHE", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    min_cache_symbols = int(os.getenv("APEX_MIN_CACHE_SYMBOLS", "100") or "100")
    if not disable_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if hasattr(cached, "enriched") and hasattr(cached, "all_dates"):
                cached_symbols = list(getattr(cached, "enriched", {}).keys())
                # Respect selected universe when loading cache; do not auto-promote to full cache.
                cached_lookup = {str(sym).upper(): sym for sym in cached_symbols}
                requested_upper = [str(sym).upper() for sym in symbols]
                if requested_upper:
                    selected_cache_symbols = [
                        cached_lookup[su] for su in requested_upper if su in cached_lookup
                    ]
                else:
                    selected_cache_symbols = cached_symbols

                target_min = min_cache_symbols if not requested_upper else max(
                    min_cache_symbols, int(len(requested_upper) * 0.60)
                )
                if len(selected_cache_symbols) >= target_min:
                    filtered_enriched = {
                        sym: cached.enriched[sym]
                        for sym in selected_cache_symbols
                        if sym in cached.enriched
                    }
                    prepared = type(cached)(
                        enriched=filtered_enriched,
                        all_dates=getattr(cached, "all_dates"),
                    )
                    symbols = list(filtered_enriched.keys())
                    print(
                        "...Loaded cached indicators: "
                        f"{len(symbols)} symbols (filtered from cache universe)"
                    )
                else:
                    print(
                        f"...Ignoring cached indicators: matched {len(selected_cache_symbols)} symbols "
                        f"(min required {target_min})"
                    )
            cached = None
            gc.collect()
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}")
    elif disable_cache:
        print("...Indicator cache disabled via APEX_DISABLE_INDICATOR_CACHE")

    data = None
    if prepared is None:
        data = fetch_data_pack(symbols, days=trading_days, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=trading_days, backtest_mode=True)
    
    print("...Preparing & Compressing...")
    # Strict Prep Config
    PREP_CONFIG = {
        "parameters": {
            "min_rs": 60,           
            "adx_threshold": 10,  
            "vol_ma_ratio": 0.75,
            "bb_width_threshold": 0.40 
        }
    }
    if prepared is None:
        prepared = prepare_backtest_data(data, symbols, start_date=START_DATE, global_data=g_data)
        print(f"📊 DATA POOL: {len(prepared.enriched)} tickers prepared.")
        write_cache = str(os.getenv("APEX_WRITE_INDICATOR_CACHE", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if write_cache and universe_limit <= 0:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"💾 Saved indicator cache: {cache_path}")
            except Exception as exc:
                print(f"⚠️ Failed to save indicator cache: {exc}")
        elif write_cache:
            print("💾 Skipping indicator cache write for limited-universe run.")
    else:
        print(f"📊 DATA POOL: {len(prepared.enriched)} tickers prepared (cache).")
    if PRUNE_PREPARED_DF:
        prepared = _prune_prepared_frames(prepared)
    prepared_mem_before = _estimate_prepared_bytes(prepared)
    prepared = compress_data(prepared)
    prepared_mem_after = _estimate_prepared_bytes(prepared)
    print(
        f"🧠 Prepared memory estimate: {_bytes_to_gb(prepared_mem_before):.2f} GB -> "
        f"{_bytes_to_gb(prepared_mem_after):.2f} GB"
    )
    universe_membership_by_day = None
    if universe_name == "RUSSELL3000":
        try:
            all_dates_raw = getattr(prepared, "all_dates", None)
            all_dates_seq = list(all_dates_raw) if all_dates_raw is not None else []
            membership, membership_source = build_russell3000_membership_by_day(
                all_dates_seq,
                allow_missing_days=not REQUIRE_PIT_DAY_MEMBERSHIP,
            )
            if membership and len(membership) == len(all_dates_seq):
                universe_membership_by_day = membership
                print(f"📌 PIT day-membership loaded for optimizer: source={membership_source}")
            else:
                msg = (
                    "PIT day-membership is unavailable, incomplete, or misaligned with the prepared calendar. "
                    "Failing closed."
                )
                if REQUIRE_PIT_DAY_MEMBERSHIP:
                    raise RuntimeError(msg)
                print(f"⚠️ {msg} Optimizer will run without per-day membership filter.")
        except Exception as exc:
            if REQUIRE_PIT_DAY_MEMBERSHIP:
                raise
            print(f"⚠️ Failed to build PIT day-membership: {exc}")
    global_mem = _estimate_data_bytes(g_data)
    effective_workers, worker_plan = _resolve_effective_workers(MAX_WORKERS, prepared_mem_after, global_mem)
    print(
        "🧠 Worker memory plan: "
        f"dataset={worker_plan['dataset_gb']:.2f} GB | per_worker={worker_plan['per_worker_gb']:.2f} GB | "
        f"RAM budget={worker_plan['ram_budget_gb']:.2f} GB | requested={worker_plan['requested']} | "
        f"effective={worker_plan['effective']}"
    )
    if effective_workers < MAX_WORKERS:
        print(
            f"⚠️  Auto-limiting workers to {effective_workers} for memory safety "
            f"(requested {MAX_WORKERS})."
        )
    if isinstance(data, dict):
        data.clear()
    data = None
    gc.collect()

    checkpoint = load_checkpoint() if RESUME_CHECKPOINT else None
    start_gen = 0
    if checkpoint:
        ckpt_gen, ckpt_population, _ = checkpoint
        if isinstance(ckpt_population, list) and len(ckpt_population) == POPULATION_SIZE:
            population = ckpt_population
            start_gen = int(ckpt_gen) + 1
            print(f"🔁 Resuming from checkpoint at generation {start_gen + 1}/{GENERATIONS}")
        else:
            print("⚠️ Checkpoint population mismatch. Starting fresh population.")
            seeds = _seed_population(POPULATION_SIZE)
            population = list(seeds)
            while len(population) < POPULATION_SIZE:
                population.append(generate_random_genome())
    else:
        seeds = _seed_population(POPULATION_SIZE)
        if seeds:
            print(f"🌱 Seeded initial population with {len(seeds)} known practical genomes.")
        population = list(seeds)
        while len(population) < POPULATION_SIZE:
            population.append(generate_random_genome())
    dd_cap = MAX_DD_CAP
    executor_kwargs = {
        "max_workers": effective_workers,
        "initializer": _init_worker,
        "initargs": (prepared, g_data, universe_membership_by_day),
    }
    if POOL_MAX_TASKS_PER_CHILD > 0:
        executor_kwargs["max_tasks_per_child"] = POOL_MAX_TASKS_PER_CHILD
    try:
        executor_ctx = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)
    except TypeError:
        executor_kwargs.pop("max_tasks_per_child", None)
        executor_ctx = concurrent.futures.ProcessPoolExecutor(**executor_kwargs)
    with executor_ctx as executor:
        for gen in range(start_gen, GENERATIONS):
            print(f"\n🧬 GEN {gen+1}/{GENERATIONS} (Superperformance)")
            start_time = time.time()

            tasks, dupes = _build_tasks_from_population(population, dd_cap)
            if dupes > 0:
                print(f"   ♻️ Deduped {dupes} duplicate genomes before evaluation.")
            results = []

            while True:
                results = []
                batch_start = time.time()
                futures = [executor.submit(evaluate_genome, task) for task in tasks]
                pending = set(futures)
                next_heartbeat = time.time() + 60.0
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=15.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        if GENOME_TIMEOUT_MIN > 0:
                            elapsed_batch_min = (time.time() - batch_start) / 60.0
                            if elapsed_batch_min >= GENOME_TIMEOUT_MIN:
                                skipped = len(pending)
                                for fut in list(pending):
                                    fut.cancel()
                                pending = set()
                                for _ in range(skipped):
                                    results.append(
                                        {
                                            "id": -1,
                                            "score": -999,
                                            "error": (
                                                f"timeout>{GENOME_TIMEOUT_MIN:.1f}m "
                                                "skipped by timeout guard"
                                            ),
                                        }
                                    )
                                print(
                                    f"⚠️  Timeout guard triggered at {elapsed_batch_min:.1f}m; "
                                    f"skipped {skipped} straggler genomes."
                                )
                                break
                        if time.time() >= next_heartbeat:
                            print(
                                f"   ...running {len(results)}/{len(tasks)} genomes complete "
                                f"| elapsed={(time.time() - start_time)/60.0:.1f}m"
                            )
                            next_heartbeat = time.time() + 60.0
                        continue
                    for future in done:
                        res = future.result()
                        results.append(res)
                        if "error" not in res:
                            if res.get("disqualified"):
                                print(
                                    f"   > T:{res.get('trades', 0)} | CAGR:{res.get('cagr', 0):.1f}% "
                                    f"| OOS:{res.get('stitched_oos_cagr', float('nan')):.1f}% "
                                    f"| 5Y:{res.get('cagr_5y', float('nan')):.1f}% | DD:{res.get('dd', 0):.1f}% "
                                    f"| Fitness:-100 (DD Cap)"
                                )
                            else:
                                print(
                                    f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | DD:{res['dd']:.1f}% "
                                    f"| PF:{res.get('pf', 0.0):.2f} | Calmar:{res.get('calmar', 0):.2f} "
                                    f"| OOS:{res.get('stitched_oos_cagr', float('nan')):.1f}% "
                                    f"| 5Y:{res.get('cagr_5y', float('nan')):.1f}% | 3Y:{res.get('cagr_3y', float('nan')):.1f}% "
                                    f"| YShare:{float(res.get('max_year_pnl_share', float('nan'))):.2f} "
                                    f"| SDO:{int(res.get('same_day_open_entries', 0) or 0)} "
                                    f"| Gross:{float(res.get('max_gross_exposure_pct', 0.0) or 0.0) * 100.0:.1f}%"
                                )
                        else:
                            print(f"   ⚠️  GENOME {res['id']} FAILED: {res['error']}")

                valid = [r for r in results if "error" not in r and not r.get("disqualified")]
                if valid:
                    break
                if dd_cap < 35.0:
                    prev_cap = float(dd_cap)
                    dd_cap = 30.0 if dd_cap < 30.0 else 35.0
                    tasks, dupes = _build_tasks_from_population(population, dd_cap)
                    if dupes > 0:
                        print(f"   ♻️ Deduped {dupes} duplicate genomes before retry.")
                    print(
                        f"⚠️  All genomes disqualified at {prev_cap:.1f}% DD cap. "
                        f"Loosening to {dd_cap:.1f}% and retrying this generation."
                    )
                    continue
                print("CRITICAL: All failed or disqualified.")
                break

            if not valid:
                break

            valid.sort(key=lambda x: x["score"], reverse=True)
            winner = valid[0]

            print(
                f"🏆 WINNER: CAGR {winner['cagr']:.2f}% | DD {winner['dd']:.2f}% | "
                f"PF {winner.get('pf', 0.0):.2f} | Calmar {winner.get('calmar', 0):.2f} "
                f"| OOS {winner.get('stitched_oos_cagr', float('nan')):.2f}% "
                f"| 5Y {winner.get('cagr_5y', float('nan')):.2f}% | 3Y {winner.get('cagr_3y', float('nan')):.2f}% "
                f"| YearShare {float(winner.get('max_year_pnl_share', float('nan'))):.2f} "
                f"| Score: {winner['score']:.2f}"
            )
            print(f"🧬 DNA: {winner['genome']}")

            with open(BEST_GENOME_FILE, "w") as f:
                json.dump(winner["genome"], f, indent=4)
            last_metrics_payload = {
                "avg_win_pct": float(winner.get("avg_win_pct", 0.0) or 0.0),
                "avg_loss_pct": float(winner.get("avg_loss_pct", 0.0) or 0.0),
                "win_loss_ratio": float(winner.get("win_loss_ratio", 0.0) or 0.0),
                "pf": float(winner.get("pf", 0.0) or 0.0),
                "cagr": float(winner.get("cagr", 0.0) or 0.0),
                "cagr_5y": float(winner.get("cagr_5y", float("nan"))),
                "cagr_3y": float(winner.get("cagr_3y", float("nan"))),
                "stitched_oos_cagr": float(winner.get("stitched_oos_cagr", float("nan"))),
                "stitched_oos_folds": int(winner.get("stitched_oos_folds", 0) or 0),
                "stitched_oos_worst_fold": float(winner.get("stitched_oos_worst_fold", float("nan"))),
                "max_year_pnl_share": float(winner.get("max_year_pnl_share", float("nan"))),
                "top2_year_pnl_share": float(winner.get("top2_year_pnl_share", float("nan"))),
                "dd": float(winner.get("dd", 0.0) or 0.0),
                "trades": int(winner.get("trades", 0) or 0),
                "updated_at": pd.Timestamp.now().isoformat(),
            }
            with open(LAST_METRICS_FILE, "w") as f:
                json.dump(last_metrics_payload, f, indent=4)
            with open(RESULTS_FILE, "a") as f:
                f.write(f"{gen+1},{winner['cagr']},{winner['dd']},{winner.get('calmar', 0)},{winner['trades']},\"{winner['genome']}\"\n")

            # Breeding with diversity injection to avoid low-volatility local optima.
            elite_n = max(1, min(int(ELITE_COUNT), len(valid)))
            next_gen = []
            next_seen = set()
            for r in valid:
                g = r.get("genome")
                if not isinstance(g, dict):
                    continue
                sig = _genome_signature(g)
                if sig in next_seen:
                    continue
                next_seen.add(sig)
                next_gen.append(dict(g))
                if len(next_gen) >= elite_n:
                    break
            if not next_gen and isinstance(winner.get("genome"), dict):
                next_gen.append(dict(winner["genome"]))
                next_seen.add(_genome_signature(winner["genome"]))
            progress = float(gen + 1) / max(float(GENERATIONS), 1.0)
            mutation_rate = max(0.45, 0.80 - (0.35 * progress))
            if winner.get("cagr", 0.0) < MIN_CAGR_FLOOR:
                mutation_rate = max(mutation_rate, 0.85)

            immigrant_count = int(max(0, round(POPULATION_SIZE * max(0.0, min(IMMIGRANT_FRAC, 0.6)))))
            child_target = max(0, POPULATION_SIZE - immigrant_count)
            parent_pool = valid[: max(10, min(len(valid), 30))]
            attempt_limit = max(POPULATION_SIZE * 25, 250)
            attempts = 0
            while len(next_gen) < child_target and attempts < attempt_limit:
                attempts += 1
                p1 = random.choice(parent_pool)["genome"]
                p2 = random.choice(parent_pool)["genome"]
                child = crossover(p1, p2)
                if random.random() < mutation_rate:
                    child = mutate_genome(child)
                sig = _genome_signature(child)
                if sig in next_seen:
                    continue
                next_seen.add(sig)
                next_gen.append(child)

            while len(next_gen) < POPULATION_SIZE and attempts < (attempt_limit * 2):
                attempts += 1
                child = generate_random_genome()
                sig = _genome_signature(child)
                if sig in next_seen:
                    continue
                next_seen.add(sig)
                next_gen.append(child)

            while len(next_gen) < POPULATION_SIZE:
                next_gen.append(generate_random_genome())

            population = next_gen
            elapsed = time.time() - start_time
            print(
                f"⏱️  GEN {gen+1} elapsed: {elapsed/60.0:.1f}m | mutation={mutation_rate:.2f} "
                f"| immigrants={max(0, POPULATION_SIZE - child_target)}"
            )
            save_checkpoint(gen, population, winner)
            gc.collect()
