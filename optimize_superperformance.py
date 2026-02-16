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
    from data.universe import get_universe_symbols
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
PRUNE_PREPARED_DF = str(os.getenv("APEX_PRUNE_PREPARED_DF", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
RESUME_CHECKPOINT = str(os.getenv("APEX_RESUME_CHECKPOINT", "1") or "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
OBJECTIVE_PROFILE = str(os.getenv("APEX_OBJECTIVE_PROFILE", "superperformance") or "superperformance").strip().lower()
if OBJECTIVE_PROFILE not in {"superperformance", "balanced", "defensive"}:
    OBJECTIVE_PROFILE = "superperformance"
_TARGET_DEFAULT = "35.0" if OBJECTIVE_PROFILE == "superperformance" else "25.0"
_MIN_CAGR_DEFAULT = "20.0" if OBJECTIVE_PROFILE == "superperformance" else "10.0"
TARGET_CAGR = float(os.getenv("APEX_TARGET_CAGR", _TARGET_DEFAULT) or _TARGET_DEFAULT)
MIN_CAGR_FLOOR = float(os.getenv("APEX_MIN_CAGR_FLOOR", _MIN_CAGR_DEFAULT) or _MIN_CAGR_DEFAULT)
MIN_PF_FLOOR = float(os.getenv("APEX_MIN_PF_FLOOR", "1.20") or "1.20")
MIN_WINLOSS_RATIO = float(os.getenv("APEX_MIN_WINLOSS_RATIO", "2.50") or "2.50")
MIN_TRADES_FLOOR = int(os.getenv("APEX_MIN_TRADES_FLOOR", "50") or "50")
MAX_TRADES_SOFT = int(os.getenv("APEX_MAX_TRADES_SOFT", "700") or "700")
OPTIMIZER_COST_BPS = float(os.getenv("APEX_OPTIMIZER_COST_BPS", "0") or "0")
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
RECENT_5Y_CAGR_FLOOR = float(os.getenv("APEX_RECENT_5Y_CAGR_FLOOR", "12.0") or "12.0")
RECENT_3Y_CAGR_FLOOR = float(os.getenv("APEX_RECENT_3Y_CAGR_FLOOR", "14.0") or "14.0")
RECENT_5Y_CAGR_TARGET = float(os.getenv("APEX_RECENT_5Y_CAGR_TARGET", "16.0") or "16.0")
RECENT_3Y_CAGR_TARGET = float(os.getenv("APEX_RECENT_3Y_CAGR_TARGET", "18.0") or "18.0")
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
    "time_stop_days": [5],
    "pyramid_threshold": [0.04, 0.06, 0.08, 0.10],
    "pyramid_fraction": [0.5, 0.67],
    "pyramid_max_adds": [1, 2, 3],
    "max_positions": [6, 8, 10, 12],
    "risk_per_trade": [0.015, 0.02, 0.025, 0.03],
    "max_pos_size_pct": [0.15, 0.20, 0.25, 0.30],
    "max_total_exposure_pct_bull": [1.0, 1.2, 1.4, 1.6],
    "max_total_exposure_pct_bear": [0.1, 0.2, 0.3],
}


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

def _init_worker(prepared_data_readonly, global_data_readonly):
    global _worker_data, _worker_global_data
    _worker_data = prepared_data_readonly
    _worker_global_data = global_data_readonly


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


def evaluate_genome(genome_id_and_genome):
    try:
        genome_id, genome, max_dd_cap = genome_id_and_genome
    except Exception:
        genome_id, genome = genome_id_and_genome
        max_dd_cap = MAX_DD_CAP
    global _worker_data, _worker_global_data
    data = _worker_data
    global_data = _worker_global_data
    
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
        # Pairing clamp: enforce cash-only exposure physics
        max_exposure = float(strategy_config.get("max_total_exposure_pct_bull", 1.0) or 1.0)
        max_positions = int(strategy_config.get("max_positions", 1) or 1)
        max_pos_size = float(strategy_config.get("max_pos_size_pct", 1.0) or 1.0)
        if max_positions > 0 and (max_positions * max_pos_size) > max_exposure:
            max_pos_size = max_exposure / max_positions
            strategy_config["max_pos_size_pct"] = max_pos_size
            genome_adj["max_pos_size_pct"] = max_pos_size
        # After-close scan, next-day stop order (EP overrides with same-day close)
        strategy_config["signal_mode"] = "after_close"
        if np.isfinite(OPTIMIZER_COST_BPS) and OPTIMIZER_COST_BPS > 0:
            strategy_config["transaction_cost_bps"] = float(max(0.0, OPTIMIZER_COST_BPS))
        # Default to bull-deploy/bear-cash behavior for superperformance tuning.
        exposure_mode = str(
            os.getenv("APEX_MARKET_EXPOSURE_MODE", strategy_config.get("market_exposure_mode", "hybrid"))
            or "hybrid"
        ).strip().lower()
        if exposure_mode not in {"filter", "hard", "hybrid", "scaled", "exposure"}:
            exposure_mode = "hybrid"
        strategy_config["market_exposure_mode"] = exposure_mode
        genome_adj["market_exposure_mode"] = exposure_mode
        strategy_config["bear_max_positions"] = int(os.getenv("APEX_BEAR_MAX_POSITIONS", "1") or "1")
        strategy_config["regime_filter"] = exposure_mode in {"filter", "hard"}
        strategy_config["regime_exit"] = exposure_mode in {"filter", "hard"}
        strategy_config["market_filter_mode"] = "sma200"
        strategy_config["regime_ma"] = "sma200"
        use_traffic_light = str(os.getenv("APEX_USE_TRAFFIC_LIGHT", "0") or "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        strategy_config["use_market_regime_traffic_light"] = use_traffic_light
        genome_adj["use_market_regime_traffic_light"] = use_traffic_light
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
        strategy_config["pyramid_stop_to_avg_cost"] = True
        strategy_config["allow_margin"] = (
            float(strategy_config.get("max_total_exposure_pct_bull", 1.0) or 1.0) > 1.0
            or float(strategy_config.get("max_pos_size_pct", 1.0) or 1.0) > 1.0
        )
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
            global_data=global_data
        )
        
        if not result: return {"id": genome_id, "score": 0, "error": "No result"}
        metrics = result[0] if isinstance(result, list) else result
        
        final_val = metrics.get("final_value", 100000)
        trades = metrics.get("total_trades", 0)
        raw_dd = metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0)))
        max_dd = raw_dd * 100.0 if raw_dd < 1.0 else raw_dd
            
        # CAGR
        years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
        years = max(years, 1.0)
        cagr_pct = ((final_val / 100000.0) ** (1/years) - 1) * 100
        recent_5y_cagr = _recent_cagr_pct(metrics.get("equity_curve", []), 5)
        recent_3y_cagr = _recent_cagr_pct(metrics.get("equity_curve", []), 3)

        # DEATH PENALTY: Disqualify high drawdown genomes
        if max_dd > float(max_dd_cap):
            return {
                "id": genome_id,
                "genome": genome,
                "score": -100.0,
                "cagr": cagr_pct,
                "cagr_5y": recent_5y_cagr,
                "cagr_3y": recent_3y_cagr,
                "dd": max_dd,
                "trades": trades,
                "disqualified": True
            }
        
        trades_list = metrics.get("trades_list", []) or []
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
            cagr_curve += (cagr_pct - 30.0) * 2.25

        trades_floor = max(float(MIN_TRADES_FLOOR), 1.0)
        trade_factor = max(0.0, min(float(trades) / trades_floor, 1.0))
        quality_bonus = ((pf_clipped - 1.0) * 6.0) + ((ratio_clipped - 1.5) * 4.0)
        quality_bonus = max(-20.0, quality_bonus) * trade_factor

        score = cagr_curve * 6.0
        score += calmar * 3.0
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

        if max_dd > 30.0:
            score -= (max_dd - 30.0) * 1.5
        if trades > MAX_TRADES_SOFT:
            score -= (trades - MAX_TRADES_SOFT) / 35.0
        if cagr_pct <= 0.0:
            score -= 25.0

        # Recency retention bias:
        # keep strong modern-market performance while still optimizing full-cycle CAGR.
        if np.isfinite(recent_5y_cagr):
            score += recent_5y_cagr * 0.80
            if recent_5y_cagr < RECENT_5Y_CAGR_FLOOR:
                score -= (RECENT_5Y_CAGR_FLOOR - recent_5y_cagr) * 8.0
            elif recent_5y_cagr >= RECENT_5Y_CAGR_TARGET:
                score += 10.0
        else:
            score -= 3.0

        if np.isfinite(recent_3y_cagr):
            score += recent_3y_cagr * 0.40
            if recent_3y_cagr < RECENT_3Y_CAGR_FLOOR:
                score -= (RECENT_3Y_CAGR_FLOOR - recent_3y_cagr) * 6.0
            elif recent_3y_cagr >= RECENT_3Y_CAGR_TARGET:
                score += 6.0
        else:
            score -= 2.0

        # Soft anti-overrestriction bias:
        # Prevent optimizer from collapsing into a low-frequency "screener" profile.
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
        )
        if is_super_candidate:
            score += 35.0

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
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "win_loss_ratio": win_loss_ratio,
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
    print(f"FRICTION MODEL: optimizer_cost_bps={OPTIMIZER_COST_BPS:.1f}")
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
    symbols = get_universe_symbols(universe_name) or []
    if not symbols and universe_name != "RUSSELL3000":
        print(f"⚠️ Universe '{universe_name}' returned no symbols. Falling back to RUSSELL3000.")
        universe_name = "RUSSELL3000"
        symbols = get_universe_symbols(universe_name) or []
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
            population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    else:
        population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    dd_cap = MAX_DD_CAP
    executor_kwargs = {
        "max_workers": effective_workers,
        "initializer": _init_worker,
        "initargs": (prepared, g_data),
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
                                    f"| 5Y:{res.get('cagr_5y', float('nan')):.1f}% | DD:{res.get('dd', 0):.1f}% "
                                    f"| Fitness:-100 (DD Cap)"
                                )
                            else:
                                print(
                                    f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | DD:{res['dd']:.1f}% "
                                    f"| PF:{res.get('pf', 0.0):.2f} | Calmar:{res.get('calmar', 0):.2f} "
                                    f"| 5Y:{res.get('cagr_5y', float('nan')):.1f}% | 3Y:{res.get('cagr_3y', float('nan')):.1f}%"
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
                f"| 5Y {winner.get('cagr_5y', float('nan')):.2f}% | 3Y {winner.get('cagr_3y', float('nan')):.2f}% "
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
