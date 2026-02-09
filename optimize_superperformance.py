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
START_DATE = str(os.getenv("APEX_START_DATE", "2020-01-01") or "2020-01-01")
END_DATE = str(os.getenv("APEX_END_DATE", "2025-12-31") or "2025-12-31")
MAX_WORKERS = int(os.getenv("APEX_MAX_WORKERS", "10") or "10")
CHECKPOINT_FILE = "optimizer_checkpoint_sp.pkl" 
MAX_DD_CAP = float(os.getenv("APEX_MAX_DD_CAP", "30.0") or "30.0")
OBJECTIVE_PROFILE = str(os.getenv("APEX_OBJECTIVE_PROFILE", "superperformance") or "superperformance").strip().lower()
if OBJECTIVE_PROFILE not in {"superperformance", "balanced", "defensive"}:
    OBJECTIVE_PROFILE = "superperformance"
_TARGET_DEFAULT = "35.0" if OBJECTIVE_PROFILE == "superperformance" else "25.0"
_MIN_CAGR_DEFAULT = "20.0" if OBJECTIVE_PROFILE == "superperformance" else "10.0"
TARGET_CAGR = float(os.getenv("APEX_TARGET_CAGR", _TARGET_DEFAULT) or _TARGET_DEFAULT)
MIN_CAGR_FLOOR = float(os.getenv("APEX_MIN_CAGR_FLOOR", _MIN_CAGR_DEFAULT) or _MIN_CAGR_DEFAULT)
MIN_PF_FLOOR = float(os.getenv("APEX_MIN_PF_FLOOR", "1.20") or "1.20")
MIN_WINLOSS_RATIO = float(os.getenv("APEX_MIN_WINLOSS_RATIO", "2.50") or "2.50")
MIN_TRADES_FLOOR = int(os.getenv("APEX_MIN_TRADES_FLOOR", "120") or "120")
MAX_TRADES_SOFT = int(os.getenv("APEX_MAX_TRADES_SOFT", "700") or "700")
IMMIGRANT_FRAC = float(os.getenv("APEX_IMMIGRANT_FRAC", "0.30") or "0.30")
ELITE_COUNT = int(os.getenv("APEX_ELITE_COUNT", "5") or "5")

# --- GENOME SPACE (Optimization Variables) ---
GENE_SPACE = {
    "require_trend": [True],
    "rs_gate_min": [70, 75, 80, 85, 90],
    "market_cap_min": [500_000_000],
    "mom_rank_min": [75, 80, 90, 95],
    "runup_3m_min_pct": [15, 20, 30, 40, 50],
    "adr_min": [2, 2.5, 3, 3.5, 4],
    "template_low_52w_min_pct": [25, 30, 35],
    "template_off_high_52w_max_pct": [20, 25, 30],
    "sma200_trend_lookback": [20, 30],
    "sma200_trend_min_pct": [0, 0.5, 1],
    "min_price": [2, 5],
    "min_avg_volume_30": [100000, 250000, 500000],
    "require_rs_line_trend": [True],

    "natr_max": [2, 2.5, 3],
    "natr_days": [10],
    "natr_entry_max_pct": [7, 9, 12],
    "vol_mult": [1.25, 1.5, 2, 2.5],
    "breakout_buffer": [0, 0.001],
    "vcp_last_contraction_max_pct": [10, 12, 14, 16],
    "vcp_required_contractions": [1],
    "vcp_damping_ratio": [0.85, 0.9, 0.95],
    "vcp_volume_dryup_max_ratio": [0.9, 1, 1.1],
    "vcp_vol_contraction_ratio": [0.9, 1],
    "vcp_require_pre_breakout": [True],
    "vcp_not_breakout_buffer": [0, 0.002],

    "entry_mode": ["both"],
    "ep_gap_pct": [0.015, 0.02, 0.03, 0.04, 0.05, 0.06],
    "ep_vol_mult": [1.5, 2, 2.5, 3],
    "score_mode": ['dual_core'],
    "technical_weight": [0.60, 0.70, 0.80],
    "fundamental_weight": [0.40, 0.30, 0.20],
    "min_entry_score": [75, 80, 85, 90, 95],

    "max_stop_pct": [0.04, 0.05, 0.06, 0.07],
    "stop_limit_pct": [0.02, 0.03, 0.05],
    "stop_loss_atr_bull": [3, 3.5, 4, 5],
    "stop_loss_atr_bear": [0.5, 0.75],

    "breakeven_at_pct": [0.15, 0.2, 0.25, 0.3],
    "exit_sma_fast": ['ema10', 'ema20'],
    "exit_sma_slow": ['ema10', 'ema20', 'sma50'],
    "enable_partial_profit": [True],
    "partial_profit_mode": ['r'],
    "partial_profit_r": [3, 4, 5],
    "take_profit_chunk_pct": [0.33, 0.5],
    "profit_target_pct": [0.15, 0.2, 0.25, 0.3, 0.4],
    "time_stop_days": [0, 90, 120, 180],

    "pyramid_threshold": [0.05, 0.07, 0.10, 0.12],
    "pyramid_fraction": [0.5],
    "pyramid_max_adds": [2, 3],

    "max_positions": [3, 4, 5, 6],
    "risk_per_trade": [0.0125, 0.015, 0.02, 0.025],
    "max_pos_size_pct": [0.15, 0.2, 0.25],
    "max_total_exposure_pct_bull": [0.8, 0.9, 1],
    "max_total_exposure_pct_bear": [0, 0.05, 0.1],
}

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

def compress_data(prepared_obj):
    print("🗜️  Compressing Data (float32)...")
    for sym, s_data in prepared_obj.enriched.items():
        cols = s_data.df.select_dtypes(include=['float64']).columns
        s_data.df[cols] = s_data.df[cols].astype('float32')
        try:
            for attr in ['close', 'high', 'low', 'open', 'volume', 'rs_rating', 'adx', 'sma50', 'sma200']:
                if hasattr(s_data, attr):
                    val = getattr(s_data, attr)
                    if isinstance(val, np.ndarray) and val.dtype == 'float64':
                        setattr(s_data, attr, val.astype('float32'))
        except Exception: pass
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
        # Default to bull-deploy/bear-cash behavior for superperformance tuning.
        exposure_mode = str(
            os.getenv("APEX_MARKET_EXPOSURE_MODE", strategy_config.get("market_exposure_mode", "filter"))
            or "filter"
        ).strip().lower()
        if exposure_mode not in {"filter", "hard", "hybrid", "scaled", "exposure"}:
            exposure_mode = "filter"
        strategy_config["market_exposure_mode"] = exposure_mode
        strategy_config["bear_max_positions"] = int(os.getenv("APEX_BEAR_MAX_POSITIONS", "1") or "1")
        strategy_config["regime_filter"] = True
        strategy_config["regime_exit"] = True
        strategy_config["market_filter_mode"] = "sma200"
        strategy_config["regime_ma"] = "sma200"
        strategy_config["use_market_regime_traffic_light"] = True
        if "stop_loss_atr_bull" in strategy_config:
            strategy_config["stop_loss_atr"] = strategy_config.get("stop_loss_atr_bull")
        partial_enabled = bool(strategy_config.get("enable_partial_profit", True))
        strategy_config["split_exit"] = partial_enabled
        strategy_config["exit_ma_after_partial"] = strategy_config.get("exit_sma_slow")
        strategy_config["partial_profit_fraction"] = float(strategy_config.get("take_profit_chunk_pct", 0.5) or 0.5)
        strategy_config["partial_profit_pct"] = float(strategy_config.get("profit_target_pct", 0.08) or 0.08)
        pp_mode = str(strategy_config.get("partial_profit_mode", "r") or "r").lower().strip()
        if pp_mode not in {"r", "pct", "time"}:
            pp_mode = "r"
        strategy_config["partial_profit_mode"] = pp_mode
        pp_r = float(strategy_config.get("partial_profit_r", 3.0) or 3.0)
        if not np.isfinite(pp_r) or pp_r <= 0:
            pp_r = 3.0
        strategy_config["partial_profit_r"] = pp_r
        strategy_config["enable_partial_profit"] = partial_enabled
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

        # DEATH PENALTY: Disqualify high drawdown genomes
        if max_dd > float(max_dd_cap):
            return {
                "id": genome_id,
                "genome": genome,
                "score": -100.0,
                "cagr": cagr_pct,
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

        # FITNESS FUNCTION: CAGR-first with quality floors to target superperformance.
        dd_floor = max(max_dd, 1.0)
        calmar = cagr_pct / dd_floor if dd_floor > 0 else cagr_pct
        pf_clipped = max(0.0, min(float(pf), 5.0))
        ratio_clipped = max(0.0, min(float(win_loss_ratio), 8.0))

        cagr_curve = cagr_pct
        if cagr_pct > 10.0:
            cagr_curve += (cagr_pct - 10.0) * 0.75
        if cagr_pct > 20.0:
            cagr_curve += (cagr_pct - 20.0) * 1.25
        if cagr_pct > 30.0:
            cagr_curve += (cagr_pct - 30.0) * 2.0

        score = cagr_curve * 4.0
        score += calmar * 4.0
        score += (pf_clipped - 1.0) * 12.0
        score += (ratio_clipped - 2.0) * 8.0

        # Hard-bias away from low-return / poor-quality profiles.
        if cagr_pct < MIN_CAGR_FLOOR:
            score -= (MIN_CAGR_FLOOR - cagr_pct) * 2.0
        if pf_clipped < MIN_PF_FLOOR:
            score -= (MIN_PF_FLOOR - pf_clipped) * 16.0
        if ratio_clipped < MIN_WINLOSS_RATIO:
            score -= (MIN_WINLOSS_RATIO - ratio_clipped) * 7.0
        if trades < MIN_TRADES_FLOOR:
            score -= (MIN_TRADES_FLOOR - trades) / 2.0
        elif trades <= 350 and cagr_pct > 0:
            score += 6.0
        if max_dd > 35.0:
            score -= (max_dd - 35.0) * 1.0
        if trades > MAX_TRADES_SOFT:
            score -= (trades - MAX_TRADES_SOFT) / 45.0
        if cagr_pct <= 0.0:
            score -= 20.0

        is_super_candidate = (
            cagr_pct >= TARGET_CAGR
            and pf_clipped >= max(1.20, MIN_PF_FLOOR)
            and ratio_clipped >= max(2.5, MIN_WINLOSS_RATIO)
        )
        if is_super_candidate:
            score += 35.0

        return {
            "id": genome_id,
            "genome": genome_adj,
            "score": score,
            "cagr": cagr_pct,
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
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError: pass

    print(f"🚀 PROJECT APEX: Strategic Nuclear Reset")
    print(f"HARDWARE: M3 Max | WORKERS: {MAX_WORKERS}")
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
    total_days = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
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
        prepared = compress_data(prepared)
    else:
        print(f"📊 DATA POOL: {len(prepared.enriched)} tickers prepared (cache).")

    checkpoint = None # START FRESH
    start_gen = 0
    
    population = [generate_random_genome() for _ in range(POPULATION_SIZE)]
    dd_cap = MAX_DD_CAP
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(prepared, g_data),
    ) as executor:
        for gen in range(start_gen, GENERATIONS):
            print(f"\n🧬 GEN {gen+1}/{GENERATIONS} (Superperformance)")
            start_time = time.time()

            tasks = [(i, g, dd_cap) for i, g in enumerate(population)]
            results = []

            while True:
                results = []
                futures = [executor.submit(evaluate_genome, task) for task in tasks]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    results.append(res)
                    if "error" not in res:
                        if res.get("disqualified"):
                            print(f"   > T:{res.get('trades', 0)} | CAGR:{res.get('cagr', 0):.1f}% | DD:{res.get('dd', 0):.1f}% | Fitness:-100 (DD Cap)")
                        else:
                            print(
                                f"   > T:{res['trades']} | CAGR:{res['cagr']:.1f}% | DD:{res['dd']:.1f}% "
                                f"| PF:{res.get('pf', 0.0):.2f} | Calmar:{res.get('calmar', 0):.2f}"
                            )
                    else:
                        print(f"   ⚠️  GENOME {res['id']} FAILED: {res['error']}")

                valid = [r for r in results if "error" not in r and not r.get("disqualified")]
                if valid:
                    break
                if dd_cap < 30.0:
                    dd_cap = 30.0
                    tasks = [(i, g, dd_cap) for i, g in enumerate(population)]
                    print("⚠️  All genomes disqualified at 25% DD. Loosening cap to 30% and retrying this generation.")
                    continue
                print("CRITICAL: All failed or disqualified.")
                break

            if not valid:
                break

            valid.sort(key=lambda x: x["score"], reverse=True)
            winner = valid[0]

            print(
                f"🏆 WINNER: CAGR {winner['cagr']:.2f}% | DD {winner['dd']:.2f}% | "
                f"PF {winner.get('pf', 0.0):.2f} | Calmar {winner.get('calmar', 0):.2f} | Score: {winner['score']:.2f}"
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
            next_gen = [r["genome"] for r in valid[:elite_n]]
            progress = float(gen + 1) / max(float(GENERATIONS), 1.0)
            mutation_rate = max(0.45, 0.80 - (0.35 * progress))
            if winner.get("cagr", 0.0) < MIN_CAGR_FLOOR:
                mutation_rate = max(mutation_rate, 0.85)

            immigrant_count = int(max(0, round(POPULATION_SIZE * max(0.0, min(IMMIGRANT_FRAC, 0.6)))))
            child_target = max(0, POPULATION_SIZE - immigrant_count)
            parent_pool = valid[: max(10, min(len(valid), 30))]

            while len(next_gen) < child_target:
                p1 = random.choice(parent_pool)["genome"]
                p2 = random.choice(parent_pool)["genome"]
                child = crossover(p1, p2)
                if random.random() < mutation_rate:
                    child = mutate_genome(child)
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
