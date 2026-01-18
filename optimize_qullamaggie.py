import os
import json
import itertools
import pandas as pd
import numpy as np
import sys
import random
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# CORRECTED IMPORTS
from data.loader import fetch_data_pack
from data.indices import get_index_symbols
from execution.engine import prepare_backtest_data, run_backtest
from strategies.generic import GenericStrategy
from data.cache_manager import DataCache
from data.loader import clean_dataframe

# --- CONFIGURATION ---
STRATEGY_TEMPLATE = {
    "name": "Apex Kinetic VCP (Optimizer)",
    "type": "breakout",
    "entry_rules": [
        {"col": "rsi14", "op": ">", "val": 55},
        {"col": "rs_rating", "op": ">", "val": 80},
        {"col": "adr_pct", "op": ">", "val": 3.0},
        {"col": "bb_width", "op": "<", "val": 0.25},
        {"col": "close", "op": ">", "ref": "donchian_20"},
        {"col": "volume", "op": ">", "ref": "vol_ma20", "mult": 1.5}
    ],
    "exit_rules": [
        {"col": "close", "op": "<", "ref": "sma20"}
    ],
    "market_filter_mode": "traffic_light",
    "yellow_rs_floor": 85,
    "red_bypass_rs": 95,
    "stop_loss_atr": 2.0,
    "trail_atr": 0,
    "time_stop": 45,
    "risk_per_trade": 0.02,
    "max_positions": 12,
    "scoring_type": "breakout",
    "scoring_weights": {"rsi_factor": 2.0, "sniper_bonus": 150.0},
    "min_entry_score": 0.0
}

# --- THE SEARCH GRID (V2) ---
PARAM_GRID = {
    "stop_loss_atr": [1.5, 2.0, 2.5],        # Risk Management
    "rs_rating":     [80, 85, 90],           # Leader Quality
    "exit_sma":      ["sma10", "sma20"],     # Trend Duration
    "adr_pct":       [2.5, 3.5, 4.5],        # Volatility Fuel
    "red_bypass_rs": [95, 101],              # 101 = Disabled
    "rsi14":         [50, 55],               # Momentum Strength
    "bb_width":      [0.18, 0.25],           # Volatility Squeeze
    "vol_mult":      [1.2, 1.5],             # Volume Expansion
    "trail_atr":     [0.0, 2.0],             # Trailing Stop
    "time_stop":     [45, 90]                # Max Hold (days)
}

_DATA_PACK = None
_PREPARED_CACHE = None
_FINALISTS = int(os.environ.get("APEX_OPT_FINALISTS", "8"))
_MAX_COMBOS = int(os.environ.get("APEX_OPT_MAX_COMBOS", "0"))

_BASE_COLS = ("open", "high", "low", "close", "volume", "vix")

def _init_worker(data_pack):
    global _DATA_PACK
    _DATA_PACK = data_pack

def _trim_df(df):
    if df is None or df.empty:
        return df
    keep = [c for c in _BASE_COLS if c in df.columns]
    if not keep:
        return df
    return df[keep].copy()

def _load_cached_pack(symbols):
    data = {}
    for sym in symbols:
        df = DataCache.get_cached_data(sym, validate=False, allow_stale=True)
        df = clean_dataframe(df)
        df = _trim_df(df)
        if df is not None and not df.empty:
            data[sym] = df
    return data

def _trim_pack(data):
    trimmed = {}
    for sym, df in (data or {}).items():
        df = _trim_df(df)
        if df is not None and not df.empty:
            trimmed[sym] = df
    return trimmed

def _build_universe():
    try:
        # FORCE RUSSELL 3000
        print("Requesting Russell 3000 Universe...")
        universe = get_index_symbols("R3000")

        if not universe or len(universe) < 2000:
            print("Warning: 'R3000' key missing or too small. Trying 'IWV' (Russell 3000 ETF)...")
            universe = get_index_symbols("IWV")

        if not universe:
            # Absolute fallback if indices file is empty
            print("Warning: 'IWV' failed. Loading S&P 1500 as fallback.")
            universe = get_index_symbols("SP1500")

        print(f"Fetching data for {len(universe)} symbols...")
        return universe

    except Exception as e:
        print(f"ERROR loading data: {e}")
        return []

def _load_data_pack(universe):
    """Load cached data for a universe, fallback to fetch if too sparse."""
    if not universe:
        return {}
    cached = _load_cached_pack(universe)
    min_required = max(10, int(len(universe) * 0.6))
    min_required = min(500, min_required)
    if cached and len(cached) >= min_required:
        print(f"Using cached data for {len(cached)} symbols...")
        return cached
    data = fetch_data_pack(universe, backtest_mode=True)
    if not data:
        data = fetch_data_pack(universe)
    return _trim_pack(data)

def worker(params):
    """Runs a single backtest for a parameter set"""
    try:
        prepared_cache = _PREPARED_CACHE
        if prepared_cache is None:
            return {"error": "Missing pre-calculated data cache in worker"}
        # Clone Strategy
        strat = STRATEGY_TEMPLATE.copy()
        strat["name"] = f"QM_Stop{params['stop_loss_atr']}_ADR{params['adr_pct']}_{params['exit_sma']}"

        # Apply Scalar Params
        strat["stop_loss_atr"] = params["stop_loss_atr"]
        strat["red_bypass_rs"] = params["red_bypass_rs"]

        # Update Rules
        entry_rules = [r.copy() for r in strat["entry_rules"]]
        for r in entry_rules:
            if r["col"] == "rsi14":
                r["val"] = params["rsi14"]
            if r["col"] == "rs_rating":
                r["val"] = params["rs_rating"]
            if r["col"] == "adr_pct":
                r["val"] = params["adr_pct"]
            if r["col"] == "bb_width":
                r["val"] = params["bb_width"]
            if r["col"] == "volume":
                r["mult"] = params["vol_mult"]
        strat["entry_rules"] = entry_rules

        strat["exit_rules"] = [{"col": "close", "op": "<", "ref": params["exit_sma"]}]
        strat["trail_atr"] = params["trail_atr"]
        strat["time_stop"] = params["time_stop"]

        # Run Backtest
        res = run_backtest(
            GenericStrategy(strat),
            None,
            None,
            start_cash=100000.0,
            pre_calculated_data=prepared_cache,
        )

        return {
            "params": params,
            "cagr": res.get("cagr", 0),
            "max_dd": res.get("max_drawdown_pct", 0),
            "trades": res.get("total_trades", 0),
            "win_rate": res.get("hit_rate", 0),
            "profit_factor": res.get("profit_factor", 0)
        }
    except Exception as e:
        return {"error": str(e)}

def optimize():
    print("Loading Data Pack...")
    full_universe = _build_universe()
    if not full_universe:
        print("ERROR: No universe loaded. Check indices.")
        return

    data_pack = _load_data_pack(full_universe)
    if not data_pack:
        print("ERROR: No data loaded. Check loader.")
        return
    print(f"🚀 Starting V2 Optimization on FULL UNIVERSE ({len(data_pack)} symbols)...")
    global _DATA_PACK
    _DATA_PACK = data_pack
    print("🧠 Pre-calculating indicators for the entire universe (Once)...")
    global _PREPARED_CACHE
    _PREPARED_CACHE = prepare_backtest_data(data_pack, None, None, None)
    print(f"✅ Pre-calculation complete. Cached {len(_PREPARED_CACHE.enriched)} symbols.")
    print("Pre-calculation cache active for optimization.")

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    if _MAX_COMBOS > 0 and len(combinations) > _MAX_COMBOS:
        random.seed(42)
        combinations = random.sample(combinations, _MAX_COMBOS)
        print(f"Sampling {len(combinations)} parameter combinations...")

    print(f"Starting V2 Optimization: {len(combinations)} Strategies")

    def run_pool(label):
        results = []
        errors = []
        cpu_count = os.cpu_count() or 1
        max_workers = cpu_count + 4
        # Use ThreadPoolExecutor to prevent OOM (Out Of Memory) on macOS.
        # Threads share memory; processes duplicate it.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, combo) for combo in combinations]

            for i, f in enumerate(futures):
                res = f.result()
                if "error" in res:
                    errors.append(res["error"])
                    continue
                # Score = CAGR but kill if DD > 30% or Trades < 50
                score = res["cagr"]
                if abs(res["max_dd"]) > 30.0:
                    score *= 0.5
                if res["trades"] < 50:
                    score = 0

                res["score"] = score
                results.append(res)
                print(f"[{i+1}/{len(combinations)}] ADR:{res['params']['adr_pct']} Stop:{res['params']['stop_loss_atr']} -> CAGR: {res['cagr']:.1%} DD: {res['max_dd']:.1f}%")
        if errors:
            print(f"{label} errors: {len(errors)}")
            print("Sample error:", errors[0])
        return results

    results = run_pool("ThreadPoolExecutor")
    if not results:
        print("No results from ThreadPoolExecutor, retrying...")
        results = run_pool("ThreadPoolExecutor")
        if not results:
            print("ERROR: No results generated.")
            return
    results = sorted(results, key=lambda r: r.get("score", 0), reverse=True)

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    print("\nTOP 5 V2 CONFIGURATIONS")
    print(df.head(5)[["params", "cagr", "max_dd", "trades", "profit_factor"]].to_string(index=False))

    best = df.iloc[0]
    print(f"\nWINNER: {best['params']}")

if __name__ == "__main__":
    optimize()
