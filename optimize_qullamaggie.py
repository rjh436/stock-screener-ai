import copy
import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.cache_manager import DataCache
from data.loader import clean_dataframe
from data.universe import get_universe_symbols
from execution.engine import prepare_backtest_data, run_backtest
from strategies.generic import GenericStrategy

# --- CONFIGURATION ---
STRATEGY_TEMPLATE = {
    "name": "Apex Kinetic VCP (Optimizer)",
    "type": "breakout",
    "entry_rules": [
        {"col": "rsi14", "op": ">", "val": 60},
        {"col": "rs_rating", "op": ">", "val": 85},
        {"col": "bb_width", "op": "<", "val": 0.20},
        {"col": "close", "op": ">", "ref": "donchian_20"},
        {"col": "volume", "op": ">", "ref": "vol_ma20", "mult": 1.5},
    ],
    "exit_rules": [
        {"col": "close", "op": "<", "ref": "sma20"},
    ],
    "market_filter_mode": "traffic_light",
    "yellow_rs_floor": 90,
    "red_bypass_rs": 95,
    "stop_loss_atr": 2.0,
    "trail_atr": 0,
    "time_stop": 45,
    "risk_per_trade": 0.02,
    "max_positions": 12,
    "scoring_type": "breakout",
    "scoring_weights": {"rsi_factor": 2.0, "sniper_bonus": 150.0},
}

PARAM_GRID = {
    "stop_loss_atr": [1.0, 1.5, 2.0, 2.5],
    "rs_rating": [80, 85, 90],
    "exit_sma": ["sma10", "sma20", "sma50"],
    "red_bypass_rs": [90, 95, 101],
}

_DATA_PACK = None


def _load_cached_symbol(sym: str, cutoff: datetime) -> Optional[pd.DataFrame]:
    df = DataCache.get_cached_data(sym, validate=False, allow_stale=True)
    df = clean_dataframe(df)
    if df is None or df.empty:
        return None
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index >= cutoff]
    return df if not df.empty else None


def _load_cached_globals(cutoff: datetime) -> Dict[str, Optional[pd.DataFrame]]:
    spy = _load_cached_symbol("SPY", cutoff)
    vix = _load_cached_symbol("VIX", cutoff) or _load_cached_symbol("$VIX", cutoff)
    return {"SPY": spy, "VIX": vix}


def get_data(days: int = 365 * 20):
    print("Loading cached data...")
    cutoff = datetime.now() - timedelta(days=days)

    symbols = get_universe_symbols("RUSSELL3000") or []
    if not symbols:
        symbols = get_universe_symbols("SP1500") or []

    if not symbols:
        return None

    data_map: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _load_cached_symbol(sym, cutoff)
        if df is not None:
            data_map[sym] = df

    if not data_map:
        return None

    global_ctx = _load_cached_globals(cutoff)
    prepared = prepare_backtest_data(data_map, None, None, global_ctx)
    return prepared


def _init_worker(data_pack):
    global _DATA_PACK
    _DATA_PACK = data_pack


def _normalize_dd(raw_dd: float) -> float:
    if raw_dd is None:
        return 0.0
    try:
        dd_val = float(raw_dd)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(dd_val):
        return 0.0
    if abs(dd_val) > 1.0:
        return dd_val / 100.0
    return dd_val


def worker(params: Dict[str, object]):
    global _DATA_PACK
    if _DATA_PACK is None:
        return {"error": "data_not_loaded"}

    try:
        strat = copy.deepcopy(STRATEGY_TEMPLATE)
        strat["name"] = f"QM_Stop{params['stop_loss_atr']}_RS{params['rs_rating']}_{params['exit_sma']}"

        strat["stop_loss_atr"] = params["stop_loss_atr"]
        strat["red_bypass_rs"] = params["red_bypass_rs"]

        entry_rules = [r.copy() for r in strat["entry_rules"]]
        for rule in entry_rules:
            if rule.get("col") == "rs_rating":
                rule["val"] = params["rs_rating"]
        strat["entry_rules"] = entry_rules

        strat["exit_rules"] = [{"col": "close", "op": "<", "ref": params["exit_sma"]}]

        result = run_backtest(GenericStrategy(strat), _DATA_PACK, start_cash=100000.0)

        cagr = float(result.get("cagr", 0.0) or 0.0)
        max_dd = _normalize_dd(result.get("max_drawdown_pct", 0.0))
        trades = int(result.get("total_trades", 0) or 0)
        win_rate = float(result.get("hit_rate", 0.0) or 0.0)
        final_equity = float(result.get("final_value", 0.0) or 0.0)

        return {
            "params": params,
            "cagr": cagr,
            "max_dd": max_dd,
            "trades": trades,
            "win_rate": win_rate,
            "final_equity": final_equity,
        }
    except Exception as exc:
        return {"error": str(exc), "params": params}


def optimize():
    data_pack = get_data()
    if data_pack is None:
        print("No cached data loaded. Check cache or data loader.")
        return

    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, vals)) for vals in itertools.product(*values)]

    print(f"Starting optimization: {len(combinations)} strategies")

    results = []
    max_workers = min(os.cpu_count() or 1, len(combinations))

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(data_pack,)) as executor:
        futures = [executor.submit(worker, combo) for combo in combinations]
        for i, fut in enumerate(as_completed(futures), start=1):
            res = fut.result()
            if "error" in res:
                continue

            dd = abs(res["max_dd"]) if res["max_dd"] != 0 else 0.0001
            score = res["cagr"] / dd
            res["score"] = score
            results.append(res)

            print(
                f"[{i}/{len(combinations)}] Stop:{res['params']['stop_loss_atr']} "
                f"Exit:{res['params']['exit_sma']} "
                f"CAGR:{res['cagr']:.2%} DD:{res['max_dd']:.2%} Trades:{res['trades']}"
            )

    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results).sort_values("score", ascending=False)

    print("\nTop 5 configurations")
    print(df.head(5)[["params", "cagr", "max_dd", "trades", "win_rate", "score"]])

    best = df.iloc[0]
    print("\nWinner")
    print(f"Params: {best['params']}")
    print(f"CAGR: {best['cagr']:.2%}")
    print(f"Drawdown: {best['max_dd']:.2%}")


if __name__ == "__main__":
    optimize()
