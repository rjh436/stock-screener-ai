import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.cache_manager import CACHE_DIR  # noqa: E402
from data.loader import clean_dataframe  # noqa: E402
from data.indices import get_index_symbols  # noqa: E402
from execution.engine import run_backtest  # noqa: E402
from strategies.generic import GenericStrategy  # noqa: E402


GEN_CONFIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/generated_strategies.json"))


def load_cached_symbol(sym: str, cutoff) -> pd.DataFrame:
    """Load cached parquet for a symbol without triggering network fetches."""
    path = os.path.join(CACHE_DIR, f"{sym}.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        df = clean_dataframe(df)
        if df is None:
            return None
        df = df[df.index >= cutoff]
        return df if not df.empty else None
    except Exception as e:
        print(f"⚠️ Cache read failed for {sym}: {e}")
        return None


def load_cached_sp1500(days: int = 1260):
    cutoff = datetime.now() - timedelta(days=days)
    data_map = {}

    symbols = get_index_symbols("S&P 1500") or []
    print(f"📦 Loading cached data for {len(symbols)} symbols (S&P 1500)...")
    for sym in symbols:
        df = load_cached_symbol(sym, cutoff)
        if df is not None:
            data_map[sym] = df

    g_data = {}
    for sym in ["SPY", "$VIX", "VIX"]:
        df = load_cached_symbol(sym, cutoff)
        if df is not None:
            g_data[sym] = df

    vix = g_data.get("$VIX")
    if vix is None:
        vix = g_data.get("VIX")
    global_ctx = {"SPY": g_data.get("SPY"), "VIX": vix}
    return data_map, global_ctx


def load_apex_income():
    with open(GEN_CONFIG, "r") as f:
        strategies = json.load(f)
    for strat in strategies:
        if strat.get("name") == "Apex Income (Gen 9)":
            return strat
    return None


def main():
    data_map, global_ctx = load_cached_sp1500(days=1260)
    if not data_map:
        print("❌ No cached S&P 1500 data found; aborting to avoid downloads.")
        sys.exit(1)

    apex_income = load_apex_income()
    if not apex_income:
        print("❌ Apex Income (Gen 9) not found in config.")
        sys.exit(1)

    print("🚀 Running backtest for Apex Income (Gen 9) using cached S&P 1500 data...")
    result = run_backtest(
        GenericStrategy(apex_income),
        data_map,
        None,
        100000.0,
        None,
        global_ctx,
    )

    trades = result.get("total_trades", 0)
    cagr = result.get("cagr", 0.0)
    win_rate = result.get("hit_rate", 0.0)

    print(f"Total Trades: {trades}")
    print(f"CAGR: {cagr:.2%}")
    print(f"Win Rate: {win_rate:.2f}%")

    if trades == 0:
        print("⚠️ WARNING: Strategy found NO setups in S&P 1500.")


if __name__ == "__main__":
    main()
