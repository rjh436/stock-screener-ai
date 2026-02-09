#!/usr/bin/env python3
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.append(PROJECT_ROOT)

import execution.engine as engine
from execution.engine import run_backtest
from data.loader import fetch_data_pack
from strategies.superperformance import SuperperformanceStrategy


def _calc_trading_days(start_date: str, end_date: str) -> int:
    total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    return int((total_days / 365.25) * 252) + 300


def main() -> int:
    symbols = ["NVDA", "SMCI"]
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    trading_days = _calc_trading_days(start_date, end_date)

    print("=== PARABOLIC GEAR SMOKE TEST ===")
    print(f"Symbols: {symbols}")
    print(f"Window: {start_date} -> {end_date}")

    print("...Loading data...")
    data = fetch_data_pack(symbols, days=trading_days, backtest_mode=True)
    g_data = fetch_data_pack(["SPY", "VIX"], days=trading_days, backtest_mode=True)

    # Avoid loading the full-universe indicator cache for a 2-symbol smoke test.
    engine._INDICATOR_CACHE_PATH = os.path.join("/tmp", "smoke_test_indicators.pkl")
    prepared = engine.prepare_backtest_data(data, symbols, start_date=start_date, global_data=g_data)

    # Permissive params to ensure at least one trade for smoke validation.
    params = {
        "name": "Superperformance Smoke",
        "require_trend": True,
        "rs_gate_min": 0,
        "vcp_rs_min": 0,
        "vcp_sector_min": 0,
        "ep_rs_min": 0,
        "market_cap_min": 0,
        "entry_mode": "both",
        "ep_gap_pct": 0.04,
        "ep_vol_mult": 1.0,
        "breakout_buffer": 0.001,
        "stop_limit_pct": 0.1,
        "max_stop_pct": 0.05,
        "ep_max_stop_pct": 0.15,
        "breakeven_at_pct": 0.08,
        "exit_sma_fast": "ema10",
        "exit_sma_slow": "sma50",
        "take_profit_chunk_pct": 0.33,
        "profit_target_pct": 0.08,
        "time_stop_days": 5,
        "pyramid_threshold": 0.05,
        "pyramid_fraction": 0.5,
        "pyramid_max_adds": 1,
        "max_positions": 2,
        "risk_per_trade": 0.01,
        "max_pos_size_pct": 0.2,
        "max_total_exposure_pct_bull": 1.0,
        "max_total_exposure_pct_bear": 0.5,
    }

    strat = SuperperformanceStrategy(params)
    result = run_backtest(strat, prepared, start_cash=100000.0, start_date=start_date, end_date=end_date, global_data=g_data)
    if isinstance(result, list):
        result = result[0] if result else {}

    trades = result.get("trades_list", []) if isinstance(result, dict) else []
    print(f"Trades: {len(trades)}")
    if trades:
        print("Sample exits:")
        for t in trades[:3]:
            print(f"  {t.get('Symbol')} | {t.get('Reason')} | Return%={t.get('Return %'):.2f}")

    if not trades:
        print("FAIL: No trades/exits found in smoke test.")
        return 1

    print("PASS: Smoke test completed with exits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
