#!/usr/bin/env python3
from __future__ import annotations

import copy
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from execution.engine import prepare_backtest_data, run_backtest
from strategies.superperformance import SuperperformanceStrategy


START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
START_CASH = 100000.0
UNIVERSE_NAME = str(os.getenv("PHASE3_OPT_UNIVERSE", "RUSSELL3000") or "RUSSELL3000").upper()
DEFAULT_SYMBOL_LIMIT = 300
EXPORT_PATH = os.path.join("exports", "optimize_weights_results.csv")

WEIGHT_GRID: List[Tuple[float, float]] = [
    (0.70, 0.30),
    (0.60, 0.40),
    (0.50, 0.50),
]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _days_for_window(start_date: str, end_date: str, warmup_days: int = 320) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    today = pd.Timestamp.now().normalize()
    span_window = max((end_ts - start_ts).days, 1)
    span_from_today = max((today - start_ts).days, 1)
    return int(max(span_window, span_from_today) + warmup_days)


def _profit_factor(trades_list: List[Dict[str, Any]]) -> float:
    gains = 0.0
    losses = 0.0
    for trade in trades_list or []:
        try:
            pnl = float(trade.get("PnL", 0.0) or 0.0)
        except Exception:
            pnl = 0.0
        if pnl > 0:
            gains += pnl
        elif pnl < 0:
            losses += -pnl
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _sharpe_from_equity_curve(equity_curve: List[Dict[str, Any]]) -> float:
    if not equity_curve or len(equity_curve) < 3:
        return 0.0

    try:
        eq = pd.DataFrame(equity_curve)
        eq["Date"] = pd.to_datetime(eq["Date"], errors="coerce")
        eq["Equity"] = pd.to_numeric(eq["Equity"], errors="coerce")
        eq = eq.dropna(subset=["Date", "Equity"]).sort_values("Date")
        if len(eq) < 3:
            return 0.0

        rets = eq["Equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(rets) < 2:
            return 0.0

        std = float(rets.std(ddof=1) or 0.0)
        if std <= 0:
            return 0.0

        day_diffs = eq["Date"].diff().dt.days.dropna()
        avg_period_days = float(day_diffs.mean()) if not day_diffs.empty else 5.0
        if not np.isfinite(avg_period_days) or avg_period_days <= 0:
            avg_period_days = 5.0
        annual_factor = np.sqrt(252.0 / avg_period_days)
        return float((rets.mean() / std) * annual_factor)
    except Exception:
        return 0.0


def build_superperformance_config(technical_weight: float, fundamental_weight: float) -> Dict[str, Any]:
    cfg = {
        "name": "Superperformance",
        "entry_mode": "both",
        "warmup_bars": 200,
        "rs_gate_min": 80.0,
        "vcp_rs_min": 80.0,
        "vcp_sector_min": 45.0,
        "vcp_rp10_max_pct": 20.0,
        "vcp_rp20_max_pct": 25.0,
        "vcp_rp40_max_pct": 30.0,
        "vcp_last_contraction_max_pct": 12.0,
        "vcp_required_contractions": 2,
        "vcp_volume_dryup_max_ratio": 0.75,
        "ep_rs_min": 65.0,
        "ep_gap_pct": 0.06,
        "ep_vol_mult": 2.0,
        "breakout_buffer": 0.001,
        "max_stop_pct": 0.08,
        "ep_max_stop_pct": 0.20,
        "stop_limit_pct": 0.02,
        "signal_mode": "after_close",
        "risk_per_trade": 0.05,
        "max_positions": 6,
        "max_pos_size_pct": 0.25,
        "max_total_exposure_pct": 1.0,
        "time_stop_days": 60,
        "enable_partial_profit": True,
        "partial_profit_mode": "r",
        "partial_profit_r": 3.0,
        "partial_profit_fraction": 0.50,
        "move_stop_to_be": True,
        "use_market_regime_traffic_light": True,
        "market_filter_mode": "traffic_light",
        "market_exposure_mode": "hybrid",
        "bear_max_positions": 3,
        "technical_weight": float(technical_weight),
        "fundamental_weight": float(fundamental_weight),
        "log_scoring": False,
    }
    return cfg


def _load_symbols() -> List[str]:
    symbols = get_universe_symbols(UNIVERSE_NAME) or []
    if not symbols and UNIVERSE_NAME != "RUSSELL3000":
        symbols = get_universe_symbols("RUSSELL3000") or []
    symbol_limit = max(_env_int("PHASE3_OPT_SYMBOL_LIMIT", DEFAULT_SYMBOL_LIMIT), 0)
    if symbol_limit > 0:
        symbols = symbols[: min(symbol_limit, len(symbols))]
    return symbols


def _safe_result_obj(result: Any) -> Dict[str, Any]:
    if isinstance(result, list):
        return result[0] if result else {}
    if isinstance(result, dict):
        return result
    return {}


def run_grid_search(
    weight_grid: Iterable[Tuple[float, float]] | None = None,
    *,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> Dict[str, Any]:
    weights = list(weight_grid or WEIGHT_GRID)
    symbols = _load_symbols()
    if not symbols:
        raise RuntimeError("No symbols available for optimize_weights universe.")

    days = _days_for_window(start_date, end_date, warmup_days=360)
    print(
        f"[GridSearch] Loading {len(symbols)} symbols from {UNIVERSE_NAME} "
        f"for {start_date} -> {end_date} (days={days})"
    )

    data = fetch_data_pack(symbols, days=days, backtest_mode=True) or {}
    g_data = fetch_data_pack(["SPY", "$VIX", "VIX"], days=days, backtest_mode=True) or {}
    spy_df = g_data.get("SPY")
    vix_df = g_data.get("$VIX") if g_data.get("$VIX") is not None else g_data.get("VIX")
    global_data = {"SPY": spy_df, "VIX": vix_df}

    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=None,
        global_data=global_data,
    )
    if not prepared.enriched:
        raise RuntimeError("Prepared backtest dataset is empty for optimize_weights.")

    rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None

    for technical_weight, fundamental_weight in weights:
        cfg = build_superperformance_config(technical_weight, fundamental_weight)
        strategy = SuperperformanceStrategy(copy.deepcopy(cfg))
        result = _safe_result_obj(
            run_backtest(
                [strategy],
                prepared,
                start_cash=START_CASH,
                start_date=start_date,
                end_date=end_date,
                global_data=global_data,
            )
        )

        trades = result.get("trades_list") or []
        equity_curve = result.get("equity_curve") or []
        sharpe = _sharpe_from_equity_curve(equity_curve)
        pf = _profit_factor(trades)
        cagr = float(result.get("cagr", 0.0) or 0.0) * 100.0
        max_dd = float(result.get("max_drawdown_pct", 0.0) or 0.0)
        if max_dd <= 1.0:
            max_dd *= 100.0

        row = {
            "TechnicalWeight": float(technical_weight),
            "FundamentalWeight": float(fundamental_weight),
            "Sharpe": sharpe,
            "CAGR(%)": cagr,
            "MaxDD(%)": max_dd,
            "ProfitFactor": pf,
            "Trades": int(result.get("total_trades", 0) or 0),
            "FinalValue": float(result.get("final_value", START_CASH) or START_CASH),
        }
        rows.append(row)

        print(
            "[GridSearch] "
            f"tech={technical_weight:.2f} fund={fundamental_weight:.2f} "
            f"Sharpe={sharpe:.3f} PF={pf:.3f} Trades={row['Trades']}"
        )

        if best_row is None or row["Sharpe"] > best_row["Sharpe"]:
            best_row = row

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    pd.DataFrame(rows).sort_values("Sharpe", ascending=False).to_csv(EXPORT_PATH, index=False)

    if best_row is None:
        raise RuntimeError("Grid search did not produce a valid result row.")

    print(
        "Optimal Weight Configuration: "
        f"Technical={best_row['TechnicalWeight']:.2f}, "
        f"Fundamental={best_row['FundamentalWeight']:.2f}, "
        f"Sharpe={best_row['Sharpe']:.3f}"
    )

    return {
        "best": best_row,
        "rows": rows,
        "symbols_requested": len(symbols),
        "symbols_prepared": len(prepared.enriched),
        "export_path": EXPORT_PATH,
    }


if __name__ == "__main__":
    run_grid_search()
