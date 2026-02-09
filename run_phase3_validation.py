#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from execution.engine import prepare_backtest_data, run_backtest
from execution.market_regime import compute_regime_series
from optimize_weights import build_superperformance_config, run_grid_search
from strategies.superperformance import SuperperformanceStrategy


START_CASH = 100000.0
KNOWN_WINNERS: List[Tuple[str, str, str]] = [
    ("TSLA", "2020-01-01", "2020-12-31"),
    ("NVDA", "2023-01-01", "2023-12-31"),
    ("SMCI", "2023-01-01", "2023-12-31"),
]
FULL_START_DATE = "2021-01-01"
FULL_END_DATE = pd.Timestamp.now().date().isoformat()
FULL_EXPORT_PATH = os.path.join("exports", "phase3_results.csv")
DEFAULT_TUNED_CONFIG_PATH = os.path.join("config", "superperformance_winner.json")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _days_for_window(start_date: str, end_date: str, warmup_days: int = 320) -> int:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    today = pd.Timestamp.now().normalize()
    span_window = max((end_ts - start_ts).days, 1)
    span_from_today = max((today - start_ts).days, 1)
    return int(max(span_window, span_from_today) + warmup_days)


def _safe_result_obj(result: Any) -> Dict[str, Any]:
    if isinstance(result, list):
        return result[0] if result else {}
    if isinstance(result, dict):
        return result
    return {}


def _load_strategy_override(path: str | None) -> Dict[str, Any]:
    raw = str(path or "").strip()
    if not raw:
        return {}
    if not os.path.exists(raw):
        print(f"[Config] Strategy override file not found: {raw}. Using defaults.")
        return {}
    try:
        with open(raw, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"[Config] Loaded strategy override: {raw}")
            return data
    except Exception as exc:
        print(f"[Config] Failed to load strategy override '{raw}': {exc}")
    return {}


def _compose_strategy_config(
    technical_weight: float,
    fundamental_weight: float,
    strategy_override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = build_superperformance_config(technical_weight, fundamental_weight)
    if isinstance(strategy_override, dict) and strategy_override:
        cfg.update(copy.deepcopy(strategy_override))
    # Keep explicit CLI weights authoritative for this validation run.
    cfg["technical_weight"] = float(technical_weight)
    cfg["fundamental_weight"] = float(fundamental_weight)
    return cfg


def _profit_factor(trades_list: List[Dict[str, Any]]) -> float:
    gains = 0.0
    losses = 0.0
    for trade in trades_list or []:
        pnl = _safe_float(trade.get("PnL", 0.0), 0.0)
        if pnl > 0:
            gains += pnl
        elif pnl < 0:
            losses += -pnl
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _average_winner_loser(trades_list: List[Dict[str, Any]]) -> Tuple[float, float]:
    wins: List[float] = []
    losses: List[float] = []
    for trade in trades_list or []:
        pnl = _safe_float(trade.get("PnL", 0.0), 0.0)
        if pnl > 0:
            wins.append(pnl)
        elif pnl < 0:
            losses.append(abs(pnl))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_win, avg_loss


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


def _looks_like_rate_limit(message: str) -> bool:
    text = str(message or "").lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "too many requests",
            "429",
            "thrott",
            "quota",
            "temporarily unavailable",
        )
    )


def _fetch_data_pack_with_retry(
    symbols: List[str],
    *,
    days: int,
    label: str,
    max_attempts: int = 3,
    min_coverage: float = 0.60,
    backtest_mode: bool = False,
    force_fresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    total = max(len(symbols), 1)
    last_data: Dict[str, pd.DataFrame] = {}

    for attempt in range(1, max_attempts + 1):
        try:
            data = fetch_data_pack(
                symbols,
                days=days,
                backtest_mode=backtest_mode,
                force_fresh=force_fresh,
            ) or {}
        except Exception as exc:
            print(f"[{label}] Fetch attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts and _looks_like_rate_limit(str(exc)):
                print(f"[{label}] API limits detected. Cooling down for 60 seconds before retry.")
                time.sleep(60)
                continue
            if attempt < max_attempts:
                time.sleep(5)
                continue
            return {}

        coverage = len(data) / float(total)
        print(
            f"[{label}] Loaded {len(data)}/{len(symbols)} symbols "
            f"({coverage:.1%}) on attempt {attempt}/{max_attempts}"
        )
        last_data = data

        if coverage >= min_coverage:
            return data
        if attempt >= max_attempts:
            return data

        print(
            f"[{label}] Coverage below threshold ({min_coverage:.0%}). "
            "Cooling down for 60 seconds and retrying."
        )
        time.sleep(60)

    return last_data


def _breakout_gate_reason(df: pd.DataFrame, i: int, params: Dict[str, Any]) -> Tuple[bool, str]:
    row = df.iloc[i]
    close_px = _safe_float(row.get("close"), 0.0)
    sma200 = _safe_float(row.get("sma200"), 0.0)
    rs_rating = _safe_float(row.get("rs_rating"), 0.0)
    sector_rs = _safe_float(row.get("sector_rs"), 50.0)

    if close_px <= sma200:
        return False, "Rejected by Trend (close <= SMA200)"
    if rs_rating < float(params.get("vcp_rs_min", 80.0) or 80.0):
        return False, "Rejected by RS floor (VCP)"
    if sector_rs < float(params.get("vcp_sector_min", 50.0) or 50.0):
        return False, "Rejected by Sector RS"

    rp5 = _safe_float(row.get("range_pct_5"), np.nan)
    rp10 = _safe_float(row.get("range_pct_10"), np.nan)
    rp20 = _safe_float(row.get("range_pct_20"), np.nan)
    rp40 = _safe_float(row.get("range_pct_40"), np.nan)
    last_contraction = np.nanmin([rp5, rp10]) if np.isfinite(rp5) or np.isfinite(rp10) else np.nan
    vcp_rp10_max = float(params.get("vcp_rp10_max_pct", 20.0) or 20.0)
    vcp_rp20_max = float(params.get("vcp_rp20_max_pct", 25.0) or 25.0)
    vcp_rp40_max = float(params.get("vcp_rp40_max_pct", 30.0) or 30.0)
    vcp_last_contraction_max = float(params.get("vcp_last_contraction_max_pct", 12.0) or 12.0)
    vcp_required_contractions = int(params.get("vcp_required_contractions", 2) or 2)

    contractions = 0
    if np.isfinite(rp10) and rp10 <= vcp_rp10_max:
        contractions += 1
    if np.isfinite(rp20) and rp20 <= vcp_rp20_max:
        contractions += 1
    if np.isfinite(rp40) and rp40 <= vcp_rp40_max:
        contractions += 1

    if not np.isfinite(last_contraction) or last_contraction > vcp_last_contraction_max:
        return False, "Rejected by VCP tightness"
    if contractions < vcp_required_contractions:
        return False, "Rejected by VCP contraction count"

    if i > 0:
        prev = df.iloc[i - 1]
        vol_prev = _safe_float(prev.get("volume"), np.nan)
        vol_ma50_prev = _safe_float(prev.get("vol_ma50"), np.nan)
        dryup_max_ratio = float(params.get("vcp_volume_dryup_max_ratio", 0.75) or 0.75)
        if np.isfinite(vol_ma50_prev) and vol_ma50_prev > 0 and vol_prev > (vol_ma50_prev * dryup_max_ratio):
            return False, "Rejected by VCP volume dry-up"

    prior_high = _safe_float(
        row.get("high_20_prev")
        or row.get("highest10_1")
        or row.get("prev_high"),
        0.0,
    )
    if prior_high <= 0:
        return False, "Rejected by missing pivot trigger"

    return True, "Breakout pass"


def _ep_gate_reason(df: pd.DataFrame, i: int, params: Dict[str, Any]) -> Tuple[bool, str]:
    if i < 1:
        return False, "Rejected by EP warmup"

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    prev_close = _safe_float(prev.get("close"), 0.0)
    open_px = _safe_float(row.get("open"), 0.0)
    close_px = _safe_float(row.get("close"), 0.0)
    high_px = _safe_float(row.get("high"), 0.0)
    low_px = _safe_float(row.get("low"), 0.0)
    vol = _safe_float(row.get("volume"), 0.0)
    vol_ma50 = _safe_float(row.get("vol_ma50"), 0.0)
    rs_rating = _safe_float(row.get("rs_rating"), 0.0)

    if prev_close <= 0 or open_px <= 0:
        return False, "Rejected by EP setup data"

    gap_pct = (open_px - prev_close) / prev_close
    ep_gap = max(float(params.get("ep_gap_pct", 0.06) or 0.06), 0.06)
    if gap_pct < ep_gap:
        return False, "Rejected by EP gap threshold"

    ep_vol_mult = max(float(params.get("ep_vol_mult", 2.0) or 2.0), 2.0)
    if vol_ma50 <= 0 or vol < (vol_ma50 * ep_vol_mult):
        return False, "Rejected by EP volume surge"

    if close_px <= open_px:
        return False, "Rejected by EP close<open"

    clv = _safe_float(row.get("clv"), np.nan)
    if not np.isfinite(clv):
        rng = max(high_px - low_px, 0.0)
        clv = ((close_px - low_px) / rng) if rng > 0 else 0.0
    if clv <= 0.70:
        return False, "Rejected by EP CLV"

    sma50 = _safe_float(row.get("sma50"), 0.0)
    if sma50 > 0 and close_px <= sma50:
        return False, "Rejected by EP SMA50"

    ep_rs_min = float(params.get("ep_rs_min", 60.0) or 60.0)
    if rs_rating < ep_rs_min:
        return False, "Rejected by EP RS floor"

    day_range = _safe_float(row.get("true_range"), 0.0)
    if day_range <= 0:
        day_range = max(high_px - low_px, 0.0)
    tr_ma50 = _safe_float(row.get("tr_ma50"), 0.0)
    if tr_ma50 > 0 and day_range <= (1.25 * tr_ma50):
        return False, "Rejected by EP range expansion"

    return True, "EP pass"


def _analyze_miss_report(
    df: pd.DataFrame | None,
    params: Dict[str, Any],
    start_date: str,
    end_date: str,
    spy_df: pd.DataFrame | None,
) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"summary": "No price history available for diagnostic window."}

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    warmup = int(params.get("warmup_bars", 200) or 200)
    checked = 0
    signal_ready = 0
    vcp_rejects: Dict[str, int] = {}
    ep_rejects: Dict[str, int] = {}

    for i, ts in enumerate(df.index):
        ts = pd.Timestamp(ts)
        if ts < start_ts or ts > end_ts:
            continue
        if i < warmup:
            continue

        checked += 1
        row = df.iloc[i]
        close_px = _safe_float(row.get("close"), 0.0)
        rs_rating = _safe_float(row.get("rs_rating"), 0.0)
        natr = _safe_float(row.get("natr"), 0.0)

        if close_px <= 0:
            vcp_rejects["Rejected by invalid close"] = vcp_rejects.get("Rejected by invalid close", 0) + 1
            continue
        if rs_rating < float(params.get("rs_gate_min", 85.0) or 85.0):
            vcp_rejects["Rejected by primary RS gate"] = vcp_rejects.get("Rejected by primary RS gate", 0) + 1
            continue
        natr_guard = float(params.get("natr_entry_max_pct", 7.0) or 7.0)
        if natr > natr_guard:
            vcp_rejects["Rejected by volatility guard (NATR)"] = vcp_rejects.get(
                "Rejected by volatility guard (NATR)", 0
            ) + 1
            continue

        breakout_ok, breakout_reason = _breakout_gate_reason(df, i, params)
        ep_ok, ep_reason = _ep_gate_reason(df, i, params)

        if breakout_ok or ep_ok:
            signal_ready += 1
        else:
            vcp_rejects[breakout_reason] = vcp_rejects.get(breakout_reason, 0) + 1
            ep_rejects[ep_reason] = ep_rejects.get(ep_reason, 0) + 1

    top_vcp = max(vcp_rejects.items(), key=lambda x: x[1])[0] if vcp_rejects else "No dominant VCP reject"
    top_ep = max(ep_rejects.items(), key=lambda x: x[1])[0] if ep_rejects else "No dominant EP reject"

    red_days = 0
    regime_days = 0
    if spy_df is not None and not spy_df.empty:
        try:
            regime = compute_regime_series(spy_df)
            if not regime.empty:
                sliced = regime[(regime.index >= start_ts) & (regime.index <= end_ts)]
                regime_days = int(len(sliced))
                red_days = int((sliced == "RED").sum())
        except Exception:
            pass

    return {
        "checked_bars": checked,
        "signal_ready_bars": signal_ready,
        "top_vcp_reject": top_vcp,
        "top_ep_reject": top_ep,
        "regime_red_days": red_days,
        "regime_days": regime_days,
        "summary": (
            f"Checked={checked}, ReadySignals={signal_ready}, "
            f"TopVCP='{top_vcp}', TopEP='{top_ep}', "
            f"RegimeRED={red_days}/{regime_days}"
        ),
    }


def _run_single_symbol_diagnostic(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    technical_weight: float,
    fundamental_weight: float,
    strategy_override: Dict[str, Any] | None = None,
    cache_only: bool = False,
) -> Dict[str, Any]:
    days = _days_for_window(start_date, end_date, warmup_days=420)
    fetch_backtest_mode = bool(cache_only)
    fetch_force_fresh = not bool(cache_only)
    fetch_attempts = 1 if cache_only else 2
    price_data = _fetch_data_pack_with_retry(
        [symbol],
        days=days,
        label=f"{symbol}-diagnostic",
        max_attempts=fetch_attempts,
        min_coverage=1.0,
        backtest_mode=fetch_backtest_mode,
        force_fresh=fetch_force_fresh,
    )
    global_raw = _fetch_data_pack_with_retry(
        ["SPY", "$VIX", "VIX"],
        days=days,
        label=f"{symbol}-global",
        max_attempts=fetch_attempts,
        min_coverage=0.66,
        backtest_mode=fetch_backtest_mode,
        force_fresh=fetch_force_fresh,
    )
    spy_df = global_raw.get("SPY")
    vix_df = global_raw.get("$VIX") if global_raw.get("$VIX") is not None else global_raw.get("VIX")
    global_data = {"SPY": spy_df, "VIX": vix_df}

    prev_disable_cache = os.environ.get("APEX_DISABLE_INDICATOR_CACHE")
    os.environ["APEX_DISABLE_INDICATOR_CACHE"] = "1"
    try:
        prepared = prepare_backtest_data(
            price_data,
            symbol_universe=[symbol],
            start_date=None,
            global_data=global_data,
        )
    finally:
        if prev_disable_cache is None:
            os.environ.pop("APEX_DISABLE_INDICATOR_CACHE", None)
        else:
            os.environ["APEX_DISABLE_INDICATOR_CACHE"] = prev_disable_cache
    if symbol not in prepared.enriched:
        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "entry_detected": False,
            "profit_factor": 0.0,
            "total_trades": 0,
            "final_value": START_CASH,
            "miss_report": {"summary": "No enriched data prepared for symbol."},
        }

    cfg = _compose_strategy_config(
        technical_weight,
        fundamental_weight,
        strategy_override=strategy_override,
    )
    cfg["log_scoring"] = True
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
    total_trades = int(result.get("total_trades", 0) or 0)
    final_value = float(result.get("final_value", START_CASH) or START_CASH)
    entry_detected = (total_trades > 0) or (abs(final_value - START_CASH) > 1e-6)
    pf = _profit_factor(trades)
    sharpe = _sharpe_from_equity_curve(result.get("equity_curve") or [])

    out: Dict[str, Any] = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "entry_detected": bool(entry_detected),
        "profit_factor": float(pf),
        "total_trades": total_trades,
        "final_value": final_value,
        "sharpe": sharpe,
        "miss_report": None,
    }
    if not entry_detected:
        out["miss_report"] = _analyze_miss_report(
            prepared.enriched[symbol].df,
            cfg,
            start_date,
            end_date,
            spy_df,
        )
    return out


def _run_known_winner_suite(
    *,
    technical_weight: float,
    fundamental_weight: float,
    symbols_filter: set[str] | None = None,
    strategy_override: Dict[str, Any] | None = None,
    cache_only: bool = False,
) -> List[Dict[str, Any]]:
    print(
        f"\n[Task 3.1] Known Winner Diagnostic "
        f"(weights: tech={technical_weight:.2f}, fund={fundamental_weight:.2f}, cache_only={bool(cache_only)})"
    )
    results: List[Dict[str, Any]] = []
    for symbol, start_date, end_date in KNOWN_WINNERS:
        if symbols_filter and symbol.upper() not in symbols_filter:
            continue
        res = _run_single_symbol_diagnostic(
            symbol,
            start_date,
            end_date,
            technical_weight=technical_weight,
            fundamental_weight=fundamental_weight,
            strategy_override=strategy_override,
            cache_only=cache_only,
        )
        results.append(res)
        print(
            f"[Task 3.1] {symbol} | Trades={res['total_trades']} "
            f"| EntryDetected={res['entry_detected']} | PF={res['profit_factor']:.3f}"
        )
        miss = res.get("miss_report")
        if isinstance(miss, dict):
            print(f"[Task 3.1] MISS REPORT {symbol}: {miss.get('summary')}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 validation runner")
    parser.add_argument(
        "--task",
        choices=["all", "known-winners", "optimize", "full"],
        default=str(os.getenv("PHASE3_TASK", "all") or "all").strip().lower(),
    )
    parser.add_argument(
        "--symbols",
        default=str(os.getenv("PHASE3_ONLY_SYMBOLS", "") or "").strip(),
        help="Comma-separated symbol filter for known-winner diagnostic.",
    )
    parser.add_argument(
        "--tech-weight",
        type=float,
        default=float(os.getenv("PHASE3_TECH_WEIGHT", "0.60") or 0.60),
    )
    parser.add_argument(
        "--fund-weight",
        type=float,
        default=float(os.getenv("PHASE3_FUND_WEIGHT", "0.40") or 0.40),
    )
    parser.add_argument(
        "--strategy-config",
        default=str(os.getenv("PHASE3_STRATEGY_CONFIG", "") or "").strip(),
        help="Optional JSON file with strategy parameters to merge into validation runs.",
    )
    parser.add_argument(
        "--use-tuned-config",
        action="store_true",
        help="Use config/superperformance_winner.json as strategy override.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=str(os.getenv("PHASE3_CACHE_ONLY", "0") or "0").strip().lower() in {"1", "true", "yes"},
        help="Use cache-only data loading (no forced API refresh).",
    )
    return parser.parse_args()


def _resolve_full_universe() -> Tuple[str, List[str]]:
    mode = str(os.getenv("PHASE3_FULL_UNIVERSE_MODE", "RUSSELL3000") or "RUSSELL3000").upper()
    if mode == "RUSSELL3000":
        symbols = get_universe_symbols("RUSSELL3000") or []
        return mode, symbols
    sp500 = get_universe_symbols("SP500") or []
    if len(sp500) > 500:
        sp500 = sp500[:500]
    return "LIQUID500", sp500


def _run_full_universe_validation(
    *,
    technical_weight: float,
    fundamental_weight: float,
    strategy_override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    universe_mode, symbols = _resolve_full_universe()
    if not symbols:
        raise RuntimeError("Full-universe validation could not load symbols.")

    print(f"\n[Task 3.3] Full Validation Universe: {universe_mode} ({len(symbols)} symbols)")
    days = _days_for_window(FULL_START_DATE, FULL_END_DATE, warmup_days=380)

    data = _fetch_data_pack_with_retry(
        symbols,
        days=days,
        label="full-universe",
        max_attempts=3,
        min_coverage=0.50,
        backtest_mode=False,
    )
    global_raw = _fetch_data_pack_with_retry(
        ["SPY", "$VIX", "VIX"],
        days=days,
        label="full-universe-global",
        max_attempts=3,
        min_coverage=0.66,
        backtest_mode=False,
    )
    spy_df = global_raw.get("SPY")
    vix_df = global_raw.get("$VIX") if global_raw.get("$VIX") is not None else global_raw.get("VIX")
    global_data = {"SPY": spy_df, "VIX": vix_df}

    prepared = prepare_backtest_data(
        data,
        symbol_universe=symbols,
        start_date=None,
        global_data=global_data,
    )
    if not prepared.enriched:
        raise RuntimeError("Prepared dataset is empty for full-universe validation.")

    cfg = _compose_strategy_config(
        technical_weight,
        fundamental_weight,
        strategy_override=strategy_override,
    )
    cfg["log_scoring"] = False
    strategy = SuperperformanceStrategy(copy.deepcopy(cfg))
    result = _safe_result_obj(
        run_backtest(
            [strategy],
            prepared,
            start_cash=START_CASH,
            start_date=FULL_START_DATE,
            end_date=FULL_END_DATE,
            global_data=global_data,
        )
    )

    trades = result.get("trades_list") or []
    os.makedirs(os.path.dirname(FULL_EXPORT_PATH), exist_ok=True)
    pd.DataFrame(trades).to_csv(FULL_EXPORT_PATH, index=False)

    cagr = float(result.get("cagr", 0.0) or 0.0) * 100.0
    max_dd = float(result.get("max_drawdown_pct", 0.0) or 0.0)
    if max_dd <= 1.0:
        max_dd *= 100.0
    win_rate = float(result.get("hit_rate", 0.0) or 0.0)
    pf = _profit_factor(trades)
    avg_win, avg_loss = _average_winner_loser(trades)

    summary = {
        "universe_mode": universe_mode,
        "symbols_prepared": len(prepared.enriched),
        "final_value": float(result.get("final_value", START_CASH) or START_CASH),
        "cagr_pct": cagr,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "avg_winner": avg_win,
        "avg_loser": avg_loss,
        "total_trades": int(result.get("total_trades", 0) or 0),
        "export_path": FULL_EXPORT_PATH,
    }
    print(
        "[Task 3.3] "
        f"CAGR={summary['cagr_pct']:.2f}% | MaxDD={summary['max_drawdown_pct']:.2f}% | "
        f"WinRate={summary['win_rate_pct']:.2f}% | AvgWin=${summary['avg_winner']:.2f} | "
        f"AvgLoss=${summary['avg_loser']:.2f}"
    )
    print(f"[Task 3.3] Saved transaction log to {FULL_EXPORT_PATH}")
    return summary


def main() -> None:
    args = _parse_args()
    symbols_filter = {s.strip().upper() for s in str(args.symbols).split(",") if s.strip()}
    task = str(args.task or "all").strip().lower()
    strategy_override: Dict[str, Any] = {}
    if bool(args.use_tuned_config):
        strategy_override = _load_strategy_override(DEFAULT_TUNED_CONFIG_PATH)
    elif str(args.strategy_config or "").strip():
        strategy_override = _load_strategy_override(str(args.strategy_config).strip())

    print("PHASE 3: Autonomous Tuning & Validation")

    # Task 3.1
    if task in {"all", "known-winners"}:
        winner_results = _run_known_winner_suite(
            technical_weight=float(args.tech_weight),
            fundamental_weight=float(args.fund_weight),
            symbols_filter=symbols_filter or None,
            strategy_override=strategy_override or None,
            cache_only=bool(args.cache_only),
        )
        winner_entries_ok = all(bool(r.get("entry_detected")) for r in winner_results) if winner_results else False
        winner_pf_ok = all(
            (float(r.get("profit_factor", 0.0) or 0.0) > 2.0)
            for r in winner_results
            if bool(r.get("entry_detected"))
        )
        print(
            f"[Task 3.1] Success Criteria | EntryTriggeredAll={winner_entries_ok} | "
            f"ProfitFactor>2All={winner_pf_ok}"
        )
        if task == "known-winners":
            return

    # Task 3.2
    tech_w = float(args.tech_weight)
    fund_w = float(args.fund_weight)
    if task in {"all", "optimize"}:
        print("\n[Task 3.2] Parameter Optimization (Grid Search)")
        grid = run_grid_search()
        best = grid.get("best") or {}
        tech_w = float(best.get("TechnicalWeight", tech_w) or tech_w)
        fund_w = float(best.get("FundamentalWeight", fund_w) or fund_w)
        print(
            f"[Task 3.2] Optimal Weight Configuration: "
            f"technical={tech_w:.2f}, fundamental={fund_w:.2f}, sharpe={float(best.get('Sharpe', 0.0) or 0.0):.3f}"
        )
        if task == "optimize":
            return

    # Task 3.3
    if task in {"all", "full"}:
        full_summary = _run_full_universe_validation(
            technical_weight=tech_w,
            fundamental_weight=fund_w,
            strategy_override=strategy_override or None,
        )
        print(
            "\n[Task 3.3] Final Performance Summary | "
            f"Universe={full_summary['universe_mode']} | "
            f"CAGR={full_summary['cagr_pct']:.2f}% | "
            f"MaxDD={full_summary['max_drawdown_pct']:.2f}% | "
            f"WinRate={full_summary['win_rate_pct']:.2f}%"
        )


if __name__ == "__main__":
    main()
