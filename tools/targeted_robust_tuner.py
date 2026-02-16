#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures
import copy
import json
import multiprocessing
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.loader import fetch_data_pack
from data.universe import get_universe_symbols
from execution.engine import run_backtest
from strategies.superperformance import SuperperformanceStrategy


CONFIG_GENERATED = ROOT / "config" / "generated_strategies.json"
CONFIG_WINNER = ROOT / "config" / "superperformance_winner.json"
CACHE_PATH = ROOT / "data" / "cache_indicators.pkl"
LOG_DIR = ROOT / "logs"

FULL_START = str(os.getenv("APEX_TUNER_FULL_START_DATE", "2006-02-12") or "2006-02-12")
FULL_END = str(os.getenv("APEX_TUNER_FULL_END_DATE", "2026-02-13") or "2026-02-13")
RECENT_START = str(os.getenv("APEX_TUNER_RECENT_START_DATE", "2022-01-01") or "2022-01-01")
RECENT_END = str(os.getenv("APEX_TUNER_RECENT_END_DATE", "2026-02-13") or "2026-02-13")
TX_COST_BPS = float(os.getenv("APEX_TUNER_COST_BPS", "50") or "50")
MAX_WORKERS = int(os.getenv("APEX_TUNER_WORKERS", "2") or "2")
SHORTLIST_NON_BASELINE = int(os.getenv("APEX_TUNER_SHORTLIST", "2") or "2")


_WORKER_PREPARED = None
_WORKER_GLOBAL = None


@dataclass
class EvalResult:
    name: str
    window: str
    start_date: str
    end_date: str
    cost_bps: float
    cagr: float
    dd: float
    pf: float
    trades: int
    elapsed_sec: float
    error: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "window": self.window,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "cost_bps": self.cost_bps,
            "cagr": self.cagr,
            "dd": self.dd,
            "pf": self.pf,
            "trades": self.trades,
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
        }


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


def _cagr(final_value: float, start_date: str, end_date: str, start_cash: float = 100000.0) -> float:
    years = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    years = max(years, 1.0)
    if final_value <= 0:
        return -100.0
    return ((final_value / start_cash) ** (1.0 / years) - 1.0) * 100.0


def _load_generated_superperformance() -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    data = json.loads(CONFIG_GENERATED.read_text())
    if not isinstance(data, list):
        raise RuntimeError("config/generated_strategies.json must contain a list.")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() == "superperformance" or str(item.get("name", "")).lower() == "superperformance":
            return dict(item), data, idx
    raise RuntimeError("No Superperformance entry found in config/generated_strategies.json.")


def _compress_cached_prepared(prepared_obj: Any) -> Any:
    # Keep memory bounded for multiprocess evaluation.
    for _, s_data in getattr(prepared_obj, "enriched", {}).items():
        try:
            cols = s_data.df.select_dtypes(include=["float64"]).columns
            if len(cols) > 0:
                s_data.df[cols] = s_data.df[cols].astype("float32")
        except Exception:
            pass
        try:
            for attr in getattr(type(s_data), "__slots__", ()):
                if attr == "df":
                    continue
                val = getattr(s_data, attr)
                if isinstance(val, np.ndarray) and val.dtype == np.float64:
                    setattr(s_data, attr, val.astype(np.float32, copy=False))
                elif isinstance(val, np.ndarray) and val.dtype == np.int64 and attr == "gidx":
                    setattr(s_data, attr, val.astype(np.int32, copy=False))
        except Exception:
            pass
    return prepared_obj


def _load_prepared_r3000() -> Any:
    if not CACHE_PATH.exists():
        raise RuntimeError(f"Cache file missing: {CACHE_PATH}")
    with CACHE_PATH.open("rb") as f:
        cached = pickle.load(f)
    if not hasattr(cached, "enriched") or not hasattr(cached, "all_dates"):
        raise RuntimeError("Unexpected cache format in data/cache_indicators.pkl")
    requested = [str(s).upper() for s in (get_universe_symbols("RUSSELL3000") or [])]
    cached_lookup = {str(sym).upper(): sym for sym in cached.enriched.keys()}
    selected = [cached_lookup[s] for s in requested if s in cached_lookup]
    filtered = {sym: cached.enriched[sym] for sym in selected if sym in cached.enriched}
    prepared = type(cached)(enriched=filtered, all_dates=getattr(cached, "all_dates"))
    return _compress_cached_prepared(prepared)


def _build_candidate_configs(base_cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    base = copy.deepcopy(base_cfg)
    base.pop("type", None)
    base.pop("name", None)
    base["entry_mode"] = "both"

    cands: Dict[str, Dict[str, Any]] = {"baseline": copy.deepcopy(base)}

    # Hybrid 1: robust-leverage blend between active baseline and recent optimizer candidate.
    h1 = copy.deepcopy(base)
    h1.update(
        {
            "rs_gate_min": 88,
            "min_avg_volume_30": 100000,
            "fundamental_growth_min_pct": 15,
            "high_tight_flag_override_pct": 90,
            "min_entry_score": 40.0,
            "max_stop_pct": 0.08,
            "stop_loss_atr_bull": 4,
            "pyramid_max_adds": 2,
            "max_positions": 8,
            "risk_per_trade": 0.02,
            "max_pos_size_pct": 0.15,
            "max_total_exposure_pct_bull": 1.4,
            "max_total_exposure_pct_bear": 0.1,
        }
    )
    cands["hybrid_balance"] = h1

    # Hybrid 2: keep baseline breadth, tighten quality and stop logic.
    h2 = copy.deepcopy(base)
    h2.update(
        {
            "rs_gate_min": 88,
            "vcp_lookback_bars": 40,
            "vcp_extrema_order": 3,
            "ep_close_near_high_min": 0.75,
            "min_entry_score": 45.0,
            "max_stop_pct": 0.08,
            "stop_loss_atr_bull": 3,
            "max_positions": 8,
            "risk_per_trade": 0.02,
            "max_pos_size_pct": 0.15,
            "max_total_exposure_pct_bull": 1.3,
            "max_total_exposure_pct_bear": 0.1,
        }
    )
    cands["hybrid_quality"] = h2

    # Hybrid 3: balanced gates with moderate deployment.
    h3 = copy.deepcopy(base)
    h3.update(
        {
            "rs_gate_min": 85,
            "fundamental_growth_min_pct": 15,
            "high_tight_flag_override_pct": 90,
            "min_entry_score": 40.0,
            "stop_loss_atr_bull": 4,
            "max_stop_pct": 0.08,
            "max_positions": 10,
            "risk_per_trade": 0.02,
            "max_pos_size_pct": 0.14,
            "max_total_exposure_pct_bull": 1.4,
            "max_total_exposure_pct_bear": 0.1,
        }
    )
    cands["hybrid_deploy"] = h3
    return cands


def _init_worker(prepared: Any, global_data: Dict[str, Any]) -> None:
    global _WORKER_PREPARED, _WORKER_GLOBAL
    _WORKER_PREPARED = prepared
    _WORKER_GLOBAL = global_data


def _evaluate_config(
    *,
    name: str,
    cfg: Dict[str, Any],
    window: str,
    start_date: str,
    end_date: str,
    cost_bps: float,
    prepared: Any,
    global_data: Dict[str, Any],
) -> EvalResult:
    t0 = time.time()
    try:
        local_cfg = copy.deepcopy(cfg)
        local_cfg["name"] = f"tuner_{name}_{window}"
        local_cfg["entry_mode"] = "both"
        local_cfg["transaction_cost_bps"] = float(max(0.0, cost_bps))
        strat = SuperperformanceStrategy(local_cfg)
        result = run_backtest(
            [strat],
            prepared,
            start_cash=100000.0,
            start_date=start_date,
            end_date=end_date,
            global_data=global_data,
        )
        metrics = result[0] if isinstance(result, list) else result
        final_value = float(metrics.get("final_value", 100000.0) or 100000.0)
        cagr = _cagr(final_value, start_date, end_date)
        raw_dd = float(metrics.get("max_drawdown_pct", metrics.get("max_drawdown", metrics.get("drawdown", 0.0))) or 0.0)
        dd = raw_dd * 100.0 if raw_dd < 1.0 else raw_dd
        trades = int(metrics.get("total_trades", 0) or 0)
        pf = float(_profit_factor(metrics.get("trades_list", []) or []))
        return EvalResult(
            name=name,
            window=window,
            start_date=start_date,
            end_date=end_date,
            cost_bps=float(cost_bps),
            cagr=float(cagr),
            dd=float(dd),
            pf=float(pf),
            trades=trades,
            elapsed_sec=float(time.time() - t0),
            error="",
        )
    except Exception as exc:
        return EvalResult(
            name=name,
            window=window,
            start_date=start_date,
            end_date=end_date,
            cost_bps=float(cost_bps),
            cagr=-999.0,
            dd=999.0,
            pf=0.0,
            trades=0,
            elapsed_sec=float(time.time() - t0),
            error=str(exc),
        )


def _run_task(task: Tuple[str, Dict[str, Any], str, str, str, float]) -> EvalResult:
    name, cfg, window, start_date, end_date, cost_bps = task
    return _evaluate_config(
        name=name,
        cfg=cfg,
        window=window,
        start_date=start_date,
        end_date=end_date,
        cost_bps=cost_bps,
        prepared=_WORKER_PREPARED,
        global_data=_WORKER_GLOBAL,
    )


def _stage_score(res: EvalResult) -> float:
    score = float(res.cagr)
    score += min(max(res.pf, 0.0), 6.0) * 2.5
    score -= max(0.0, res.dd - 30.0) * 2.0
    if res.trades < 80:
        score -= (80 - res.trades) * 0.04
    return score


def _final_score(full_res: EvalResult, recent_res: EvalResult) -> float:
    score = (full_res.cagr * 0.65) + (recent_res.cagr * 0.35)
    score += min(max(full_res.pf, 0.0), 6.0) * 2.0
    score += min(max(recent_res.pf, 0.0), 6.0) * 1.0
    score -= max(0.0, full_res.dd - 30.0) * 1.8
    score -= max(0.0, recent_res.dd - 20.0) * 0.8
    return score


def _promotable(best: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    if best["name"] == "baseline":
        return False
    full_best: EvalResult = best["full"]
    full_base: EvalResult = baseline["full"]
    recent_best: EvalResult = best["recent"]
    recent_base: EvalResult = baseline["recent"]
    if full_best.cagr < (full_base.cagr + 0.5):
        return False
    if recent_best.cagr < (recent_base.cagr - 1.5):
        return False
    if full_best.dd > (full_base.dd + 2.5):
        return False
    if full_best.pf < (full_base.pf - 0.1):
        return False
    return True


def _write_strategy_promotion(configs: Dict[str, Dict[str, Any]], winner_name: str) -> None:
    winner_cfg = copy.deepcopy(configs[winner_name])
    existing, payload, idx = _load_generated_superperformance()
    promoted_entry = copy.deepcopy(winner_cfg)
    promoted_entry["name"] = existing.get("name", "Superperformance")
    promoted_entry["type"] = existing.get("type", "superperformance")
    payload[idx] = promoted_entry
    CONFIG_GENERATED.write_text(json.dumps(payload, indent=4))
    winner_cfg.pop("name", None)
    winner_cfg.pop("type", None)
    CONFIG_WINNER.write_text(json.dumps(winner_cfg, indent=4))


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOG_DIR / f"targeted_robust_tuner_{ts}.json"
    latest_path = LOG_DIR / "targeted_robust_tuner_latest.json"

    print("🔧 Targeted Robust Tuner (local search around active baseline)")
    print(f"Window full: {FULL_START} -> {FULL_END} | recent: {RECENT_START} -> {RECENT_END} | cost={TX_COST_BPS:.1f}bps")

    base_entry, _, _ = _load_generated_superperformance()
    candidate_cfgs = _build_candidate_configs(base_entry)
    print(f"Candidates: {list(candidate_cfgs.keys())}")

    prepared = _load_prepared_r3000()
    print(f"Prepared symbols: {len(getattr(prepared, 'enriched', {}))}")

    days = int((pd.Timestamp(FULL_END) - pd.Timestamp(FULL_START)).days + 400)
    global_data = fetch_data_pack(["SPY", "VIX"], days=days, backtest_mode=True) or {}

    try:
        multiprocessing.set_start_method("fork" if sys.platform == "darwin" else "spawn", force=True)
    except Exception:
        pass
    mp_ctx = multiprocessing.get_context("fork" if sys.platform == "darwin" else "spawn")

    # Stage 1: 20Y robustness screen at target friction.
    stage1_tasks: List[Tuple[str, Dict[str, Any], str, str, str, float]] = []
    for name, cfg in candidate_cfgs.items():
        stage1_tasks.append((name, cfg, "full", FULL_START, FULL_END, TX_COST_BPS))

    stage1: Dict[str, EvalResult] = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, MAX_WORKERS),
        mp_context=mp_ctx,
        initializer=_init_worker,
        initargs=(prepared, global_data),
    ) as ex:
        futures = [ex.submit(_run_task, t) for t in stage1_tasks]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            stage1[res.name] = res
            if res.error:
                print(f"[Stage1] {res.name}: ERROR {res.error}")
            else:
                print(
                    f"[Stage1] {res.name}: CAGR={res.cagr:.2f}% DD={res.dd:.2f}% "
                    f"PF={res.pf:.2f} trades={res.trades} t={res.elapsed_sec:.1f}s"
                )

    ranked_stage1 = sorted(
        [r for r in stage1.values() if not r.error],
        key=lambda r: _stage_score(r),
        reverse=True,
    )
    shortlist = ["baseline"]
    for r in ranked_stage1:
        if r.name == "baseline":
            continue
        shortlist.append(r.name)
        if len(shortlist) >= (1 + max(1, SHORTLIST_NON_BASELINE)):
            break
    shortlist = list(dict.fromkeys(shortlist))
    print(f"Shortlist for recent validation: {shortlist}")

    # Stage 2: recent robustness retention.
    stage2_tasks: List[Tuple[str, Dict[str, Any], str, str, str, float]] = []
    for name in shortlist:
        stage2_tasks.append((name, candidate_cfgs[name], "recent", RECENT_START, RECENT_END, TX_COST_BPS))

    stage2: Dict[str, EvalResult] = {}
    for task in stage2_tasks:
        name, cfg, window, start_date, end_date, cost_bps = task
        res = _evaluate_config(
            name=name,
            cfg=cfg,
            window=window,
            start_date=start_date,
            end_date=end_date,
            cost_bps=cost_bps,
            prepared=prepared,
            global_data=global_data,
        )
        stage2[res.name] = res
        if res.error:
            print(f"[Stage2] {res.name}: ERROR {res.error}")
        else:
            print(
                f"[Stage2] {res.name}: CAGR={res.cagr:.2f}% DD={res.dd:.2f}% "
                f"PF={res.pf:.2f} trades={res.trades} t={res.elapsed_sec:.1f}s"
            )

    merged: List[Dict[str, Any]] = []
    for name in shortlist:
        f = stage1.get(name)
        r = stage2.get(name)
        if f is None or r is None or f.error or r.error:
            continue
        merged.append(
            {
                "name": name,
                "full": f,
                "recent": r,
                "score": _final_score(f, r),
            }
        )
    merged.sort(key=lambda x: x["score"], reverse=True)
    if not merged:
        raise RuntimeError("No valid merged results from targeted tuner.")

    baseline_row = next((x for x in merged if x["name"] == "baseline"), None)
    if baseline_row is None:
        raise RuntimeError("Baseline result missing; cannot decide promotion safely.")

    best_row = merged[0]
    promote = _promotable(best_row, baseline_row)

    if promote:
        _write_strategy_promotion(candidate_cfgs, best_row["name"])
        decision = f"PROMOTED {best_row['name']}"
    else:
        decision = f"KEPT baseline (best={best_row['name']})"

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_generated": str(CONFIG_GENERATED),
        "config_winner": str(CONFIG_WINNER),
        "decision": decision,
        "cost_bps": TX_COST_BPS,
        "full_window": {"start": FULL_START, "end": FULL_END},
        "recent_window": {"start": RECENT_START, "end": RECENT_END},
        "shortlist": shortlist,
        "stage1": {k: v.as_dict() for k, v in stage1.items()},
        "stage2": {k: v.as_dict() for k, v in stage2.items()},
        "merged": [
            {
                "name": row["name"],
                "score": row["score"],
                "full": row["full"].as_dict(),
                "recent": row["recent"].as_dict(),
            }
            for row in merged
        ],
        "promoted": bool(promote),
    }
    report_path.write_text(json.dumps(report, indent=2))
    latest_path.write_text(json.dumps(report, indent=2))

    print(f"Decision: {decision}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
