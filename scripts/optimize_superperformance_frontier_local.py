#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.loader import fetch_data_pack
from data.universe import build_russell3000_membership_by_day
from execution.engine import run_backtest
from strategies.superperformance import SuperperformanceStrategy
import optimize_superperformance as opt


RESULTS_DIR = ROOT / "logs"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_JSONL = RESULTS_DIR / f"superperformance_frontier_{RUN_TAG}.jsonl"

BASE_CONFIG_PATH = Path(
    os.getenv(
        "SUPERPERF_FRONTIER_BASE_CONFIG",
        str(ROOT / "config" / "superperformance_alpha_b4.json"),
    )
)
BEST_CONFIG_PATH = Path(
    os.getenv(
        "SUPERPERF_FRONTIER_BEST_CONFIG",
        str(ROOT / "config" / "superperformance_frontier_candidate.json"),
    )
)

START_10Y = "2016-02-26"
START_5Y = "2021-02-26"
START_3Y = "2023-02-26"
END_DATE = "2026-02-26"

ITERATIONS = int(os.getenv("SUPERPERF_FRONTIER_ITERATIONS", "28") or "28")
SEED = int(os.getenv("SUPERPERF_FRONTIER_SEED", "42") or "42")
TRADING_DAYS = int(os.getenv("SUPERPERF_FRONTIER_TRADING_DAYS", "3000") or "3000")

GATE_3Y_CAGR = float(os.getenv("SUPERPERF_FRONTIER_GATE_3Y_CAGR", "18.0") or "18.0")
GATE_3Y_DD = float(os.getenv("SUPERPERF_FRONTIER_GATE_3Y_DD", "45.0") or "45.0")
GATE_3Y_TRADES = int(os.getenv("SUPERPERF_FRONTIER_GATE_3Y_TRADES", "35") or "35")

GATE_5Y_CAGR = float(os.getenv("SUPERPERF_FRONTIER_GATE_5Y_CAGR", "23.0") or "23.0")
GATE_5Y_DD = float(os.getenv("SUPERPERF_FRONTIER_GATE_5Y_DD", "45.0") or "45.0")
GATE_5Y_TRADES = int(os.getenv("SUPERPERF_FRONTIER_GATE_5Y_TRADES", "70") or "70")

MIN_FLOOR_5Y_CAGR = float(os.getenv("SUPERPERF_FRONTIER_MIN_FLOOR_5Y_CAGR", "30.0") or "30.0")
MIN_FLOOR_10Y_CAGR = float(os.getenv("SUPERPERF_FRONTIER_MIN_FLOOR_10Y_CAGR", "0.0") or "0.0")

MUTATION_SPACE: Dict[str, List[Any]] = {
    "max_positions": [2, 3, 4],
    "risk_per_trade": [0.06, 0.08, 0.10, 0.12],
    "max_pos_size_pct": [0.25, 0.3333, 0.50],
    "max_total_exposure_pct_bull": [0.85, 0.95, 1.0],
    "max_total_exposure_pct_bear": [0.0, 0.35, 0.7, 1.0],
    "market_exposure_mode": ["exposure", "scaled", "hybrid"],
    "use_market_regime_traffic_light": [False, True],
    "bear_cash_mode": ["off", "hard"],
    "bear_max_positions": [0, 1, 2],
    "yellow_risk_scalar": [0.6, 0.8, 1.0],
    "orange_risk_scalar": [0.25, 0.45, 0.7],
    "yellow_max_positions": [2, 3, 4],
    "orange_max_positions": [1, 2, 3],
    "rs_gate_min": [84, 85, 87, 90],
    "min_entry_score": [20, 25, 30, 35],
    "vcp_lookback_bars": [25, 30, 40],
    "vcp_breakout_volume_mult": [1.5, 1.75, 2.0],
    "breakout_buffer": [0.0005, 0.001, 0.002],
    "max_stop_pct": [0.05, 0.06, 0.07],
    "stop_limit_pct": [0.02, 0.03, 0.05],
    "stop_loss_atr_bull": [3.0, 3.5, 4.0],
    "stop_loss_atr_bear": [0.5, 0.75, 1.0],
    "time_stop_days": [5, 7, 10, 12],
    "time_stop_profit_pct": [0.0, 0.002, 0.005, 0.01],
    "ep_gap_pct": [6, 8, 10],
    "ep_close_near_high_min": [0.55, 0.65, 0.75],
    "ep_max_stop_pct": [0.10, 0.12, 0.15],
    "pyramid_threshold": [0.06, 0.08, 0.10],
    "pyramid_fraction": [0.33, 0.50, 0.67],
    "pyramid_max_adds": [1, 2, 3],
    "vcp_gap_chase_max_pct": [0.0, 0.005, 0.01],
    "vcp_gap_chase_rs_min": [93, 95, 97],
    "vcp_gap_chase_score_min": [65, 70, 75],
}


@dataclass
class TrialResult:
    idx: int
    score: float
    cagr3: float
    dd3: float
    trades3: int
    cagr5: float
    dd5: float
    trades5: int
    cagr10: float
    dd10: float
    trades10: int
    eval5: bool
    eval10: bool
    config: Dict[str, Any]


def _pct_dd(value: Any) -> float:
    try:
        dd = float(value or 0.0)
    except Exception:
        return 0.0
    if dd <= 1.0:
        dd *= 100.0
    return dd


def _signature(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _evaluate(
    cfg: Dict[str, Any],
    prepared: Any,
    membership: List[set],
    global_data: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    t0 = time.time()
    print(
        f"  ▶ eval start window={start_date}..{end_date} "
        f"name={str(cfg.get('name', ''))[:36]}",
        flush=True,
    )
    strat = SuperperformanceStrategy(cfg)
    result = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership,
    )
    out = result[0] if isinstance(result, list) else result
    cagr = float(out.get("cagr", 0.0) or 0.0) * 100.0
    dd = _pct_dd(out.get("max_drawdown_pct", 0.0))
    trades = int(out.get("total_trades", 0) or 0)
    print(
        f"  ◀ eval done window={start_date}..{end_date} "
        f"cagr={cagr:.2f}% dd={dd:.2f}% trades={trades} "
        f"elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return out


def _normalize(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["name"] = str(out.get("name", "Superperformance Frontier Candidate") or "Superperformance Frontier Candidate")
    out["type"] = "superperformance"
    out["signal_mode"] = "after_close"
    out["entry_day_stop_mode"] = "close_confirmed"
    out["enforce_next_day_exit_execution"] = True
    out["allow_margin"] = False

    max_positions = max(1, min(int(out.get("max_positions", 3) or 3), 8))
    out["max_positions"] = max_positions

    max_expo_bull = float(out.get("max_total_exposure_pct_bull", 1.0) or 1.0)
    max_expo_bull = max(0.5, min(max_expo_bull, 1.0))
    out["max_total_exposure_pct_bull"] = max_expo_bull

    cap_by_slots = max_expo_bull / float(max_positions)
    max_pos_pct = float(out.get("max_pos_size_pct", cap_by_slots) or cap_by_slots)
    max_pos_pct = max(0.05, min(max_pos_pct, 0.50, cap_by_slots))
    out["max_pos_size_pct"] = max_pos_pct
    out["vcp_max_pos_size_pct"] = min(float(out.get("vcp_max_pos_size_pct", max_pos_pct) or max_pos_pct), max_pos_pct)
    out["ep_max_pos_size_pct"] = min(float(out.get("ep_max_pos_size_pct", max_pos_pct) or max_pos_pct), max_pos_pct)

    out["risk_per_trade"] = max(0.002, min(float(out.get("risk_per_trade", 0.10) or 0.10), 0.15))
    out["max_total_exposure_pct_bear"] = max(0.0, min(float(out.get("max_total_exposure_pct_bear", 1.0) or 1.0), 1.0))

    mode = str(out.get("market_exposure_mode", "exposure") or "exposure").lower()
    if mode not in {"exposure", "scaled", "hybrid", "filter"}:
        mode = "exposure"
    out["market_exposure_mode"] = mode

    tl = bool(out.get("use_market_regime_traffic_light", False))
    out["use_market_regime_traffic_light"] = tl
    out["traffic_light_block_exposure_mode"] = bool(out.get("traffic_light_block_exposure_mode", tl))

    bear_cash = str(out.get("bear_cash_mode", "off") or "off").lower()
    if bear_cash not in {"off", "hard", "cash", "all_cash"}:
        bear_cash = "off"
    out["bear_cash_mode"] = bear_cash
    if bear_cash in {"hard", "cash", "all_cash"}:
        out["max_total_exposure_pct_bear"] = 0.0
        out["bear_max_positions"] = 0
    else:
        out["bear_max_positions"] = max(0, min(int(out.get("bear_max_positions", 1) or 1), max_positions))

    out["yellow_risk_scalar"] = max(0.1, min(float(out.get("yellow_risk_scalar", 0.8) or 0.8), 1.0))
    out["orange_risk_scalar"] = max(0.05, min(float(out.get("orange_risk_scalar", 0.45) or 0.45), 1.0))
    out["yellow_max_positions"] = max(1, min(int(out.get("yellow_max_positions", max_positions) or max_positions), max_positions))
    out["orange_max_positions"] = max(1, min(int(out.get("orange_max_positions", out["yellow_max_positions"]) or out["yellow_max_positions"]), out["yellow_max_positions"]))

    out["stop_limit_pct"] = max(0.01, min(float(out.get("stop_limit_pct", 0.03) or 0.03), 0.10))
    out["max_stop_pct"] = max(0.03, min(float(out.get("max_stop_pct", 0.06) or 0.06), 0.12))
    out["ep_max_stop_pct"] = max(0.05, min(float(out.get("ep_max_stop_pct", 0.12) or 0.12), 0.20))
    out["stop_loss_atr_bull"] = max(1.5, min(float(out.get("stop_loss_atr_bull", 3.5) or 3.5), 6.0))
    out["stop_loss_atr_bear"] = max(0.3, min(float(out.get("stop_loss_atr_bear", 0.75) or 0.75), 2.0))
    out["time_stop_days"] = max(0, min(int(out.get("time_stop_days", 7) or 7), 40))
    out["time_stop_profit_pct"] = max(0.0, min(float(out.get("time_stop_profit_pct", 0.01) or 0.01), 0.05))
    out["pyramid_max_adds"] = max(0, min(int(out.get("pyramid_max_adds", 2) or 2), 4))

    out["trend_template_mode"] = str(out.get("trend_template_mode", "classic") or "classic").lower()
    if out["trend_template_mode"] not in {"classic", "strict"}:
        out["trend_template_mode"] = "classic"
    return out


def _mutate(parent: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    parent_sig = _signature(parent)
    for _ in range(16):
        child = dict(parent)
        n_mut = rng.randint(3, 9)
        keys = rng.sample(list(MUTATION_SPACE.keys()), n_mut)
        for key in keys:
            options = MUTATION_SPACE[key]
            if not options:
                continue
            cur = child.get(key)
            choices = [v for v in options if v != cur]
            child[key] = rng.choice(choices if choices else options)
        child = _normalize(child)
        if _signature(child) != parent_sig:
            return child
    return _normalize(dict(parent))


def _score(obj3: Dict[str, Any], obj5: Dict[str, Any] | None, obj10: Dict[str, Any] | None) -> float:
    c3 = float(obj3.get("cagr", 0.0) or 0.0) * 100.0
    d3 = _pct_dd(obj3.get("max_drawdown_pct", 0.0))
    t3 = int(obj3.get("total_trades", 0) or 0)
    score = (0.5 * c3) - (0.08 * d3)
    if t3 < 30:
        score -= (30 - t3) * 0.15

    if obj5 is not None:
        c5 = float(obj5.get("cagr", 0.0) or 0.0) * 100.0
        d5 = _pct_dd(obj5.get("max_drawdown_pct", 0.0))
        t5 = int(obj5.get("total_trades", 0) or 0)
        score += (0.8 * c5) - (0.12 * d5)
        if t5 < 65:
            score -= (65 - t5) * 0.1
        if c5 < MIN_FLOOR_5Y_CAGR:
            score -= (MIN_FLOOR_5Y_CAGR - c5) * 4.0

    if obj10 is not None:
        c10 = float(obj10.get("cagr", 0.0) or 0.0) * 100.0
        d10 = _pct_dd(obj10.get("max_drawdown_pct", 0.0))
        t10 = int(obj10.get("total_trades", 0) or 0)
        audit = obj10.get("audit_report") or {}
        gross = float(audit.get("max_gross_exposure_pct", 0.0) or 0.0)
        sdo = int(audit.get("same_day_open_entries", 0) or 0)

        score += (1.4 * c10) - (0.22 * d10)
        if c10 >= 20.0:
            score += 5.0
        if c10 >= 22.5:
            score += 5.0
        if d10 <= 34.0:
            score += 3.0
        if t10 < 110:
            score -= (110 - t10) * 0.1
        if gross > 1.0:
            score -= (gross - 1.0) * 500.0
        if sdo > 0:
            score -= min(200.0, 25.0 + (sdo * 0.5))
        if MIN_FLOOR_10Y_CAGR > 0 and c10 < MIN_FLOOR_10Y_CAGR:
            score -= (MIN_FLOOR_10Y_CAGR - c10) * 6.0
    return score


def main() -> None:
    random.seed(SEED)
    rng = random.Random(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG_PATH, "r") as f:
        base_cfg = json.load(f)
    base_cfg = _normalize(base_cfg)

    with open(ROOT / "data" / "cache_indicators.pkl", "rb") as f:
        prepared = pickle.load(f)
    prepared = opt.compress_data(prepared)

    all_dates = list(getattr(prepared, "all_dates"))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=TRADING_DAYS, backtest_mode=True) or {}
    print(
        f"frontier_search run_tag={RUN_TAG} iterations={ITERATIONS} "
        f"prepared={len(prepared.enriched)} membership={membership_source}"
    )

    seen = set()
    leaderboard: List[TrialResult] = []

    def _record(t: TrialResult) -> None:
        leaderboard.append(t)
        leaderboard.sort(key=lambda x: x.score, reverse=True)
        del leaderboard[12:]

    def _pack(
        idx: int,
        score: float,
        obj3: Dict[str, Any],
        obj5: Dict[str, Any] | None,
        obj10: Dict[str, Any] | None,
        cfg: Dict[str, Any],
    ) -> TrialResult:
        return TrialResult(
            idx=idx,
            score=score,
            cagr3=float(obj3.get("cagr", 0.0) or 0.0) * 100.0,
            dd3=_pct_dd(obj3.get("max_drawdown_pct", 0.0)),
            trades3=int(obj3.get("total_trades", 0) or 0),
            cagr5=(float(obj5.get("cagr", 0.0) or 0.0) * 100.0) if obj5 else float("nan"),
            dd5=_pct_dd(obj5.get("max_drawdown_pct", 0.0)) if obj5 else float("nan"),
            trades5=int(obj5.get("total_trades", 0) or 0) if obj5 else 0,
            cagr10=(float(obj10.get("cagr", 0.0) or 0.0) * 100.0) if obj10 else float("nan"),
            dd10=_pct_dd(obj10.get("max_drawdown_pct", 0.0)) if obj10 else float("nan"),
            trades10=int(obj10.get("total_trades", 0) or 0) if obj10 else 0,
            eval5=bool(obj5 is not None),
            eval10=bool(obj10 is not None),
            config=cfg,
        )

    base_sig = _signature(base_cfg)
    seen.add(base_sig)
    obj3 = _evaluate(base_cfg, prepared, membership, global_data, START_3Y, END_DATE)
    obj5 = _evaluate(base_cfg, prepared, membership, global_data, START_5Y, END_DATE)
    obj10 = _evaluate(base_cfg, prepared, membership, global_data, START_10Y, END_DATE)
    base_trial = _pack(0, _score(obj3, obj5, obj10), obj3, obj5, obj10, base_cfg)
    _record(base_trial)
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(base_trial.__dict__) + "\n")
    print(
        f"[0] score={base_trial.score:.2f} "
        f"c3={base_trial.cagr3:.2f}% d3={base_trial.dd3:.2f}% "
        f"c5={base_trial.cagr5:.2f}% d5={base_trial.dd5:.2f}% "
        f"c10={base_trial.cagr10:.2f}% d10={base_trial.dd10:.2f}%"
    )

    for i in range(1, ITERATIONS + 1):
        parent = leaderboard[rng.randrange(max(1, min(len(leaderboard), 6)))].config
        child = _mutate(parent, rng)
        sig = _signature(child)
        if sig in seen:
            continue
        seen.add(sig)

        obj3 = _evaluate(child, prepared, membership, global_data, START_3Y, END_DATE)
        c3 = float(obj3.get("cagr", 0.0) or 0.0) * 100.0
        d3 = _pct_dd(obj3.get("max_drawdown_pct", 0.0))
        t3 = int(obj3.get("total_trades", 0) or 0)
        run5 = bool(c3 >= GATE_3Y_CAGR and d3 <= GATE_3Y_DD and t3 >= GATE_3Y_TRADES)

        obj5 = None
        obj10 = None
        if run5:
            obj5 = _evaluate(child, prepared, membership, global_data, START_5Y, END_DATE)
            c5 = float(obj5.get("cagr", 0.0) or 0.0) * 100.0
            d5 = _pct_dd(obj5.get("max_drawdown_pct", 0.0))
            t5 = int(obj5.get("total_trades", 0) or 0)
            run10 = bool(c5 >= GATE_5Y_CAGR and d5 <= GATE_5Y_DD and t5 >= GATE_5Y_TRADES)
            if run10:
                obj10 = _evaluate(child, prepared, membership, global_data, START_10Y, END_DATE)

        score = _score(obj3, obj5, obj10)
        trial = _pack(i, score, obj3, obj5, obj10, child)
        _record(trial)
        with open(RESULTS_JSONL, "a") as f:
            f.write(json.dumps(trial.__dict__) + "\n")

        best = leaderboard[0]
        print(
            f"[{i}] score={trial.score:.2f} c3={trial.cagr3:.2f}% "
            f"c5={trial.cagr5:.2f}% c10={trial.cagr10:.2f}% eval10={trial.eval10} | "
            f"best={best.score:.2f}/{best.cagr10:.2f}% dd10={best.dd10:.2f}%"
        )
        gc.collect()

    best = leaderboard[0]
    best_cfg = dict(best.config)
    best_cfg["name"] = "Superperformance Frontier Candidate"
    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump(best_cfg, f, indent=2)
        f.write("\n")
    print(
        f"BEST score={best.score:.2f} c3={best.cagr3:.2f}% c5={best.cagr5:.2f}% "
        f"c10={best.cagr10:.2f}% dd10={best.dd10:.2f}% trades10={best.trades10} "
        f"saved={BEST_CONFIG_PATH}"
    )
    print(f"TRIAL_LOG {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
