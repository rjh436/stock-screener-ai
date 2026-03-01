#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import os
import pickle
import random
import sys
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
from strategies.apex_fusion import ApexFusionStrategy
import optimize_superperformance as opt


RESULTS_DIR = ROOT / "logs"
RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_JSONL = RESULTS_DIR / f"apex_fusion_local_search_{RUN_TAG}.jsonl"

BASE_CONFIG_PATH = Path(
    os.getenv(
        "APEX_FUSION_BASE_CONFIG",
        str(ROOT / "config" / "apex_fusion_practical_v2.json"),
    )
)
BEST_CONFIG_PATH = Path(
    os.getenv(
        "APEX_FUSION_BEST_CONFIG",
        str(ROOT / "config" / "apex_fusion_practical_v2_candidate.json"),
    )
)

START_10Y = "2016-02-26"
END_10Y = "2026-02-26"
START_5Y = "2021-02-26"
END_5Y = "2026-02-26"

ITERATIONS = int(os.getenv("APEX_FUSION_ITERATIONS", "40") or "40")
SEED = int(os.getenv("APEX_FUSION_SEED", "42") or "42")
TRADING_DAYS = int(os.getenv("APEX_FUSION_TRADING_DAYS", "3000") or "3000")

GATE_5Y_CAGR = float(os.getenv("APEX_FUSION_GATE_5Y_CAGR", "12.0") or "12.0")
GATE_5Y_DD = float(os.getenv("APEX_FUSION_GATE_5Y_DD", "35.0") or "35.0")
GATE_5Y_TRADES = int(os.getenv("APEX_FUSION_GATE_5Y_TRADES", "45") or "45")

MUTATION_SPACE: Dict[str, List[Any]] = {
    "enable_ep": [False, True],
    "min_signal_strength": [88, 90, 92, 94, 96],
    "min_entry_score": [86, 88, 90, 92, 94],
    "rs_gate_min": [88, 90, 92, 94],
    "momentum_gate_min": [75, 80, 85, 88],
    "near_high_min_pct": [0.82, 0.85, 0.88, 0.90],
    "prior_runup_min_pct": [20, 25, 30, 35],
    "fundamental_growth_min_pct": [10, 15, 20, 25],
    "fundamental_override_rs_min": [94, 96, 98],
    "fundamental_override_ret3m_min": [30, 40, 50],
    "min_avg_volume_30": [100000, 150000, 200000, 300000],
    "adr_min_pct": [0.03, 0.035, 0.04, 0.05],
    "sector_rs_min": [45, 50, 55, 60],
    "max_vix": [32, 36, 40],
    "breakout_buffer": [0.001, 0.0015, 0.002],
    "setup_proximity_pct": [0.015, 0.02, 0.025, 0.03],
    "setup_high_proximity_pct": [0.008, 0.012, 0.015],
    "setup_volume_max_mult": [1.05, 1.10, 1.15, 1.25],
    "setup_clv_min": [0.50, 0.55, 0.60],
    "setup_max_base_depth_pct": [22, 26, 30, 35],
    "setup_contraction_ratio": [0.70, 0.75, 0.80, 0.85],
    "setup_vol_dryup_max": [0.85, 0.95, 1.05],
    "ep_gap_pct": [8, 10, 12],
    "ep_vol_mult": [3.0, 3.5, 4.0],
    "ep_close_near_high_min": [0.75, 0.8, 0.85],
    "ep_max_extension_above_sma20": [0.08, 0.10, 0.12],
    "ep_max_stop_pct": [0.07, 0.09, 0.11],
    "max_stop_pct": [0.05, 0.06, 0.07],
    "stop_limit_pct": [0.03, 0.04, 0.05],
    "stop_loss_atr_bull": [2.5, 3.0, 3.5],
    "trail_atr": [2.0, 2.5, 3.0],
    "trail_activation": [1.0, 1.5, 2.0],
    "time_stop_days": [8, 10, 12, 15, 20],
    "time_stop_profit_pct": [0.0, 0.002, 0.005],
    "pyramid_enabled": [False, True],
    "pyramid_threshold": [0.04, 0.06, 0.08],
    "pyramid_fraction": [0.33, 0.4, 0.5],
    "pyramid_max_adds": [1, 2, 3],
    "max_positions": [4, 5, 6],
    "risk_per_trade": [0.008, 0.01, 0.012, 0.015],
    "max_pos_size_pct": [0.15, 0.20, 0.25],
    "max_total_exposure_pct_bull": [0.8, 0.9, 1.0],
    "max_total_exposure_pct_bear": [0.0, 0.15, 0.25, 0.35],
    "market_exposure_mode": ["filter", "scaled", "hybrid"],
    "bear_cash_mode": ["hard", "off"],
    "yellow_risk_scalar": [0.6, 0.75, 0.9],
    "orange_risk_scalar": [0.25, 0.4, 0.55],
    "yellow_max_positions": [3, 4, 5],
    "orange_max_positions": [1, 2, 3],
    "bear_max_positions": [0, 1, 2],
}


@dataclass
class TrialResult:
    idx: int
    score: float
    cagr10: float
    dd10: float
    trades10: int
    cagr5: float
    dd5: float
    trades5: int
    eval10: bool
    config: Dict[str, Any]


def _pct_drawdown(metric_dd: Any) -> float:
    try:
        dd = float(metric_dd or 0.0)
    except Exception:
        return 0.0
    if dd <= 1.0:
        dd *= 100.0
    return dd


def _evaluate_period(
    cfg: Dict[str, Any],
    prepared: Any,
    membership: list[set],
    global_data: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    strat = ApexFusionStrategy(cfg)
    res = run_backtest(
        [strat],
        prepared,
        start_cash=100000.0,
        start_date=start_date,
        end_date=end_date,
        global_data=global_data,
        universe_membership_by_day=membership,
    )
    return res[0] if isinstance(res, list) else res


def _score(obj10: Dict[str, Any] | None, obj5: Dict[str, Any]) -> float:
    cagr5 = float(obj5.get("cagr", 0.0) or 0.0) * 100.0
    dd5 = _pct_drawdown(obj5.get("max_drawdown_pct", 0.0))
    trades5 = int(obj5.get("total_trades", 0) or 0)

    if obj10 is None:
        score = (0.60 * cagr5)
        if dd5 > 35.0:
            score -= (dd5 - 35.0) * 0.8
        if trades5 < 45:
            score -= (45 - trades5) * 0.20
        return score - 6.0

    cagr10 = float(obj10.get("cagr", 0.0) or 0.0) * 100.0
    dd10 = _pct_drawdown(obj10.get("max_drawdown_pct", 0.0))
    trades10 = int(obj10.get("total_trades", 0) or 0)
    audit10 = obj10.get("audit_report") or {}
    gross10 = float(audit10.get("max_gross_exposure_pct", 0.0) or 0.0)
    sdo10 = int(audit10.get("same_day_open_entries", 0) or 0)

    score = (1.10 * cagr10) + (0.45 * cagr5)
    if cagr10 >= 20.0:
        score += 4.0
    if cagr10 >= 25.0:
        score += 4.0
    if dd10 > 32.0:
        score -= (dd10 - 32.0) * 1.1
    if dd5 > 30.0:
        score -= (dd5 - 30.0) * 0.7
    if trades10 < 95:
        score -= (95 - trades10) * 0.15
    if trades10 > 500:
        score -= (trades10 - 500) * 0.03
    if cagr10 < 0:
        score -= abs(cagr10) * 1.5
    if gross10 > 1.0:
        score -= (gross10 - 1.0) * 400.0
    if sdo10 > 0:
        score -= min(140.0, 20.0 + (sdo10 * 0.30))
    return score


def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["name"] = str(out.get("name", "Apex Fusion Candidate") or "Apex Fusion Candidate")
    out["type"] = "apex_fusion"
    out["signal_mode"] = "after_close"
    out["entry_day_stop_mode"] = "close_confirmed"
    out["enforce_next_day_exit_execution"] = True
    out["allow_margin"] = False
    out["use_market_regime_traffic_light"] = True
    out["traffic_light_block_exposure_mode"] = True

    max_pos = int(out.get("max_positions", 5) or 5)
    max_pos = max(1, min(max_pos, 8))
    out["max_positions"] = max_pos

    max_expo = float(out.get("max_total_exposure_pct_bull", 1.0) or 1.0)
    max_expo = max(0.5, min(max_expo, 1.0))
    out["max_total_exposure_pct_bull"] = max_expo

    max_pos_pct = float(out.get("max_pos_size_pct", 0.20) or 0.20)
    max_pos_pct = max(0.05, min(max_pos_pct, 0.25))
    cap_by_slots = max_expo / float(max_pos)
    out["max_pos_size_pct"] = min(max_pos_pct, cap_by_slots)

    out["risk_per_trade"] = max(0.002, min(float(out.get("risk_per_trade", 0.01) or 0.01), 0.02))
    out["max_total_exposure_pct_bear"] = max(
        0.0,
        min(float(out.get("max_total_exposure_pct_bear", 0.25) or 0.25), 1.0),
    )

    mode = str(out.get("market_exposure_mode", "scaled") or "scaled").lower()
    if mode not in {"scaled", "hybrid", "filter", "exposure"}:
        mode = "scaled"
    out["market_exposure_mode"] = mode

    bear_cash_mode = str(out.get("bear_cash_mode", "off") or "off").lower()
    if bear_cash_mode not in {"off", "hard", "cash", "all_cash"}:
        bear_cash_mode = "off"
    out["bear_cash_mode"] = bear_cash_mode
    if bear_cash_mode in {"hard", "cash", "all_cash"}:
        out["max_total_exposure_pct_bear"] = 0.0
        out["bear_max_positions"] = 0
    else:
        out["bear_max_positions"] = max(1, int(out.get("bear_max_positions", 1) or 1))

    out["yellow_max_positions"] = max(1, min(int(out.get("yellow_max_positions", max_pos) or max_pos), max_pos))
    out["orange_max_positions"] = max(1, min(int(out.get("orange_max_positions", max_pos) or max_pos), out["yellow_max_positions"]))

    out["yellow_risk_scalar"] = max(0.1, min(float(out.get("yellow_risk_scalar", 0.8) or 0.8), 1.0))
    out["orange_risk_scalar"] = max(0.05, min(float(out.get("orange_risk_scalar", 0.4) or 0.4), 1.0))

    out["stop_limit_pct"] = max(0.01, min(float(out.get("stop_limit_pct", 0.04) or 0.04), 0.08))
    out["max_stop_pct"] = max(0.03, min(float(out.get("max_stop_pct", 0.06) or 0.06), 0.12))
    out["ep_max_stop_pct"] = max(0.04, min(float(out.get("ep_max_stop_pct", 0.09) or 0.09), 0.15))
    out["time_stop_days"] = max(0, min(int(out.get("time_stop_days", 12) or 12), 40))

    out["trail_ma"] = bool(out.get("trail_ma", True))
    out["pyramid_enabled"] = bool(out.get("pyramid_enabled", True))
    return out


def _signature(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _mutate(parent: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    parent_sig = _signature(parent)
    for _ in range(16):
        child = dict(parent)
        n_mut = rng.randint(3, 8)
        keys = rng.sample(list(MUTATION_SPACE.keys()), n_mut)
        for key in keys:
            options = list(MUTATION_SPACE[key])
            if not options:
                continue
            current = child.get(key)
            diff_opts = [x for x in options if x != current]
            child[key] = rng.choice(diff_opts if diff_opts else options)
        child = _normalize_config(child)
        if _signature(child) != parent_sig:
            return child
    return _normalize_config(dict(parent))


def main() -> None:
    random.seed(SEED)
    rng = random.Random(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG_PATH, "r") as f:
        base_cfg = json.load(f)
    base_cfg = _normalize_config(base_cfg)

    with open(ROOT / "data" / "cache_indicators.pkl", "rb") as f:
        prepared = pickle.load(f)
    prepared = opt.compress_data(prepared)

    all_dates = list(getattr(prepared, "all_dates"))
    membership, membership_source = build_russell3000_membership_by_day(all_dates)
    global_data = fetch_data_pack(["SPY", "VIX"], days=TRADING_DAYS, backtest_mode=True) or {}
    print(
        f"apex_fusion_search run_tag={RUN_TAG} iterations={ITERATIONS} "
        f"prepared={len(prepared.enriched)} membership={membership_source}"
    )

    seen = set()
    leaderboard: List[TrialResult] = []

    def _record(trial: TrialResult) -> None:
        leaderboard.append(trial)
        leaderboard.sort(key=lambda x: x.score, reverse=True)
        del leaderboard[10:]

    base_sig = _signature(base_cfg)
    seen.add(base_sig)
    obj5 = _evaluate_period(base_cfg, prepared, membership, global_data, START_5Y, END_5Y)
    obj10 = _evaluate_period(base_cfg, prepared, membership, global_data, START_10Y, END_10Y)
    base_trial = TrialResult(
        idx=0,
        score=_score(obj10, obj5),
        cagr10=float(obj10.get("cagr", 0.0) or 0.0) * 100.0,
        dd10=_pct_drawdown(obj10.get("max_drawdown_pct", 0.0)),
        trades10=int(obj10.get("total_trades", 0) or 0),
        cagr5=float(obj5.get("cagr", 0.0) or 0.0) * 100.0,
        dd5=_pct_drawdown(obj5.get("max_drawdown_pct", 0.0)),
        trades5=int(obj5.get("total_trades", 0) or 0),
        eval10=True,
        config=base_cfg,
    )
    _record(base_trial)
    with open(RESULTS_JSONL, "a") as out:
        out.write(json.dumps(base_trial.__dict__) + "\n")
    print(
        f"[0] score={base_trial.score:.2f} cagr10={base_trial.cagr10:.2f}% "
        f"dd10={base_trial.dd10:.2f}% trades10={base_trial.trades10} "
        f"cagr5={base_trial.cagr5:.2f}%"
    )

    for i in range(1, ITERATIONS + 1):
        parent = leaderboard[rng.randrange(max(1, min(len(leaderboard), 5)))].config
        child = _mutate(parent, rng)
        sig = _signature(child)
        if sig in seen:
            continue
        seen.add(sig)

        obj5 = _evaluate_period(child, prepared, membership, global_data, START_5Y, END_5Y)
        cagr5 = float(obj5.get("cagr", 0.0) or 0.0) * 100.0
        dd5 = _pct_drawdown(obj5.get("max_drawdown_pct", 0.0))
        trades5 = int(obj5.get("total_trades", 0) or 0)
        should_run_10y = bool(cagr5 >= GATE_5Y_CAGR and dd5 <= GATE_5Y_DD and trades5 >= GATE_5Y_TRADES)
        obj10 = None
        if should_run_10y:
            obj10 = _evaluate_period(child, prepared, membership, global_data, START_10Y, END_10Y)

        trial = TrialResult(
            idx=i,
            score=_score(obj10, obj5),
            cagr10=(float(obj10.get("cagr", 0.0) or 0.0) * 100.0) if obj10 is not None else float("nan"),
            dd10=_pct_drawdown(obj10.get("max_drawdown_pct", 0.0)) if obj10 is not None else float("nan"),
            trades10=int(obj10.get("total_trades", 0) or 0) if obj10 is not None else 0,
            cagr5=cagr5,
            dd5=dd5,
            trades5=trades5,
            eval10=bool(obj10 is not None),
            config=child,
        )
        _record(trial)
        with open(RESULTS_JSONL, "a") as out:
            out.write(json.dumps(trial.__dict__) + "\n")

        best = leaderboard[0]
        print(
            f"[{i}] score={trial.score:.2f} cagr10={trial.cagr10:.2f}% dd10={trial.dd10:.2f}% "
            f"trades10={trial.trades10} cagr5={trial.cagr5:.2f}% eval10={trial.eval10} | "
            f"best={best.score:.2f}/{best.cagr10:.2f}%"
        )
        gc.collect()

    best = leaderboard[0]
    best_payload = dict(best.config)
    best_payload["name"] = "Apex Fusion (Practical) V2 Candidate"
    with open(BEST_CONFIG_PATH, "w") as f:
        json.dump(best_payload, f, indent=2)
        f.write("\n")
    print(
        f"BEST score={best.score:.2f} cagr10={best.cagr10:.2f}% dd10={best.dd10:.2f}% "
        f"trades10={best.trades10} cagr5={best.cagr5:.2f}% dd5={best.dd5:.2f}% "
        f"saved={BEST_CONFIG_PATH}"
    )
    print(f"TRIAL_LOG {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
