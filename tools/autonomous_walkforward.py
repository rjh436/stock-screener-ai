#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
OPTIMIZER = ROOT / "optimize_superperformance.py"
VALIDATION = ROOT / "run_phase3_validation.py"
WINNER_FILE = ROOT / "config" / "superperformance_winner.json"
GENERATED_FILE = ROOT / "config" / "generated_strategies.json"
LAST_METRICS_FILE = ROOT / "config" / "superperformance_last_metrics.json"
CHAMPION_FILE = ROOT / "config" / "superperformance_walkforward_champion.json"
LOG_DIR = ROOT / "logs"
EXPORT_DIR = ROOT / "exports"
CHECKPOINT_FILE = ROOT / "optimizer_checkpoint_sp.pkl"


@dataclass(frozen=True)
class OOSWindow:
    label: str
    start: str
    end: str
    min_cagr: float
    max_dd: float
    min_pf: float


def _venv_python() -> str:
    for rel in (".venv/bin/python", "venv/bin/python"):
        cand = ROOT / rel
        if cand.exists():
            return str(cand)
    return "python3"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _sync_generated_from_winner(genome: Dict[str, Any]) -> None:
    payload = _read_json(GENERATED_FILE, [])
    if not isinstance(payload, list):
        payload = []
    updated = False
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        if str(item.get("name", "")).strip().lower() == "superperformance":
            meta = {k: item.get(k) for k in ("name", "type", "enabled", "validated_metrics") if k in item}
            merged = dict(meta)
            merged.update(genome)
            merged["name"] = meta.get("name", "Superperformance")
            merged["type"] = meta.get("type", "Superperformance")
            payload[idx] = merged
            updated = True
            break
    if not updated:
        payload.append({"name": "Superperformance", "type": "Superperformance", **genome})
    _write_json(GENERATED_FILE, payload)


def _run_stream(cmd: List[str], *, env: Dict[str, str], label: str) -> int:
    print(f"\n[{label}] CMD: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return int(proc.wait())


def _run_capture(cmd: List[str], *, env: Dict[str, str], label: str) -> tuple[int, str]:
    print(f"\n[{label}] CMD: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if output.strip():
        print(output, end="" if output.endswith("\n") else "\n")
    return int(proc.returncode), output


def _known_winner_gate(
    genome: Dict[str, Any],
    *,
    require: bool,
    min_pf: float,
    required_pf_pass: int,
) -> Dict[str, Any]:
    if not require:
        return {"pass": True, "entries": 0, "pf_pass": 0, "required_pf_pass": required_pf_pass, "min_pf": min_pf}

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(genome, tf)
            tmp_path = Path(tf.name)
        env = os.environ.copy()
        env.setdefault("FUNDAMENTAL_CACHE_ONLY", "1")
        env.setdefault("APEX_FUNDAMENTAL_CACHE_ONLY", "1")
        env.setdefault("FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")
        env.setdefault("APEX_FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")
        cmd = [
            _venv_python(),
            "-u",
            str(VALIDATION),
            "--task",
            "known-winners",
            "--strategy-config",
            str(tmp_path),
            "--cache-only",
        ]
        code, output = _run_capture(cmd, env=env, label="known-winners")
        if code != 0:
            return {
                "pass": False,
                "entries": 0,
                "pf_pass": 0,
                "required_pf_pass": required_pf_pass,
                "min_pf": min_pf,
                "error": f"validation exit code {code}",
            }
        rows = re.findall(
            r"\[Task 3\.1\]\s+([A-Z0-9._-]+)\s+\|\s+Entries=(\d+)\s+\|\s+Trades=(\d+)\s+\|\s+EntryDetected=(True|False)\s+\|\s+PF=([0-9.]+|inf)",
            output,
            flags=re.IGNORECASE,
        )
        entries = 0
        pf_pass = 0
        for _sym, _entries_txt, _trades_txt, detected_txt, pf_txt in rows:
            detected = str(detected_txt).lower() == "true"
            if not detected:
                continue
            entries += 1
            try:
                pf_val = float(pf_txt)
            except Exception:
                pf_val = 0.0
            if pf_val >= min_pf:
                pf_pass += 1
        passed = (entries >= 3) and (pf_pass >= required_pf_pass)
        return {
            "pass": bool(passed),
            "entries": int(entries),
            "pf_pass": int(pf_pass),
            "required_pf_pass": int(required_pf_pass),
            "min_pf": float(min_pf),
            "error": "",
        }
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _oos_eval(genome: Dict[str, Any], window: OOSWindow, run_tag: str) -> Dict[str, Any]:
    tmp_cfg: Path | None = None
    tmp_summary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(genome, tf)
            tmp_cfg = Path(tf.name)
        tmp_summary = LOG_DIR / f"wf_summary_{run_tag}_{window.label}.json"
        export_path = EXPORT_DIR / f"phase3_results_{run_tag}_{window.label}.csv"
        env = os.environ.copy()
        env.setdefault("FUNDAMENTAL_CACHE_ONLY", "1")
        env.setdefault("APEX_FUNDAMENTAL_CACHE_ONLY", "1")
        env.setdefault("FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")
        env.setdefault("APEX_FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")
        cmd = [
            _venv_python(),
            "-u",
            str(VALIDATION),
            "--task",
            "full",
            "--strategy-config",
            str(tmp_cfg),
            "--cache-only",
            "--full-start-date",
            window.start,
            "--full-end-date",
            window.end,
            "--export-path",
            str(export_path),
            "--summary-json",
            str(tmp_summary),
        ]
        code = _run_stream(cmd, env=env, label=f"oos:{window.label}")
        if code != 0:
            return {
                "label": window.label,
                "start": window.start,
                "end": window.end,
                "pass": False,
                "error": f"validation exit code {code}",
            }
        summary = _read_json(tmp_summary, {})
        if not isinstance(summary, dict) or not summary:
            return {
                "label": window.label,
                "start": window.start,
                "end": window.end,
                "pass": False,
                "error": "missing summary json",
            }
        cagr = float(summary.get("cagr_pct", 0.0) or 0.0)
        dd = float(summary.get("max_drawdown_pct", 999.0) or 999.0)
        pf = float(summary.get("profit_factor", 0.0) or 0.0)
        trades = int(summary.get("total_trades", 0) or 0)
        passed = bool(cagr >= window.min_cagr and dd <= window.max_dd and pf >= window.min_pf and trades > 0)
        return {
            "label": window.label,
            "start": window.start,
            "end": window.end,
            "cagr_pct": cagr,
            "max_drawdown_pct": dd,
            "profit_factor": pf,
            "total_trades": trades,
            "min_cagr_req": window.min_cagr,
            "max_dd_req": window.max_dd,
            "min_pf_req": window.min_pf,
            "pass": passed,
            "error": "",
            "summary_path": str(tmp_summary),
        }
    finally:
        if tmp_cfg is not None:
            try:
                tmp_cfg.unlink(missing_ok=True)
            except Exception:
                pass


def _candidate_score(in_sample: Dict[str, Any], oos_results: List[Dict[str, Any]], known_gate: Dict[str, Any]) -> float:
    cagr = float(in_sample.get("cagr", 0.0) or 0.0)
    dd = float(in_sample.get("dd", 999.0) or 999.0)
    pf = float(in_sample.get("pf", 0.0) or 0.0)
    score = (cagr * 2.0) + (min(pf, 6.0) * 4.0) - (max(dd - 25.0, 0.0) * 1.5)
    weights = {"oos_2019_2021": 1.2, "oos_2022_now": 1.8}
    for res in oos_results:
        if not isinstance(res, dict):
            continue
        w = float(weights.get(str(res.get("label", "")), 1.0))
        o_cagr = float(res.get("cagr_pct", 0.0) or 0.0)
        o_dd = float(res.get("max_drawdown_pct", 999.0) or 999.0)
        o_pf = float(res.get("profit_factor", 0.0) or 0.0)
        score += (o_cagr * w) + (min(o_pf, 5.0) * 2.0) - (max(o_dd - 30.0, 0.0) * w)
        if not bool(res.get("pass", False)):
            score -= 20.0
    if bool(known_gate.get("pass", False)):
        score += 8.0
    else:
        score -= 15.0
    return float(score)


def _load_champion() -> Dict[str, Any] | None:
    payload = _read_json(CHAMPION_FILE, None)
    if isinstance(payload, dict) and isinstance(payload.get("genome"), dict):
        return payload
    return None


def _write_champion(genome: Dict[str, Any], in_sample: Dict[str, Any], oos: List[Dict[str, Any]], known_gate: Dict[str, Any], score: float, source: str) -> Dict[str, Any]:
    payload = {
        "genome": dict(genome),
        "in_sample": dict(in_sample),
        "oos": list(oos),
        "known_winners": dict(known_gate),
        "score": float(score),
        "source": str(source),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(CHAMPION_FILE, payload)
    return payload


def _optimizer_env(train_start: str, train_end: str) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("APEX_UNIVERSE", "RUSSELL3000")
    env.setdefault("APEX_POPULATION_SIZE", "20")
    env.setdefault("APEX_GENERATIONS", "6")
    env.setdefault("APEX_MAX_WORKERS", "10")
    env.setdefault("APEX_MAX_WORKERS_HARD_CAP", "6")
    env.setdefault("APEX_MP_START_METHOD", "fork")
    env.setdefault("APEX_PRUNE_PREPARED_DF", "1")
    env.setdefault("APEX_MEMORY_UTILIZATION", "0.72")
    env.setdefault("APEX_MEMORY_HEADROOM_GB", "6")
    env.setdefault("APEX_POOL_MAX_TASKS_PER_CHILD", "4")
    env.setdefault("APEX_MARKET_EXPOSURE_MODE", "exposure")
    env.setdefault(
        "APEX_OPTIMIZER_COST_BPS",
        str(os.getenv("APEX_WF_OPTIMIZER_COST_BPS", "25") or "25"),
    )
    env.setdefault(
        "APEX_OPTIMIZER_TRACE_REJECTS",
        str(os.getenv("APEX_WF_OPTIMIZER_TRACE_REJECTS", "0") or "0"),
    )
    env.setdefault(
        "APEX_OPTIMIZER_TRACE_MAX_LINES",
        str(os.getenv("APEX_WF_OPTIMIZER_TRACE_MAX_LINES", "3000") or "3000"),
    )
    env.setdefault(
        "APEX_MAX_TRADES_SOFT",
        str(os.getenv("APEX_WF_MAX_TRADES_SOFT", "650") or "650"),
    )
    env.setdefault("APEX_RESUME_CHECKPOINT", "0")
    env["APEX_START_DATE"] = str(train_start)
    env["APEX_END_DATE"] = str(train_end)
    return env


def _load_in_sample_metrics() -> Dict[str, Any]:
    payload = _read_json(LAST_METRICS_FILE, {})
    if not isinstance(payload, dict):
        return {}
    return {
        "cagr": float(payload.get("cagr", 0.0) or 0.0),
        "cagr_5y": float(payload.get("cagr_5y", 0.0) or 0.0),
        "cagr_3y": float(payload.get("cagr_3y", 0.0) or 0.0),
        "dd": float(payload.get("dd", 999.0) or 999.0),
        "pf": float(payload.get("pf", 0.0) or 0.0),
        "win_loss_ratio": float(payload.get("win_loss_ratio", 0.0) or 0.0),
        "trades": int(payload.get("trades", 0) or 0),
        "updated_at": str(payload.get("updated_at", "")),
    }


def _promotable(
    in_sample: Dict[str, Any],
    oos_results: List[Dict[str, Any]],
    known_gate: Dict[str, Any],
    known_gate_mode: str,
) -> bool:
    mode = str(known_gate_mode or "penalty").strip().lower()
    if mode in {"hard", "strict", "require", "required"} and not bool(known_gate.get("pass", False)):
        return False
    cagr = float(in_sample.get("cagr", 0.0) or 0.0)
    dd = float(in_sample.get("dd", 999.0) or 999.0)
    pf = float(in_sample.get("pf", 0.0) or 0.0)
    trades = int(in_sample.get("trades", 0) or 0)
    if cagr < float(os.getenv("APEX_WF_MIN_IS_CAGR", "15") or "15"):
        return False
    if dd > float(os.getenv("APEX_WF_MAX_IS_DD", "32") or "32"):
        return False
    if pf < float(os.getenv("APEX_WF_MIN_IS_PF", "1.25") or "1.25"):
        return False
    if trades < int(os.getenv("APEX_WF_MIN_IS_TRADES", "120") or "120"):
        return False
    return all(bool(r.get("pass", False)) for r in oos_results)


def _default_windows() -> List[OOSWindow]:
    today = datetime.now().date().isoformat()
    dd_cap = float(os.getenv("APEX_WF_OOS_MAX_DD", "35") or "35")
    min_pf = float(os.getenv("APEX_WF_OOS_MIN_PF", "1.20") or "1.20")
    return [
        OOSWindow(
            label="oos_2019_2021",
            start=str(os.getenv("APEX_WF_OOS1_START", "2019-01-01") or "2019-01-01"),
            end=str(os.getenv("APEX_WF_OOS1_END", "2021-12-31") or "2021-12-31"),
            min_cagr=float(os.getenv("APEX_WF_OOS1_MIN_CAGR", "8.0") or "8.0"),
            max_dd=dd_cap,
            min_pf=min_pf,
        ),
        OOSWindow(
            label="oos_2022_now",
            start=str(os.getenv("APEX_WF_OOS2_START", "2022-01-01") or "2022-01-01"),
            end=str(os.getenv("APEX_WF_OOS2_END", today) or today),
            min_cagr=float(os.getenv("APEX_WF_OOS2_MIN_CAGR", "15.0") or "15.0"),
            max_dd=dd_cap,
            min_pf=min_pf,
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous walk-forward optimizer for Superperformance.")
    parser.add_argument("--max-loops", type=int, default=int(os.getenv("APEX_WF_MAX_LOOPS", "4") or "4"))
    parser.add_argument("--no-improve-stop", type=int, default=int(os.getenv("APEX_WF_NO_IMPROVE_STOP", "2") or "2"))
    parser.add_argument("--sleep-sec", type=int, default=int(os.getenv("APEX_WF_SLEEP_SEC", "5") or "5"))
    parser.add_argument("--preflight", action="store_true", help="Print config and exit without running optimization.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_start = str(os.getenv("APEX_WF_TRAIN_START", "2006-02-16") or "2006-02-16")
    train_end = str(os.getenv("APEX_WF_TRAIN_END", "2018-12-31") or "2018-12-31")
    known_gate_mode = str(os.getenv("APEX_WF_KNOWN_GATE_MODE", "penalty") or "penalty").strip().lower()
    if known_gate_mode in {"0", "off", "false", "no", "disabled", "ignore"}:
        known_gate_mode = "off"
    elif known_gate_mode in {"1", "hard", "strict", "require", "required", "true", "yes"}:
        known_gate_mode = "hard"
    else:
        known_gate_mode = "penalty"
    require_known = known_gate_mode in {"penalty", "hard"}
    known_min_pf = float(os.getenv("APEX_WF_KNOWN_MIN_PF", "1.25") or "1.25")
    known_pf_required = int(os.getenv("APEX_WF_KNOWN_PF_REQUIRED", "2") or "2")
    windows = _default_windows()

    print("AUTONOMOUS WALK-FORWARD TUNER")
    print(f"Train window: {train_start} -> {train_end}")
    print(f"OOS windows: {[f'{w.label}:{w.start}->{w.end}' for w in windows]}")
    print(f"Known-winner gate mode: {known_gate_mode} (evaluate={require_known})")
    print(f"Loop limits: max_loops={args.max_loops}, no_improve_stop={args.no_improve_stop}")

    if args.preflight:
        return 0

    champion = _load_champion()
    if champion:
        print(
            "Loaded existing walk-forward champion | "
            f"score={float(champion.get('score', 0.0) or 0.0):.2f} "
            f"| updated_at={champion.get('updated_at', '')}"
        )
        try:
            _write_json(WINNER_FILE, champion.get("genome", {}))
            _sync_generated_from_winner(champion.get("genome", {}))
        except Exception:
            pass

    no_improve = 0
    for loop_idx in range(1, max(1, args.max_loops) + 1):
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n===== WALKFORWARD LOOP {loop_idx}/{args.max_loops} | {run_tag} =====")

        env = _optimizer_env(train_start, train_end)
        reset_checkpoint = str(os.getenv("APEX_WF_RESET_CHECKPOINT_EACH_LOOP", "1") or "1").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if reset_checkpoint and CHECKPOINT_FILE.exists():
            try:
                CHECKPOINT_FILE.unlink()
                print(f"[optimize] Removed stale checkpoint: {CHECKPOINT_FILE}")
            except Exception as exc:
                print(f"[optimize] Warning: failed to remove checkpoint ({exc})")
        code = _run_stream([_venv_python(), "-u", str(OPTIMIZER)], env=env, label="optimize")
        if code != 0:
            print(f"Optimizer failed with code {code}.")
            return code

        genome = _read_json(WINNER_FILE, {})
        if not isinstance(genome, dict) or not genome:
            print("Winner genome missing after optimization.")
            return 2
        in_sample = _load_in_sample_metrics()
        known_gate = _known_winner_gate(
            genome,
            require=require_known,
            min_pf=known_min_pf,
            required_pf_pass=known_pf_required,
        )
        oos_results: List[Dict[str, Any]] = []
        for window in windows:
            oos_results.append(_oos_eval(genome, window, run_tag))
        score = _candidate_score(in_sample, oos_results, known_gate)
        promotable = _promotable(
            in_sample,
            oos_results,
            known_gate,
            known_gate_mode=known_gate_mode,
        )

        print(
            f"Candidate summary | IS CAGR={in_sample.get('cagr', 0.0):.2f}% "
            f"| IS DD={in_sample.get('dd', 999.0):.2f}% | IS PF={in_sample.get('pf', 0.0):.2f} "
            f"| known_gate={bool(known_gate.get('pass', False))} | score={score:.2f}"
        )
        print(
            f"Known-winner detail | mode={known_gate_mode} | pass={bool(known_gate.get('pass', False))} "
            f"| entries={int(known_gate.get('entries', 0) or 0)} "
            f"| pf_pass={int(known_gate.get('pf_pass', 0) or 0)}/"
            f"{int(known_gate.get('required_pf_pass', 0) or 0)}"
        )
        for res in oos_results:
            print(
                f"OOS {res.get('label')} | pass={bool(res.get('pass', False))} "
                f"| CAGR={float(res.get('cagr_pct', 0.0) or 0.0):.2f}% "
                f"| DD={float(res.get('max_drawdown_pct', 999.0) or 999.0):.2f}% "
                f"| PF={float(res.get('profit_factor', 0.0) or 0.0):.2f} "
                f"| Trades={int(res.get('total_trades', 0) or 0)}"
            )

        champion_score = float(champion.get("score", float("-inf")) if champion else float("-inf"))
        improved = bool(promotable and (not champion or score > champion_score))
        if improved:
            champion = _write_champion(
                genome=genome,
                in_sample=in_sample,
                oos=oos_results,
                known_gate=known_gate,
                score=score,
                source=f"loop_{loop_idx}",
            )
            _write_json(WINNER_FILE, genome)
            _sync_generated_from_winner(genome)
            no_improve = 0
            print(f"✅ Champion promoted. New score={score:.2f}")
        else:
            no_improve += 1
            print(
                "↩️  Candidate not promoted "
                f"(promotable={promotable}, score={score:.2f}, champion_score={champion_score:.2f})."
            )
            if champion and isinstance(champion.get("genome"), dict):
                _write_json(WINNER_FILE, champion.get("genome", {}))
                _sync_generated_from_winner(champion.get("genome", {}))

        if champion and no_improve >= max(1, args.no_improve_stop):
            print(
                "Converged: no champion improvement across "
                f"{no_improve} consecutive loops."
            )
            break
        time.sleep(max(0, int(args.sleep_sec)))

    if champion:
        print("\nFINAL CHAMPION")
        print(
            f"Score={float(champion.get('score', 0.0) or 0.0):.2f} | "
            f"Updated={champion.get('updated_at', '')}"
        )
        return 0
    print("\nNo promotable champion produced.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
