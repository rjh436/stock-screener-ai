import ast
import csv
import json
import math
import tempfile
from collections import Counter
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OPTIMIZER = os.path.join(ROOT, "optimize_superperformance.py")
WINNER_FILE = os.path.join(ROOT, "config", "superperformance_winner.json")
GOLDEN_FILE = os.path.join(ROOT, "config", "GOLDEN_GENOME.json")
CHAMPION_FILE = os.path.join(ROOT, "config", "superperformance_champion.json")
RESULTS_FILE = os.path.join(ROOT, "superperformance_results.csv")
LAST_METRICS_FILE = os.path.join(ROOT, "config", "superperformance_last_metrics.json")


def _objective_profile() -> str:
    profile = str(os.getenv("APEX_OBJECTIVE_PROFILE", "superperformance") or "superperformance").strip().lower()
    if profile not in {"superperformance", "balanced", "defensive"}:
        profile = "superperformance"
    return profile


def _venv_python() -> str:
    for rel in (".venv/bin/python", "venv/bin/python"):
        cand = os.path.join(ROOT, rel)
        if os.path.exists(cand):
            return cand
    return "python3"


def run_optimizer() -> float:
    cmd = [_venv_python(), "-u", OPTIMIZER]
    print(f"\n=== RUN OPTIMIZER: {' '.join(cmd)} ===")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    last_vcp = None
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if "Avg VCP Candidates:" in line:
            try:
                last_vcp = float(line.strip().split(":")[-1])
            except Exception:
                pass
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"Optimizer failed with exit code {code}")
    return last_vcp if last_vcp is not None else -1.0


def load_winner_genome() -> dict:
    if not os.path.exists(WINNER_FILE):
        raise FileNotFoundError(f"Winner file missing: {WINNER_FILE}")
    with open(WINNER_FILE, "r") as f:
        return json.load(f)


def _write_winner_genome(genome: dict) -> None:
    with open(WINNER_FILE, "w") as f:
        json.dump(genome, f, indent=4)


def _fitness_tuple(metrics: dict) -> tuple[float, float, float, float, float, int]:
    cagr = float(metrics.get("cagr", 0.0) or 0.0)
    calmar = float(metrics.get("calmar", 0.0) or 0.0)
    pf = float(metrics.get("pf", 1.0) or 1.0)
    win_loss_ratio = float(metrics.get("win_loss_ratio", 2.0) or 2.0)
    dd = float(metrics.get("dd", 999.0) or 999.0)
    trades = int(metrics.get("trades", 0) or 0)
    return (cagr, calmar, pf, win_loss_ratio, -dd, -abs(trades - 450))


def _is_promotable(metrics: dict) -> bool:
    trades = int(metrics.get("trades", 0) or 0)
    cagr = float(metrics.get("cagr", 0.0) or 0.0)
    dd = float(metrics.get("dd", 999.0) or 999.0)
    pf = float(metrics.get("pf", 0.0) or 0.0)
    win_loss_ratio = float(metrics.get("win_loss_ratio", 0.0) or 0.0)
    objective_profile = _objective_profile()
    if objective_profile == "superperformance":
        min_promote_cagr = float(os.getenv("APEX_MIN_CAGR_PROMOTE_SUPER", "25.0") or 25.0)
        min_promote_pf = float(os.getenv("APEX_PROMOTE_MIN_PF_SUPER", "1.25") or 1.25)
        min_promote_ratio = float(os.getenv("APEX_PROMOTE_MIN_WINLOSS_RATIO_SUPER", "3.0") or 3.0)
    else:
        min_promote_cagr = float(os.getenv("APEX_PROMOTE_MIN_CAGR", "8.0") or 8.0)
        min_promote_pf = float(os.getenv("APEX_PROMOTE_MIN_PF", "1.15") or 1.15)
        min_promote_ratio = float(os.getenv("APEX_PROMOTE_MIN_WINLOSS_RATIO", "3.0") or 3.0)
    return (
        trades > 0
        and cagr >= min_promote_cagr
        and dd > 0.0
        and dd <= 30.0
        and pf >= min_promote_pf
        and win_loss_ratio >= min_promote_ratio
    )


def _is_better(metrics_new: dict, metrics_old: dict) -> bool:
    return _fitness_tuple(metrics_new) > _fitness_tuple(metrics_old)


def _known_winner_gate(genome: dict) -> dict:
    objective_profile = _objective_profile()
    min_pf = float(os.getenv("APEX_KNOWN_WINNER_MIN_PF", "1.25") or 1.25)
    pf_default = "3" if objective_profile == "superperformance" else "2"
    pf_required = int(os.getenv("APEX_KNOWN_WINNER_PF_REQUIRED", pf_default) or pf_default)
    pf_required = max(1, min(3, pf_required))
    timeout_sec = int(os.getenv("APEX_KNOWN_WINNER_TIMEOUT_SEC", "2400") or "2400")
    cache_only = str(os.getenv("APEX_KNOWN_WINNER_CACHE_ONLY", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(genome, tf)
            tmp_path = tf.name

        cmd = [_venv_python(), "-u", "run_phase3_validation.py", "--task", "known-winners", "--strategy-config", tmp_path]
        if cache_only:
            cmd.append("--cache-only")
        run_env = os.environ.copy()
        if cache_only:
            run_env.setdefault("FUNDAMENTAL_CACHE_ONLY", "1")
            run_env.setdefault("APEX_FUNDAMENTAL_CACHE_ONLY", "1")
            run_env.setdefault("FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")
            run_env.setdefault("APEX_FUNDAMENTAL_DISABLE_NETWORK_FETCH", "1")

        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=max(60, timeout_sec),
            env=run_env,
        )
        output = (proc.stdout or "") + "\n" + (proc.stderr or "")
        entry_rows = re.findall(
            r"\[Task 3\.1\]\s+([A-Z0-9._-]+)\s+\|\s+Trades=(\d+)\s+\|\s+EntryDetected=(True|False)\s+\|\s+PF=([0-9.]+)",
            output,
        )
        miss_rows = re.findall(
            r"\[Task 3\.1\]\s+MISS REPORT\s+([A-Z0-9._-]+):\s*(.+)",
            output,
        )

        if not entry_rows:
            err_tail = "\n".join(output.strip().splitlines()[-8:])
            if proc.returncode != 0:
                return {
                    "available": False,
                    "entries": 0,
                    "pf_pass": 0,
                    "pf_required": pf_required,
                    "min_pf": min_pf,
                    "pass": False,
                    "error": f"known-winner subprocess failed (code={proc.returncode})",
                    "miss_reasons": [err_tail] if err_tail else [],
                }
            return {
                "available": False,
                "entries": 0,
                "pf_pass": 0,
                "pf_required": pf_required,
                "min_pf": min_pf,
                "pass": False,
                "error": "known-winner subprocess produced no parseable results",
                "miss_reasons": [err_tail] if err_tail else [],
            }

        entries = 0
        pf_pass = 0
        low_pf: list[str] = []
        miss_counter: Counter[str] = Counter()
        for sym, _trades, detected, pf_txt in entry_rows:
            entry_detected = str(detected).strip().lower() == "true"
            if entry_detected:
                entries += 1
            try:
                pf = float(pf_txt)
            except Exception:
                pf = 0.0
            if entry_detected and pf >= min_pf:
                pf_pass += 1
            elif entry_detected:
                low_pf.append(f"{sym}={pf:.2f}")

        for _sym, reason in miss_rows:
            reason_txt = str(reason or "").strip()
            if not reason_txt:
                continue
            for seg in reason_txt.split(","):
                seg = seg.strip()
                if "TopVCP='" in seg:
                    v = seg.split("TopVCP='", 1)[-1].rstrip("'")
                    if v and not v.lower().startswith("no dominant"):
                        miss_counter[v] += 1
                elif "TopEP='" in seg:
                    v = seg.split("TopEP='", 1)[-1].rstrip("'")
                    if v and not v.lower().startswith("no dominant"):
                        miss_counter[v] += 1

        miss_reasons = [f"{k} ({v})" for k, v in miss_counter.most_common(4)]
        if low_pf:
            miss_reasons.append(f"Low PF ({min_pf:.2f}+ needed): " + ", ".join(low_pf))
        return {
            "available": True,
            "entries": entries,
            "pf_pass": pf_pass,
            "pf_required": pf_required,
            "min_pf": min_pf,
            "pass": (entries >= 3) and (pf_pass >= pf_required),
            "error": "",
            "miss_reasons": miss_reasons,
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "entries": 0,
            "pf_pass": 0,
            "pf_required": pf_required,
            "min_pf": min_pf,
            "pass": False,
            "error": f"known-winner gate timed out after {timeout_sec}s",
            "miss_reasons": [],
        }
    except Exception as exc:
        return {"available": False, "entries": 0, "pass": False, "error": str(exc)}
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _parse_last_results_row() -> dict | None:
    if not os.path.exists(RESULTS_FILE):
        return None
    rows = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    if not rows:
        return None
    last = rows[-1]
    # Expected: gen, cagr, dd, calmar, trades, genome
    if len(last) < 5:
        return None
    try:
        return {
            "cagr": float(last[1]),
            "dd": float(last[2]),
            "calmar": float(last[3]),
            "trades": int(float(last[4])),
        }
    except Exception:
        return None


def _load_last_metrics() -> dict | None:
    if not os.path.exists(LAST_METRICS_FILE):
        return None
    try:
        with open(LAST_METRICS_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _parse_recent_best_result(lookback: int = 25) -> dict | None:
    if not os.path.exists(RESULTS_FILE):
        return None

    parsed: list[dict] = []
    with open(RESULTS_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                metrics = {
                    "cagr": float(row[1]),
                    "dd": float(row[2]),
                    "calmar": float(row[3]),
                    "trades": int(float(row[4])),
                }
                genome = ast.literal_eval(row[5])
                if isinstance(genome, dict):
                    parsed.append({"metrics": metrics, "genome": genome})
            except Exception:
                continue

    if not parsed:
        return None

    tail = parsed[-max(int(lookback), 1) :]
    promotable = [x for x in tail if _is_promotable(x["metrics"])]
    if not promotable:
        return None
    promotable.sort(key=lambda x: _fitness_tuple(x["metrics"]), reverse=True)
    return promotable[0]


def _load_champion() -> dict | None:
    strict_promote = str(os.getenv("APEX_STRICT_CHAMPION_PROMOTION", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if os.path.exists(CHAMPION_FILE):
        try:
            with open(CHAMPION_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("genome"), dict) and isinstance(data.get("metrics"), dict):
                metrics = data.get("metrics", {}) or {}
                cagr = float(metrics.get("cagr", 0.0) or 0.0)
                dd = float(metrics.get("dd", 0.0) or 0.0)
                trades = int(metrics.get("trades", 0) or 0)
                metrics_note = str(data.get("metrics_note", "") or "").strip()
                # Ignore placeholder champions with unknown/empty metrics.
                if metrics_note or (trades == 0 and cagr == 0.0 and dd == 0.0):
                    pass
                else:
                    if strict_promote and not _is_promotable(metrics):
                        print(
                            "⚠️  Stored champion below promote floor; ignoring stale baseline "
                            f"(CAGR {cagr:.2f}%, DD {dd:.2f}%)."
                        )
                    else:
                        return data
        except Exception:
            pass
    best_recent = _parse_recent_best_result()
    if isinstance(best_recent, dict):
        return {
            "genome": best_recent["genome"],
            "metrics": best_recent["metrics"],
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "source": "recent_results",
        }
    return None


def _save_champion(genome: dict, metrics: dict, source: str, known_winners: dict | None = None) -> dict:
    payload = {
        "genome": dict(genome or {}),
        "metrics": {
            "cagr": float(metrics.get("cagr", 0.0) or 0.0),
            "dd": float(metrics.get("dd", 0.0) or 0.0),
            "calmar": float(metrics.get("calmar", 0.0) or 0.0),
            "trades": int(metrics.get("trades", 0) or 0),
            "pf": float(metrics.get("pf", 0.0) or 0.0),
            "win_loss_ratio": float(metrics.get("win_loss_ratio", 0.0) or 0.0),
        },
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
    }
    if isinstance(known_winners, dict) and known_winners:
        payload["known_winners"] = known_winners
    with open(CHAMPION_FILE, "w") as f:
        json.dump(payload, f, indent=4)
    return payload


def _format_list(vals) -> str:
    out = []
    for v in vals:
        if isinstance(v, str):
            out.append(repr(v))
            continue
        if isinstance(v, float):
            s = f"{v:.4f}".rstrip("0").rstrip(".")
            out.append(s)
        else:
            out.append(str(v))
    return "[" + ", ".join(out) + "]"


def _update_gene_list(gene: str, transform_fn) -> bool:
    with open(OPTIMIZER, "r") as f:
        text = f.read()
    pattern = re.compile(rf'("{re.escape(gene)}"\s*:\s*)(\[[^\]]*\])')
    m = pattern.search(text)
    if not m:
        print(f"⚠️  Gene not found: {gene}")
        return False
    try:
        vals = ast.literal_eval(m.group(2))
    except Exception as e:
        print(f"⚠️  Failed to parse gene list for {gene}: {e}")
        return False
    new_vals = transform_fn(list(vals))
    new_vals = list(dict.fromkeys(new_vals))
    text = text[: m.start(2)] + _format_list(new_vals) + text[m.end(2) :]
    with open(OPTIMIZER, "w") as f:
        f.write(text)
    return True


def _set_gene_list(gene: str, values) -> bool:
    fixed = list(values)
    return _update_gene_list(gene, lambda _vals: fixed)


def enforce_superperformance_lockdown() -> None:
    print("ACTION: Superperformance Lockdown (strict Minervini/Qullamaggie constraints)")
    _set_gene_list("rs_gate_min", [80, 85, 90])
    _set_gene_list("mom_rank_min", [75, 80, 90])
    _set_gene_list("runup_3m_min_pct", [20, 30, 40])
    _set_gene_list("adr_min", [2.5, 3.0, 3.5])
    _set_gene_list("template_low_52w_min_pct", [25, 30, 35])
    _set_gene_list("template_off_high_52w_max_pct", [20, 25, 30])
    _set_gene_list("sma200_trend_lookback", [20, 30])
    _set_gene_list("sma200_trend_min_pct", [0.0, 0.5, 1.0])
    _set_gene_list("min_price", [2.0, 5.0])
    _set_gene_list("min_avg_volume_30", [100000, 250000, 500000])
    _set_gene_list("require_rs_line_trend", [True])
    _set_gene_list("min_entry_score", [80, 85, 90, 95])
    _set_gene_list("max_stop_pct", [0.04, 0.05, 0.06, 0.07])
    _set_gene_list("vcp_last_contraction_max_pct", [10, 12, 14])
    _set_gene_list("vcp_damping_ratio", [0.80, 0.85, 0.90])
    _set_gene_list("vcp_volume_dryup_max_ratio", [0.60, 0.75, 0.90])
    _set_gene_list("vcp_vol_contraction_ratio", [0.90, 1.0])
    _set_gene_list("vcp_require_pre_breakout", [True])
    _set_gene_list("vcp_not_breakout_buffer", [0.0, 0.002])
    _set_gene_list("vol_mult", [1.5, 2.0, 2.5])
    _set_gene_list("take_profit_chunk_pct", [0.33, 0.50])
    _set_gene_list("enable_partial_profit", [True])
    _set_gene_list("partial_profit_mode", ["r"])
    _set_gene_list("partial_profit_r", [3.0, 4.0, 5.0])
    _set_gene_list("exit_sma_fast", ["ema10", "ema20"])
    _set_gene_list("exit_sma_slow", ["ema10", "ema20", "sma50"])
    _set_gene_list("risk_per_trade", [0.01, 0.0125, 0.015, 0.02])
    _set_gene_list("max_pos_size_pct", [0.10, 0.12, 0.15, 0.20])
    _set_gene_list("max_total_exposure_pct_bull", [0.8, 0.9, 1.0])
    _set_gene_list("max_total_exposure_pct_bear", [0.0, 0.05, 0.10])


def superperformance_velocity_boost(cagr: float, trades: int) -> None:
    """
    Bounded relaxation for superperformance mode:
    increase at-bats without collapsing into low-quality noise.
    """
    print("ACTION: Superperformance Velocity Boost (bounded relaxation)")
    if trades < 120 or cagr < 5.0:
        _set_gene_list("rs_gate_min", [70, 75, 80, 85, 90])
        _set_gene_list("mom_rank_min", [75, 80, 90, 95])
        _set_gene_list("runup_3m_min_pct", [15, 20, 30, 40, 50])
        _set_gene_list("adr_min", [2.0, 2.5, 3.0, 3.5, 4.0])
        _set_gene_list("min_entry_score", [75, 80, 85, 90, 95])
        _set_gene_list("ep_gap_pct", [0.015, 0.02, 0.03, 0.04, 0.05, 0.06])
        _set_gene_list("ep_vol_mult", [1.5, 2.0, 2.5, 3.0])
        _set_gene_list("vol_mult", [1.25, 1.5, 2.0, 2.5])
        _set_gene_list("vcp_last_contraction_max_pct", [10, 12, 14, 16])
        _set_gene_list("vcp_damping_ratio", [0.85, 0.90, 0.95])
        _set_gene_list("vcp_volume_dryup_max_ratio", [0.9, 1.0, 1.1])
        _set_gene_list("risk_per_trade", [0.0125, 0.015, 0.02, 0.025])
        _set_gene_list("max_positions", [3, 4, 5, 6])
        _set_gene_list("max_pos_size_pct", [0.15, 0.20, 0.25])
        _set_gene_list("time_stop_days", [0, 90, 120, 180])


def widen_net() -> None:
    print("ACTION: Widening Net (lower RS/EP thresholds, relax VCP gates, raise risk_per_trade)")

    def lower_rs_gate(vals):
        vals = [int(v) for v in vals]
        min_val = min(vals) if vals else 85
        new_min = max(70, min_val - 5)
        if new_min not in vals:
            vals.append(new_min)
        return sorted(set(vals))

    def raise_risk(vals):
        new_vals = []
        for v in vals:
            nv = min(0.03, float(v) + 0.0025)
            new_vals.append(round(nv, 3))
        return sorted(set(new_vals))

    def lower_ep_gap(vals):
        vals = [round(float(v), 3) for v in vals]
        min_val = min(vals) if vals else 0.08
        new_min = max(0.04, round(min_val - 0.01, 3))
        if new_min not in vals:
            vals.append(new_min)
        return sorted(set(vals))

    def lower_breakout_buffer(vals):
        vals = [round(float(v), 4) for v in vals]
        min_val = min(vals) if vals else 0.002
        new_min = max(0.0, round(min_val - 0.001, 4))
        if new_min not in vals:
            vals.append(new_min)
        return sorted(set(vals))

    def relax_vcp_tightness(vals):
        vals = [round(float(v), 2) for v in vals]
        max_val = max(vals) if vals else 12.0
        new_max = min(18.0, round(max_val + 1.0, 2))
        if new_max not in vals:
            vals.append(new_max)
        return sorted(set(vals))

    def relax_vcp_dryup(vals):
        vals = [round(float(v), 2) for v in vals]
        max_val = max(vals) if vals else 0.75
        new_max = min(1.10, round(max_val + 0.05, 2))
        if new_max not in vals:
            vals.append(new_max)
        return sorted(set(vals))

    _update_gene_list("rs_gate_min", lower_rs_gate)
    _update_gene_list("risk_per_trade", raise_risk)
    _update_gene_list("ep_gap_pct", lower_ep_gap)
    _update_gene_list("breakout_buffer", lower_breakout_buffer)
    _update_gene_list("vcp_last_contraction_max_pct", relax_vcp_tightness)
    _update_gene_list("vcp_volume_dryup_max_ratio", relax_vcp_dryup)


def targeted_known_winner_repair(known_gate: dict) -> None:
    reasons = " | ".join(str(x) for x in (known_gate or {}).get("miss_reasons", [])).lower()
    if not reasons:
        return
    objective_profile = _objective_profile()
    strict_super = objective_profile == "superperformance"
    entries = int((known_gate or {}).get("entries", 0) or 0)
    has_ep_gap_miss = "ep gap threshold" in reasons
    pf_pass = int((known_gate or {}).get("pf_pass", 0) or 0)

    if strict_super and entries < 3:
        print("ACTION: Known-winner recall boost (entries<3). Broadening EP/VCP and momentum thresholds.")
        _set_gene_list("rs_gate_min", [70, 75, 80, 85])
        _set_gene_list("mom_rank_min", [70, 75, 80, 90])
        _set_gene_list("runup_3m_min_pct", [10, 15, 20, 30])
        _set_gene_list("adr_min", [2.0, 2.5, 3.0, 3.5])
        _set_gene_list("min_entry_score", [75, 80, 85, 90])
        _set_gene_list("ep_gap_pct", [0.015, 0.02, 0.03, 0.04, 0.05])
        _set_gene_list("ep_vol_mult", [1.5, 2.0, 2.5])
        _set_gene_list("vcp_last_contraction_max_pct", [10, 12, 14, 16])
        _set_gene_list("vcp_damping_ratio", [0.85, 0.90, 0.95])
        _set_gene_list("vcp_volume_dryup_max_ratio", [0.9, 1.0, 1.1])

    if "primary rs gate" in reasons:
        print("ACTION: Known-winner misses show RS gate pressure. Capping rs_gate_min to <=85.")

        def cap_rs(vals):
            min_floor = 70 if strict_super else 70
            kept = sorted({int(v) for v in vals if min_floor <= int(v) <= 90})
            if 80 not in kept:
                kept.append(80)
            if strict_super and 70 not in kept:
                kept.append(70)
            if not strict_super and 75 not in kept:
                kept.append(75)
            if 85 not in kept:
                kept.append(85)
            return sorted(set(kept))

        _update_gene_list("rs_gate_min", cap_rs)

    if "ep gap threshold" in reasons:
        print("ACTION: Known-winner misses show EP gap threshold pressure. Capping ep_gap_pct to <=0.06.")

        def lower_ep(vals):
            min_ep = 0.02 if strict_super else 0.03
            max_ep = 0.05 if strict_super else 0.06
            kept = sorted({round(float(v), 3) for v in vals if min_ep <= float(v) <= max_ep})
            if not kept:
                kept = [min_ep, 0.03, 0.04, 0.05]
            else:
                for add in (min_ep, 0.03, 0.04, 0.05):
                    kept.append(add)
            return sorted(set(kept))

        _update_gene_list("ep_gap_pct", lower_ep)

    if "vcp tightness" in reasons:
        print("ACTION: Known-winner misses show VCP tightness pressure. Capping to looser VCP contraction settings.")

        def relax_tight(vals):
            if strict_super:
                kept = sorted({round(float(v), 2) for v in vals if 10.0 <= float(v) <= 14.0})
                if not kept:
                    kept = [10.0, 12.0, 14.0]
                else:
                    kept.extend([10.0, 12.0, 14.0])
            else:
                kept = sorted({round(float(v), 2) for v in vals if float(v) >= 14.0})
                if not kept:
                    kept = [14.0, 16.0, 18.0]
                else:
                    kept.extend([14.0, 16.0, 18.0])
            return sorted(set(kept))

        _update_gene_list("vcp_last_contraction_max_pct", relax_tight)

        def relax_contractions(vals):
            kept = sorted({int(v) for v in vals if int(v) <= 1})
            if 1 not in kept:
                kept.append(1)
            return sorted(set(kept))

        _update_gene_list("vcp_required_contractions", relax_contractions)

    if "vcp volume dry-up" in reasons:
        print("ACTION: Known-winner misses show VCP dry-up pressure. Relaxing dry-up ratio cap.")

        def relax_dryup(vals):
            vals = sorted({round(float(v), 2) for v in vals})
            vals.append(1.0 if strict_super else 1.1)
            return sorted(set(vals))

        _update_gene_list("vcp_volume_dryup_max_ratio", relax_dryup)

    if "low pf" in reasons and entries >= 3 and pf_pass < entries:
        print("ACTION: Known-winner misses show low PF. Tightening risk and avoiding weak-gap churn.")

        def cap_risk(vals):
            kept = sorted({round(float(v), 3) for v in vals if float(v) <= 0.015})
            if not kept:
                kept = [0.01, 0.012, 0.015]
            return sorted(set(kept))

        def tighten_ep_gap(vals):
            kept = sorted({round(float(v), 3) for v in vals if float(v) >= 0.05})
            if not kept:
                kept = [0.05, 0.06]
            return sorted(set(kept))

        def lengthen_time_stop(vals):
            kept = sorted({int(v) for v in vals if int(v) >= 30})
            if not kept:
                kept = [30, 45, 60]
            return sorted(set(kept))

        _update_gene_list("risk_per_trade", cap_risk)
        if not has_ep_gap_miss:
            _update_gene_list("ep_gap_pct", tighten_ep_gap)
        _update_gene_list("time_stop_days", lengthen_time_stop)
        _set_gene_list("take_profit_chunk_pct", [0.33])
        _set_gene_list("exit_sma_slow", ["ema20", "sma50"])


def tighten_shield() -> None:
    print("ACTION: Tightening Shield (reduce max_pos_size_pct, stop_loss_atr_bull)")

    def reduce_pos(vals):
        new_vals = []
        for v in vals:
            nv = max(0.2, float(v) - 0.05)
            new_vals.append(round(nv, 2))
        return sorted(set(new_vals))

    def reduce_stop(vals):
        new_vals = []
        for v in vals:
            nv = max(1.0, float(v) - 0.25)
            new_vals.append(round(nv, 2))
        return sorted(set(new_vals))

    _update_gene_list("max_pos_size_pct", reduce_pos)
    _update_gene_list("stop_loss_atr_bull", reduce_stop)


def quality_tighten() -> None:
    print("ACTION: Quality Tighten (raise momentum floor, cap volatility, cut bear exposure)")

    def raise_mom_floor(vals):
        kept = sorted({int(v) for v in vals if int(v) >= 70})
        return kept or [70, 90]

    def cap_natr(vals):
        kept = sorted({round(float(v), 2) for v in vals if float(v) <= 3.0})
        return kept or [2.0, 2.5, 3.0]

    def cap_bear_exposure(vals):
        kept = sorted({round(float(v), 2) for v in vals if float(v) <= 0.25})
        return kept or [0.0, 0.25]

    _update_gene_list("mom_rank_min", raise_mom_floor)
    _update_gene_list("natr_max", cap_natr)
    _update_gene_list("max_total_exposure_pct_bear", cap_bear_exposure)


def aggressive_expand() -> None:
    print("ACTION: Aggressive Expansion (looser entries, broader stops, larger trend carry)")

    def lower_rs_floor(vals):
        kept = sorted({int(v) for v in vals})
        kept.extend([65, 70, 75, 80, 85, 90])
        return sorted(set(kept))

    def broaden_max_stop(vals):
        kept = sorted({round(float(v), 3) for v in vals})
        kept.extend([0.07, 0.08, 0.09, 0.10])
        return sorted(set(kept))

    def enforce_take_chunk(vals):
        kept = sorted({round(float(v), 3) for v in vals if float(v) <= 0.33})
        kept.extend([0.0, 0.1, 0.15, 0.2, 0.25])
        return sorted(set(kept))

    def widen_bull_stop(vals):
        kept = sorted({round(float(v), 3) for v in vals})
        kept.extend([3.0, 3.5, 4.0, 5.0, 6.0])
        return sorted(set(kept))

    def boost_risk(vals):
        kept = sorted({round(float(v), 3) for v in vals})
        kept.extend([0.03, 0.04, 0.05, 0.06, 0.08])
        return sorted(set(kept))

    def keep_fast_ma(vals):
        kept = sorted({str(v) for v in vals})
        kept.extend(["sma50", "sma200"])
        return sorted(set(kept))

    def keep_score_mode(vals):
        _ = vals
        return ["dual_core"]

    def keep_entry_floor(vals):
        kept = sorted({int(v) for v in vals})
        kept.extend([65, 70, 75, 80, 85])
        return sorted(set(kept))

    def raise_profit_target(vals):
        kept = sorted({round(float(v), 3) for v in vals})
        kept.extend([0.15, 0.20, 0.25, 0.30, 0.40])
        return sorted(set(kept))

    def lengthen_time_stop(vals):
        kept = sorted({int(v) for v in vals})
        kept.extend([0, 90, 120, 180, 240])
        return sorted(set(kept))

    def expand_exposure(vals):
        kept = sorted({round(float(v), 3) for v in vals})
        kept.extend([1.2, 1.5, 2.0, 2.5])
        return sorted(set(kept))

    def broaden_vcp(vals):
        kept = sorted({round(float(v), 2) for v in vals})
        kept.extend([12.0, 14.0, 16.0, 18.0, 20.0])
        return sorted(set(kept))

    def disable_partial(vals):
        _ = vals
        return [False]

    _update_gene_list("rs_gate_min", lower_rs_floor)
    _update_gene_list("max_stop_pct", broaden_max_stop)
    _update_gene_list("take_profit_chunk_pct", enforce_take_chunk)
    _update_gene_list("stop_loss_atr_bull", widen_bull_stop)
    _update_gene_list("risk_per_trade", boost_risk)
    _update_gene_list("exit_sma_fast", keep_fast_ma)
    _update_gene_list("score_mode", keep_score_mode)
    _update_gene_list("min_entry_score", keep_entry_floor)
    _update_gene_list("profit_target_pct", raise_profit_target)
    _update_gene_list("time_stop_days", lengthen_time_stop)
    _update_gene_list("max_total_exposure_pct_bull", expand_exposure)
    _update_gene_list("vcp_last_contraction_max_pct", broaden_vcp)
    _update_gene_list("enable_partial_profit", disable_partial)


def main() -> None:
    loops = 0
    golden = False
    max_loops = int(os.getenv("APEX_MAX_LOOPS", "0") or "0")
    objective_profile = _objective_profile()
    profile = str(os.getenv("APEX_PROFILE", "balanced") or "balanced").strip().lower()
    if profile not in {"aggressive", "balanced", "defensive"}:
        profile = "balanced"
    require_known_winner_gate = str(os.getenv("APEX_REQUIRE_KNOWN_WINNERS", "1") or "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if objective_profile == "superperformance":
        default_target = os.getenv("APEX_MIN_CAGR_PROMOTE_SUPER", "35.0")
    else:
        default_target = "25.0"
    try:
        target_cagr = float(os.getenv("APEX_TARGET_CAGR", default_target) or default_target)
    except Exception:
        target_cagr = float(default_target)
    print(f"Autonomous profile: {profile}")
    print(f"Objective profile: {objective_profile}")
    champion = _load_champion()
    if champion:
        cm = champion.get("metrics", {})
        print(
            "Champion baseline | "
            f"CAGR: {float(cm.get('cagr', 0.0) or 0.0):.2f}% | "
            f"MaxDD: {float(cm.get('dd', 0.0) or 0.0):.2f}% | "
            f"Calmar: {float(cm.get('calmar', 0.0) or 0.0):.3f} | "
            f"Trades: {int(cm.get('trades', 0) or 0)}"
        )
        # Keep working winner pinned to champion before new loops begin.
        try:
            _write_winner_genome(champion.get("genome", {}))
        except Exception as exc:
            print(f"⚠️  Failed to pin champion winner before loop start: {exc}")

    if objective_profile == "superperformance":
        enforce_superperformance_lockdown()
        superperformance_velocity_boost(0.0, 0)

    while True:
        loops += 1
        print(f"\n===== AUTONOMOUS LOOP {loops} =====")
        avg_vcp = run_optimizer()
        if avg_vcp <= 0.0:
            print("VCP LOGIC BROKEN")
            sys.exit(1)

        genome = load_winner_genome()
        metrics = _parse_last_results_row() or {}
        cagr = float(metrics.get("cagr", 0.0))
        dd = float(metrics.get("dd", 0.0))
        calmar = float(metrics.get("calmar", 0.0))
        trades = int(metrics.get("trades", 0))
        asym = _load_last_metrics() or {}
        pf = float(asym.get("pf", 0.0) or 0.0)
        win_loss_ratio = float(asym.get("win_loss_ratio", 0.0) or 0.0)
        candidate_metrics = {
            "cagr": cagr,
            "dd": dd,
            "calmar": calmar,
            "trades": trades,
            "pf": pf,
            "win_loss_ratio": win_loss_ratio,
        }

        print("\n=== WINNER METRICS ===")
        print(f"CAGR: {cagr:.2f}% | MaxDD: {dd:.2f}% | Calmar: {calmar:.3f} | Trades: {trades}")
        if isinstance(asym, dict):
            try:
                avg_win = float(asym.get("avg_win_pct", 0.0) or 0.0)
            except Exception:
                avg_win = 0.0
            try:
                avg_loss = float(asym.get("avg_loss_pct", 0.0) or 0.0)
            except Exception:
                avg_loss = 0.0
            try:
                ratio = float(asym.get("win_loss_ratio", 0.0) or 0.0)
            except Exception:
                ratio = 0.0
            ratio_str = f"{ratio:.2f}" if math.isfinite(ratio) else "inf"
            target_met = ratio >= 3.0 if math.isfinite(ratio) else True
            print(
                "Trade Asymmetry | "
                f"AvgWin%={avg_win:.2f} | AvgLoss%={avg_loss:.2f} | "
                f"Ratio={ratio_str} | Target3to1={target_met}"
            )

        known_gate = {"available": False, "entries": 0, "pass": True, "error": ""}
        if require_known_winner_gate:
            known_gate = _known_winner_gate(genome)
            if known_gate.get("available"):
                pf_pass = int(known_gate.get("pf_pass", 0) or 0)
                pf_required = int(known_gate.get("pf_required", 0) or 0)
                min_pf = float(known_gate.get("min_pf", 0.0) or 0.0)
                print(
                    f"Known-Winner Gate | Entries={int(known_gate.get('entries', 0))}/3 "
                    f"| PFPass={pf_pass}/{pf_required} (minPF={min_pf:.2f}) "
                    f"| Pass={bool(known_gate.get('pass'))}"
                )
                miss_reasons = known_gate.get("miss_reasons") or []
                if miss_reasons:
                    print("Known-Winner Miss Drivers: " + "; ".join(str(x) for x in miss_reasons))
            else:
                print(f"⚠️  Known-Winner Gate unavailable: {known_gate.get('error', 'unknown error')}")

        promotable = _is_promotable(candidate_metrics) and bool(known_gate.get("pass", True))
        if promotable:
            if (champion is None) or _is_better(candidate_metrics, champion.get("metrics", {})):
                champion = _save_champion(
                    genome,
                    candidate_metrics,
                    source=f"loop_{loops}",
                    known_winners=known_gate if require_known_winner_gate else None,
                )
                print("✅ Champion updated from current loop.")
            else:
                try:
                    _write_winner_genome(champion.get("genome", {}))
                    cm = champion.get("metrics", {})
                    print(
                        "↩️  Candidate underperformed champion; restored champion winner "
                        f"(CAGR {float(cm.get('cagr', 0.0) or 0.0):.2f}%, "
                        f"DD {float(cm.get('dd', 0.0) or 0.0):.2f}%, "
                        f"Calmar {float(cm.get('calmar', 0.0) or 0.0):.3f})."
                    )
                except Exception as exc:
                    print(f"⚠️  Failed to restore champion winner: {exc}")
        else:
            why = "metrics" if not _is_promotable(candidate_metrics) else "known-winner gate"
            print(f"⚠️  Candidate not promotable ({why}); champion remains active.")
            if champion:
                try:
                    _write_winner_genome(champion.get("genome", {}))
                except Exception:
                    pass

        if cagr > target_cagr and dd < 25.0:
            print("✅ OBJECTIVE MET: Saving GOLDEN_GENOME.json and exiting.")
            golden_payload = champion.get("genome", genome) if champion else genome
            with open(GOLDEN_FILE, "w") as f:
                json.dump(golden_payload, f, indent=4)
            golden = True
            break

        if require_known_winner_gate and not bool(known_gate.get("pass", True)):
            if objective_profile == "superperformance":
                print("ACTION: Known-winner gate failed. Applying bounded superperformance recovery.")
                superperformance_velocity_boost(cagr, trades)
                targeted_known_winner_repair(known_gate)
            else:
                print("ACTION: Known-winner gate failed. Widening net for next loop.")
                targeted_known_winner_repair(known_gate)
                if profile == "aggressive":
                    aggressive_expand()
                else:
                    widen_net()
        elif cagr < target_cagr:
            if objective_profile == "superperformance":
                superperformance_velocity_boost(cagr, trades)
            else:
                if profile == "aggressive":
                    aggressive_expand()
                elif profile == "defensive":
                    quality_tighten()
                else:
                    widen_net()
        elif cagr > target_cagr and dd > 20.0:
            tighten_shield()
        else:
            print("No rule triggered. Keeping parameters unchanged.")

        if max_loops > 0 and loops >= max_loops:
            break

        time.sleep(2)

    if not golden:
        if champion and isinstance(champion.get("genome"), dict):
            try:
                _write_winner_genome(champion.get("genome", {}))
            except Exception:
                pass
        print("\nCompleted loop limit without GOLDEN_GENOME.")
    else:
        print("\nGolden genome achieved.")


if __name__ == "__main__":
    main()
