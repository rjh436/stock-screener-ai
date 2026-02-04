import ast
import csv
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZER = os.path.join(ROOT, "optimize_superperformance.py")
WINNER_FILE = os.path.join(ROOT, "config", "superperformance_winner.json")
GOLDEN_FILE = os.path.join(ROOT, "config", "GOLDEN_GENOME.json")
RESULTS_FILE = os.path.join(ROOT, "superperformance_results.csv")


def _venv_python() -> str:
    cand = os.path.join(ROOT, "venv", "bin", "python")
    return cand if os.path.exists(cand) else "python3"


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


def _format_list(vals) -> str:
    out = []
    for v in vals:
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


def widen_net() -> None:
    print("ACTION: Widening Net (lower rs_min, raise risk_per_trade)")

    def lower_rs(vals):
        vals = [int(v) for v in vals]
        min_val = min(vals) if vals else 60
        new_min = max(40, min_val - 5)
        if new_min not in vals:
            vals.append(new_min)
        return sorted(set(vals))

    def raise_risk(vals):
        new_vals = []
        for v in vals:
            nv = min(0.03, float(v) + 0.005)
            new_vals.append(round(nv, 3))
        return sorted(set(new_vals))

    _update_gene_list("rs_min", lower_rs)
    _update_gene_list("risk_per_trade", raise_risk)


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


def main() -> None:
    loops = 0
    golden = False
    max_loops = int(os.getenv("APEX_MAX_LOOPS", "0") or "0")

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
        trades = int(metrics.get("trades", 0))

        print("\n=== WINNER METRICS ===")
        print(f"CAGR: {cagr:.2f}% | MaxDD: {dd:.2f}% | Trades: {trades}")

        if cagr > 25.0 and dd < 25.0:
            print("✅ OBJECTIVE MET: Saving GOLDEN_GENOME.json and exiting.")
            with open(GOLDEN_FILE, "w") as f:
                json.dump(genome, f, indent=4)
            golden = True
            break

        if cagr < 15.0 and dd < 10.0:
            widen_net()
        elif cagr > 30.0 and dd > 25.0:
            tighten_shield()
        else:
            print("No rule triggered. Keeping parameters unchanged.")

        if max_loops > 0 and loops >= max_loops:
            break

        time.sleep(2)

    if not golden:
        print("\nCompleted loop limit without GOLDEN_GENOME.")
    else:
        print("\nGolden genome achieved.")


if __name__ == "__main__":
    main()
