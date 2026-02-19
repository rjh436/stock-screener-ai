#!/usr/bin/env python3
"""Run the full product-quality audit suite and emit a unified summary report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    merged = "\n".join(x for x in [out, err] if x).strip()
    return proc.returncode, merged


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_python(root: Path, requested: str | None) -> str:
    if requested:
        return requested
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Apex Sniper product audit suite")
    parser.add_argument("--base-url", default="http://localhost:8501")
    parser.add_argument("--out-dir", default="output/playwright")
    parser.add_argument("--python", default=None, help="Python executable to use for child test scripts")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    py_exec = _resolve_python(root, args.python)

    started = time.time()
    steps = []

    def add_step(name: str, cmd: list[str]) -> None:
        rc, log = _run(cmd, cwd=root)
        steps.append({"name": name, "returncode": rc, "log": log})

    add_step(
        "desktop_forensic_audit",
        [py_exec, "tools/test_app_ux_functional_e2e.py", "--base-url", args.base_url],
    )
    add_step(
        "live_simulator_smoke",
        [py_exec, "tools/test_live_simulator_e2e.py", "--base-url", args.base_url, "--universe", "RUSSELL3000"],
    )
    add_step(
        "mobile_snapshots",
        [py_exec, "tools/test_mobile_mode_snapshots.py", "--base-url", args.base_url],
    )
    add_step(
        "repo_hygiene",
        [py_exec, "tools/repo_hygiene_audit.py"],
    )

    forensic = _read_json(out_dir / "app_ux_functional_report.json")
    smoke = _read_json(out_dir / "live_simulator_smoke_report.json")

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - started, 2),
        "base_url": args.base_url,
        "python_executable": py_exec,
        "suite_status": "pass" if all(s["returncode"] == 0 for s in steps) else "fail",
        "steps": steps,
        "forensic_summary": (forensic or {}).get("summary", {}),
        "smoke_summary": (smoke or {}).get("summary", {}),
        "artifacts": {
            "forensic_report": str(out_dir / "app_ux_functional_report.json"),
            "smoke_report": str(out_dir / "live_simulator_smoke_report.json"),
            "mobile_live": str(out_dir / "mobile_01_live.png"),
            "mobile_backtest": str(out_dir / "mobile_02_backtest.png"),
            "mobile_simulator": str(out_dir / "mobile_03_simulator.png"),
            "hygiene_reports": str((root / "output" / "hygiene").resolve()),
        },
    }

    out_path = out_dir / "product_audit_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(str(out_path))
    print(json.dumps({"suite_status": summary["suite_status"], "elapsed_sec": summary["elapsed_sec"]}))
    return 1 if summary["suite_status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
