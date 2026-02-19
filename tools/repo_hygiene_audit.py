#!/usr/bin/env python3
"""Repository hygiene audit for documentation, secrets risk, and quality hotspots."""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "hygiene"


@dataclass
class Finding:
    severity: str  # critical | high | medium | low
    category: str
    message: str
    evidence: list[str]


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return proc.returncode, (proc.stdout or ""), (proc.stderr or "")


def _git_tracked_files() -> list[str]:
    rc, out, _err = _run(["git", "ls-files"])
    if rc != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _match_any(path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.search(path) for p in patterns)


def _check_docs(findings: list[Finding]) -> None:
    required = ["README.md", "SECURITY.md", "CONTRIBUTING.md"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        findings.append(
            Finding(
                severity="high",
                category="documentation",
                message=f"Missing root docs: {', '.join(missing)}",
                evidence=missing,
            )
        )


def _check_tracked_risk_files(findings: list[Finding], tracked: list[str]) -> None:
    secret_patterns = [
        re.compile(r"(^|/)\.env($|[.])"),
        re.compile(r"schwab_tokens", re.I),
        re.compile(r"secret", re.I),
        re.compile(r"credentials?", re.I),
        re.compile(r"(^|/)\.git-credentials$"),
        re.compile(r"\.pem$", re.I),
        re.compile(r"\.key$", re.I),
    ]
    artifact_patterns = [
        re.compile(r"(^|/)logs/"),
        re.compile(r"(^|/)exports/"),
        re.compile(r"(^|/)output/"),
        re.compile(r"(^|/)tmp/"),
        re.compile(r"\.pkl$", re.I),
        re.compile(r"\.db$", re.I),
        re.compile(r"(^|/)backups/"),
        re.compile(r"(^|/)_archive/"),
    ]

    secret_hits = [p for p in tracked if _match_any(p, secret_patterns)]
    artifact_hits = [p for p in tracked if _match_any(p, artifact_patterns)]

    if secret_hits:
        findings.append(
            Finding(
                severity="critical",
                category="secrets",
                message="Potential secret-bearing files are tracked in git.",
                evidence=secret_hits[:50],
            )
        )

    if artifact_hits:
        findings.append(
            Finding(
                severity="medium",
                category="repo_hygiene",
                message="Generated/archive/runtime artifacts are tracked and may bloat reviewability.",
                evidence=artifact_hits[:80],
            )
        )


def _check_large_tracked(findings: list[Finding], tracked: list[str]) -> None:
    large = []
    threshold = 5 * 1024 * 1024  # 5 MB
    for rel in tracked:
        path = ROOT / rel
        if not path.exists() or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except Exception:
            continue
        if size >= threshold:
            large.append(f"{rel} ({size / (1024*1024):.1f} MB)")

    if large:
        findings.append(
            Finding(
                severity="medium",
                category="repo_size",
                message="Large tracked files detected.",
                evidence=large[:40],
            )
        )


def _check_broad_except(findings: list[Finding]) -> None:
    tracked = _git_tracked_files()
    candidates: list[Path] = []
    allowed_files = {"app.py", "apex_controller.py"}
    allowed_prefixes = ("data/", "execution/", "simulation/", "strategies/")
    for rel in tracked:
        if not rel.endswith(".py"):
            continue
        if rel not in allowed_files and not rel.startswith(allowed_prefixes):
            continue
        if rel.startswith("backups/") or rel.startswith("_archive/") or rel.startswith("tools/_archive/"):
            continue
        if rel.endswith(".bak") or ".bak." in rel:
            continue
        if rel.endswith("_v3_static.py") or rel.endswith(".py.bak"):
            continue
        if rel.startswith("output/") or rel.startswith("tmp/"):
            continue
        p = ROOT / rel
        if p.exists():
            candidates.append(p)

    patterns = [
        re.compile(r"^\s*except\s*:\s*(pass|continue)?\s*$"),
        re.compile(r"^\s*except Exception:\s*(pass|continue)\s*$"),
    ]
    hits: list[str] = []
    for path in candidates:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        rel = str(path.relative_to(ROOT))
        for i, line in enumerate(lines, start=1):
            if any(p.search(line) for p in patterns):
                hits.append(f"{rel}:{i}: {line.strip()}")

    if hits:
        findings.append(
            Finding(
                severity="medium",
                category="error_handling",
                message="Broad exception swallowing patterns detected in runtime code.",
                evidence=hits[:120],
            )
        )


def _score(findings: list[Finding]) -> dict:
    weights = {"critical": 40, "high": 20, "medium": 8, "low": 3}
    penalty = sum(weights.get(f.severity, 0) for f in findings)
    score = max(0, 100 - penalty)
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    return {"score": score, "penalty": penalty, "counts": counts}


def main() -> int:
    tracked = _git_tracked_files()
    findings: list[Finding] = []
    _check_docs(findings)
    _check_tracked_risk_files(findings, tracked)
    _check_large_tracked(findings, tracked)
    _check_broad_except(findings)
    score = _score(findings)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_path = OUT_DIR / f"repo_hygiene_report_{ts}.json"

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tracked_file_count": len(tracked),
        "scorecard": score,
        "findings": [
            {
                "severity": f.severity,
                "category": f.category,
                "message": f.message,
                "evidence": f.evidence,
            }
            for f in findings
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))

    print(str(out_path))
    print(
        json.dumps(
            {
                "score": score["score"],
                "counts": score["counts"],
                "finding_count": len(findings),
            }
        )
    )
    for f in findings:
        print(f"[{f.severity.upper()}] {f.category}: {f.message}")

    # fail on critical findings
    return 1 if any(f.severity == "critical" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
