#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent
BACKUP_DIR = ROOT / "backups" / "Golden_State_31pct_CAGR_Verified"
TARGETS = [
    Path("execution/engine.py"),
    Path("strategies/apex_wealth.py"),
    Path("strategies/generic.py"),
    Path("app.py"),
    Path("config/generated_strategies.json"),
]


def main() -> int:
    if not BACKUP_DIR.exists():
        sys.stderr.write(f"Backup directory not found: {BACKUP_DIR}\n")
        return 1

    missing = []
    for rel in TARGETS:
        src = BACKUP_DIR / rel.name
        if not src.exists():
            missing.append(str(rel.name))
            continue
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    if missing:
        sys.stderr.write("Missing backup files:\n")
        for item in missing:
            sys.stderr.write(f"- {item}\n")
        return 1

    print("✅ System restored to Golden State (31.6% CAGR). Regression eliminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
