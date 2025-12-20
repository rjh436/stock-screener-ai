#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
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
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for rel in TARGETS:
        src = ROOT / rel
        if not src.exists():
            missing.append(str(rel))
            continue
        dest = BACKUP_DIR / rel.name
        shutil.copy2(src, dest)

    if missing:
        sys.stderr.write("Missing files:\n")
        for item in missing:
            sys.stderr.write(f"- {item}\n")
        return 1

    timestamp = datetime.now().isoformat(timespec="seconds")
    note = "Verified $322k Equity / 31.6% CAGR on M3 Max"
    manifest = BACKUP_DIR / "manifest.txt"
    manifest.write_text(f"Timestamp: {timestamp}\n{note}\n", encoding="utf-8")

    print(f"Backup created at {BACKUP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
