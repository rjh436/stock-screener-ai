from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


GOLDEN_VERSION_INFO: Dict[str, Any] = {
    "status": "Golden State",
    "validated_metrics": {"cagr": 0.433, "avg_profit": 0.054, "win_rate": 0.658},
}


@dataclass(frozen=True, slots=True)
class _RenameRecord:
    idx: int
    strategy_type: str
    old_name: str
    new_name: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)


def _set_read_only(path: Path) -> None:
    try:
        os.chmod(path, 0o444)
    except Exception:
        # Best-effort; still a valid master copy even if the FS ignores chmod.
        pass


def _shorten(s: str, max_len: int = 72) -> str:
    s = str(s or "")
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


def _normalize_type(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if t == "wealth":
        return "wealth"
    if t in {"income", "hybrid"}:
        return t
    return "income"


def _apply_golden_state(
    strategies: List[Dict[str, Any]],
    *,
    wealth_top_name: str,
    income_top_name: str,
) -> Tuple[List[Dict[str, Any]], List[_RenameRecord]]:
    renamed: List[_RenameRecord] = []
    cleaned: List[Dict[str, Any]] = []

    wealth_seen = 0
    income_seen = 0

    for idx, strat in enumerate(strategies, start=1):
        old_name = str(strat.get("name") or "")
        s_type = _normalize_type(strat.get("type"))

        if s_type == "wealth":
            wealth_seen += 1
            if wealth_seen == 1:
                new_name = wealth_top_name
            else:
                new_name = f"Apex Wealth Sniper Candidate {wealth_seen}"
        else:
            income_seen += 1
            if income_seen == 1:
                new_name = income_top_name
            else:
                new_name = f"Apex Income Sniper Candidate {income_seen}"

        updated = dict(strat)
        updated["name"] = new_name
        updated["version_info"] = dict(GOLDEN_VERSION_INFO)

        cleaned.append(updated)
        renamed.append(_RenameRecord(idx=idx, strategy_type=s_type, old_name=old_name, new_name=new_name))

    return cleaned, renamed


def main(argv: List[str]) -> int:
    root = _repo_root()
    default_config = root / "config" / "generated_strategies.json"
    default_master = root / "config" / "golden_state_master.json"

    p = argparse.ArgumentParser(description="Finalize 'Golden State' strategy DNA + create a locked master backup.")
    p.add_argument("--config", type=Path, default=default_config, help="Path to generated strategies JSON.")
    p.add_argument("--master", type=Path, default=default_master, help="Path to golden state master JSON.")
    p.add_argument("--force", action="store_true", help="Overwrite existing master file (will re-lock read-only).")
    p.add_argument("--dry-run", action="store_true", help="Print summary but do not write any files.")
    args = p.parse_args(argv)

    config_path: Path = args.config
    master_path: Path = args.master

    if not config_path.exists():
        print(f"ERROR: Missing config: {config_path}")
        return 2

    raw = _read_json(config_path)
    if not isinstance(raw, list):
        print(f"ERROR: Expected a JSON list in {config_path}, got {type(raw).__name__}")
        return 2
    bad_item = next(((i, type(v).__name__) for i, v in enumerate(raw, start=1) if not isinstance(v, dict)), None)
    if bad_item is not None:
        idx, type_name = bad_item
        print(f"ERROR: Expected every item in {config_path} to be an object; item {idx} is {type_name}.")
        return 2

    cleaned, renamed = _apply_golden_state(
        raw,  # type: ignore[arg-type]
        wealth_top_name="Apex Sniper Wealth v1.0",
        income_top_name="Apex Sniper Income v1.0",
    )

    wealth_count = sum(1 for r in renamed if r.strategy_type == "wealth")
    income_count = len(renamed) - wealth_count

    if master_path.exists() and not args.force:
        print(f"ERROR: Refusing to overwrite existing master: {master_path}")
        print("Re-run with `--force` if you intentionally want to replace it.")
        return 3

    if not args.dry_run:
        _atomic_write_json(config_path, cleaned)
        _atomic_write_json(master_path, cleaned)
        _set_read_only(master_path)

    print("\n" + "=" * 72)
    print("GOLDEN STATE LOCKED")
    print("=" * 72)
    print(f"- Strategies processed: {len(renamed)} (wealth={wealth_count}, income/hybrid={income_count})")
    print(f"- Updated config: {config_path}")
    print(f"- Master backup:  {master_path} (read-only best-effort)")
    print("- Version info injected into every strategy:")
    print(f"  {json.dumps(GOLDEN_VERSION_INFO, indent=2)}")
    print("\nRenames:")
    for r in renamed:
        print(f"- [{r.idx:02d}] {r.strategy_type:6s}  {r.new_name}  (was: {_shorten(r.old_name)})")

    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
