from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


MASTER_WEALTH_NAME = "Apex Sniper Wealth v1.0"
MASTER_INCOME_NAME = "Apex Sniper Income v1.0"

GOLDEN_VERSION_INFO: Dict[str, Any] = {
    "status": "Golden State",
    "validated_metrics": {"cagr": 0.433, "avg_profit": 0.054, "win_rate": 0.658},
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _normalize_type(raw: Any) -> str:
    t = str(raw or "").strip().lower()
    if t == "wealth":
        return "wealth"
    if t in {"income", "hybrid"}:
        return t
    return "income"


def _name_has_candidate(name: str) -> bool:
    return "candidate" in (name or "").lower()


def _find_master_by_name(strategies: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for s in strategies:
        if str(s.get("name") or "").strip() == name:
            return s
    return None


def _find_first_by_type(strategies: List[Dict[str, Any]], *, want: str) -> Optional[Dict[str, Any]]:
    want = str(want).strip().lower()
    for s in strategies:
        if _normalize_type(s.get("type")) == want:
            return s
    return None


def _select_or_build_masters(strategies: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Prefer explicit masters by name (post-finalize flow).
    wealth = _find_master_by_name(strategies, MASTER_WEALTH_NAME)
    income = _find_master_by_name(strategies, MASTER_INCOME_NAME)

    # Fallback: pick first wealth + first income/hybrid and promote them.
    if wealth is None:
        wealth = _find_first_by_type(strategies, want="wealth")
    if income is None:
        income = _find_first_by_type(strategies, want="income") or _find_first_by_type(strategies, want="hybrid")

    if wealth is None or income is None:
        raise RuntimeError("Could not locate both a wealth and an income/hybrid strategy in the config.")

    # Clone to avoid mutating originals during selection.
    wealth_out = dict(wealth)
    wealth_out["name"] = MASTER_WEALTH_NAME
    wealth_out["type"] = "wealth"
    wealth_out["version_info"] = dict(GOLDEN_VERSION_INFO)

    income_out = dict(income)
    income_out["name"] = MASTER_INCOME_NAME
    income_out["type"] = _normalize_type(income_out.get("type"))
    if income_out["type"] == "wealth":
        income_out["type"] = "income"
    income_out["version_info"] = dict(GOLDEN_VERSION_INFO)

    return wealth_out, income_out


def main(argv: List[str]) -> int:
    root = _repo_root()
    config_path = root / "config" / "generated_strategies.json"

    p = argparse.ArgumentParser(description="Purge GA candidates and keep only Golden State masters.")
    p.add_argument("--config", type=Path, default=config_path, help="Path to config/generated_strategies.json")
    p.add_argument("--dry-run", action="store_true", help="Print what would change, without writing the file.")
    args = p.parse_args(argv)

    config: Path = args.config
    if not config.exists():
        print(f"ERROR: Missing config: {config}")
        return 2

    raw = _read_json(config)
    if not isinstance(raw, list):
        print(f"ERROR: Expected a JSON list in {config}, got {type(raw).__name__}")
        return 2

    bad_item = next(((i, type(v).__name__) for i, v in enumerate(raw, start=1) if not isinstance(v, dict)), None)
    if bad_item is not None:
        idx, type_name = bad_item
        print(f"ERROR: Expected every item in {config} to be an object; item {idx} is {type_name}.")
        return 2

    strategies: List[Dict[str, Any]] = raw  # type: ignore[assignment]

    total = len(strategies)
    candidates = sum(1 for s in strategies if _name_has_candidate(str(s.get("name") or "")))

    try:
        wealth_master, income_master = _select_or_build_masters(strategies)
    except Exception as e:
        print(f"ERROR: {e}")
        return 3

    cleaned: List[Dict[str, Any]] = [wealth_master, income_master]

    if not args.dry_run:
        _atomic_write_json(config, cleaned)

    print("\n" + "=" * 72)
    print("POPULATION CLEANED")
    print("=" * 72)
    print(f"- Source strategies: {total}")
    print(f"- Purged candidates: {candidates}")
    print(f"- Retained: {len(cleaned)} (expected=2)")
    print("\nRetained masters:")
    for s in cleaned:
        print(f"- {str(s.get('type') or '?'):6s}  {str(s.get('name') or '')}")
    print("=" * 72 + "\n")

    if len(cleaned) != 2:
        print("ERROR: Output does not contain exactly two strategies.")
        return 4

    if str(cleaned[0].get("name") or "") != MASTER_WEALTH_NAME or str(cleaned[1].get("name") or "") != MASTER_INCOME_NAME:
        print("ERROR: Output names do not match the required master names.")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

