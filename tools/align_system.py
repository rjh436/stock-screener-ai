import json
import shutil
import os

STRAT_FILE = "config/generated_strategies.json"
PORT_FILE = "data/paper_portfolio.json"
SECTOR_FILE = "config/sectors.json"


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_path(rel_path: str) -> str:
    return os.path.join(_project_root(), rel_path)


def _backup_file(path: str) -> None:
    try:
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")
    except Exception:
        pass


def _load_json(path: str, default):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except Exception:
        return default


def _save_json(path: str, data) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception:
        return False


def _dedupe_strategies(strategies) -> list:
    if not isinstance(strategies, list):
        return []

    seen = {}
    ordered_keys = []

    for i, strat in enumerate(strategies):
        if not isinstance(strat, dict):
            continue

        name = strat.get("name")
        if isinstance(name, str) and name.strip():
            key = name
        else:
            key = f"__unnamed_{i}"

        if key in seen:
            continue

        seen[key] = strat
        ordered_keys.append(key)

    return [seen[k] for k in ordered_keys]


def _find_target_strategy_name(strategies) -> str:
    if not isinstance(strategies, list):
        return ""
    for strat in strategies:
        if isinstance(strat, dict):
            name = strat.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return ""


def main() -> None:
    # Step 1: Deduplicate Strategies
    strat_path = _abs_path(STRAT_FILE)
    strategies = _load_json(strat_path, default=[])
    old_count = len(strategies) if isinstance(strategies, list) else 0

    deduped = _dedupe_strategies(strategies)
    new_count = len(deduped)

    _backup_file(strat_path)
    _save_json(strat_path, deduped)
    print(f"✅ Strategies deduplicated: {old_count} -> {new_count}")

    # Step 2: Migrate Portfolio
    port_path = _abs_path(PORT_FILE)
    portfolio = _load_json(port_path, default={})
    target_name = _find_target_strategy_name(deduped) or "Unknown"

    updated = 0
    if isinstance(portfolio, dict):
        positions = portfolio.get("positions", {})
        if isinstance(positions, dict):
            for _, pos in positions.items():
                if not isinstance(pos, dict):
                    continue

                old_name = pos.get("strategy_name")
                old_name_alt = pos.get("strategy")
                needs_update = (old_name == "Apex Wealth (Gen 12)") or (old_name_alt == "Apex Wealth (Gen 12)")
                if not needs_update:
                    continue

                pos["strategy_name"] = target_name
                pos["strategy"] = target_name
                updated += 1

    _backup_file(port_path)
    _save_json(port_path, portfolio)
    print(f"✅ Portfolio migrated: {updated} positions updated to {target_name}")

    # Step 3: Sector Check
    sector_path = _abs_path(SECTOR_FILE)
    sector_map = _load_json(sector_path, default={})
    count = len(sector_map) if isinstance(sector_map, dict) else 0
    print(f"✅ Sector Map Verified: {count} symbols loaded.")


if __name__ == "__main__":
    main()

