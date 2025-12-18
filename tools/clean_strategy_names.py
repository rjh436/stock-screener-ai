import os
import json
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "generated_strategies.json")


def main() -> None:
    try:
        with open(CONFIG_PATH, "r") as f:
            strategies = json.load(f) or []
    except FileNotFoundError:
        print(f"❌ Config not found: {CONFIG_PATH}")
        return
    except Exception as e:
        print(f"❌ Failed to read config: {e}")
        return

    if not isinstance(strategies, list):
        print(f"❌ Unexpected config format (expected list): {CONFIG_PATH}")
        return

    changed = 0
    for strat in strategies:
        if not isinstance(strat, dict):
            continue
        name = strat.get("name")
        if not isinstance(name, str) or not name:
            continue
        new_name = re.sub(r"(_mut)+", "_mut", name)
        if new_name != name:
            strat["name"] = new_name
            changed += 1

    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(strategies, f, indent=4)
    except Exception as e:
        print(f"❌ Failed to write config: {e}")
        return

    print(f"✅ Cleaned strategy names: {changed} renamed in {CONFIG_PATH}")


if __name__ == "__main__":
    main()
