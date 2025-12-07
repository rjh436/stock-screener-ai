import json
from pathlib import Path


def migrate_portfolio_names() -> None:
    """Update legacy strategy names in the paper portfolio file."""
    portfolio_path = Path("data/paper_portfolio.json")
    with portfolio_path.open("r", encoding="utf-8") as f:
        portfolio = json.load(f)

    replacements = {
        "Strategy_Apex_Gen12_Sniper": "Apex Wealth (Gen 12)",
        "Strategy_Apex_Gen9_Evolved": "Apex Income (Gen 9)",
    }

    positions = portfolio.get("positions", {})
    for position in positions.values():
        strategy = position.get("strategy")
        if not isinstance(strategy, str):
            continue
        updated = strategy
        for old, new in replacements.items():
            if old in updated:
                updated = updated.replace(old, new)
        position["strategy"] = updated

    with portfolio_path.open("w", encoding="utf-8") as f:
        json.dump(portfolio, f, indent=2)
        f.write("\n")

    print("✅ Portfolio strategy names migrated successfully.")


if __name__ == "__main__":
    migrate_portfolio_names()
