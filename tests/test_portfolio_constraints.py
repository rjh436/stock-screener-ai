import unittest

from execution.portfolio_constraints import (
    enforce_industry_cap,
    enforce_position_cap,
    enforce_sector_cap,
    enforce_turnover_budget,
)


class PortfolioConstraintsTests(unittest.TestCase):
    def test_position_cap_is_enforced(self) -> None:
        weights = {"AAA": 0.70, "BBB": 0.20, "CCC": 0.10}
        capped = enforce_position_cap(weights, 0.40)

        self.assertLessEqual(max(capped.values()), 0.40 + 1e-8)
        self.assertGreater(capped.get("BBB", 0.0), 0.20)

    def test_sector_and_industry_caps(self) -> None:
        weights = {"AAA": 0.60, "BBB": 0.30, "CCC": 0.10}
        sector_map = {"AAA": "TECH", "BBB": "TECH", "CCC": "HEALTH"}
        industry_map = {"AAA": "SOFTWARE", "BBB": "SEMIS", "CCC": "BIOTECH"}

        sector_capped = enforce_sector_cap(weights, sector_map, 0.50)
        industry_capped = enforce_industry_cap(weights, industry_map, 0.55)

        tech_total = sector_capped.get("AAA", 0.0) + sector_capped.get("BBB", 0.0)
        self.assertLessEqual(tech_total, 0.50 + 1e-8)
        self.assertLessEqual(max(industry_capped.values()), 0.55 + 1e-8)

    def test_turnover_budget_blends_weights(self) -> None:
        prev_w = {"AAA": 0.70, "BBB": 0.30}
        next_w = {"AAA": 0.10, "CCC": 0.90}
        constrained = enforce_turnover_budget(prev_w, next_w, turnover_budget=0.20)

        turnover = 0.5 * sum(abs(constrained.get(k, 0.0) - prev_w.get(k, 0.0)) for k in {"AAA", "BBB", "CCC"})
        self.assertLessEqual(turnover, 0.20 + 1e-8)
        self.assertGreater(constrained.get("AAA", 0.0), next_w["AAA"])


if __name__ == "__main__":
    unittest.main()
