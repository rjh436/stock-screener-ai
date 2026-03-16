import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

    def test_turnover_budget_priority_cleans_obsolete_tail(self) -> None:
        prev_w = {"AAA": 0.02, "BBB": 0.49, "CCC": 0.49}
        next_w = {"BBB": 0.50, "CCC": 0.50}
        constrained = enforce_turnover_budget(
            prev_w,
            next_w,
            turnover_budget=0.02,
            mode="priority",
            priority_symbols=["BBB", "CCC"],
            cleanup_weight_floor=0.03,
        )

        self.assertNotIn("AAA", constrained)
        turnover = 0.5 * sum(abs(constrained.get(k, 0.0) - prev_w.get(k, 0.0)) for k in {"AAA", "BBB", "CCC"})
        self.assertLessEqual(turnover, 0.02 + 1e-8)

    def test_turnover_budget_priority_prefers_targets_over_legacy_names(self) -> None:
        prev_w = {"AAA": 0.50, "BBB": 0.50}
        next_w = {"BBB": 0.50, "CCC": 0.50}
        constrained = enforce_turnover_budget(
            prev_w,
            next_w,
            turnover_budget=0.10,
            mode="priority",
            priority_symbols=["BBB", "CCC"],
        )
        self.assertGreater(constrained.get("CCC", 0.0), 0.0)
        self.assertLess(constrained.get("AAA", 0.0), 0.50)


if __name__ == "__main__":
    unittest.main()
