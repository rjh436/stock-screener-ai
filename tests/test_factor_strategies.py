import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy
from strategies.separate_value_momentum import SeparateValueMomentumStrategy


class FactorStrategyTests(unittest.TestCase):
    def test_cross_sectional_momentum_check_setup(self) -> None:
        frame = pd.DataFrame(
            {
                "mom_12_1": [0.30, 0.20, 0.10],
                "mom_6_1": [0.15, 0.10, 0.02],
                "proximity_52w": [0.95, 0.85, 0.70],
                "quality_growth": [0.70, 0.60, 0.30],
            },
            index=["AAA", "BBB", "CCC"],
        )
        strat = CrossSectionalMomentumStrategy({"target_count": 2})
        out = strat.check_setup(frame, date_idx=None, existing_symbols=["CCC"])

        self.assertIn("weights", out)
        self.assertEqual(len(out["selected"]), 2)
        self.assertAlmostEqual(sum(out["weights"].values()), 1.0, places=6)

    def test_separate_value_momentum_build_target(self) -> None:
        frame = pd.DataFrame(
            {
                "mom_12_1": [0.35, 0.12, 0.05, -0.02],
                "mom_6_1": [0.20, 0.10, 0.03, -0.01],
                "proximity_52w": [0.96, 0.80, 0.75, 0.65],
                "ebit_tev": [0.22, 0.18, 0.05, 0.10],
                "pe": [12.0, 18.0, 30.0, 22.0],
                "eps_growth_yoy": [10.0, 8.0, -5.0, 3.0],
                "sales_growth_yoy": [8.0, 6.0, -3.0, 4.0],
            },
            index=["AAA", "BBB", "CCC", "DDD"],
        )
        strat = SeparateValueMomentumStrategy(
            {
                "momentum_target_count": 2,
                "value_target_count": 2,
                "momentum_weight": 0.5,
                "value_weight": 0.5,
            }
        )
        out = strat.build_target(frame, date_idx=None, existing_symbols=["DDD"])

        self.assertIn("weights", out)
        self.assertGreaterEqual(len(out["weights"]), 2)
        self.assertAlmostEqual(sum(out["weights"].values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
