import unittest

import numpy as np
import pandas as pd

from execution.rank_scoring import (
    combine_rank_columns,
    compute_low_vol_rank,
    compute_momentum_rank,
    compute_quality_rank,
    compute_value_rank,
)


class RankScoringTests(unittest.TestCase):
    def test_momentum_rank_orders_symbols(self) -> None:
        frame = pd.DataFrame(
            {
                "mom_12_1": [0.45, 0.20, 0.05],
                "mom_6_1": [0.20, 0.10, -0.02],
                "proximity_52w": [0.96, 0.82, 0.65],
                "quality_growth": [0.75, 0.55, 0.30],
            },
            index=["AAA", "BBB", "CCC"],
        )

        ranked = compute_momentum_rank(frame)
        self.assertIn("momentum_rank", ranked.columns)
        self.assertEqual(list(ranked.index[:2]), ["AAA", "BBB"])
        self.assertGreater(float(ranked.loc["AAA", "momentum_rank"]), float(ranked.loc["CCC", "momentum_rank"]))

    def test_value_and_quality_rank_build_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "ebit_tev": [0.25, 0.12, 0.05],
                "pe": [10.0, 20.0, 35.0],
                "roe": [0.30, 0.15, 0.05],
                "debt_to_equity": [0.2, 0.8, 1.8],
            },
            index=["AAA", "BBB", "CCC"],
        )
        value_rank = compute_value_rank(frame)
        quality_rank = compute_quality_rank(frame)

        self.assertIn("value_rank", value_rank.columns)
        self.assertIn("quality_rank", quality_rank.columns)
        self.assertEqual(value_rank.index[0], "AAA")
        self.assertEqual(quality_rank.index[0], "AAA")

    def test_value_rank_requires_min_factor_count(self) -> None:
        frame = pd.DataFrame(
            {
                "ebit_tev": [0.30, np.nan, np.nan],
                "pe": [9.0, 12.0, np.nan],
                "pb": [1.2, np.nan, np.nan],
            },
            index=["AAA", "BBB", "CCC"],
        )
        ranked = compute_value_rank(frame, params={"min_value_factors": 2})
        self.assertTrue(np.isfinite(float(ranked.loc["AAA", "value_rank"])))
        self.assertTrue(np.isnan(float(ranked.loc["BBB", "value_rank"])))
        self.assertTrue(np.isnan(float(ranked.loc["CCC", "value_rank"])))

    def test_combine_rank_columns_uses_weights(self) -> None:
        frame = pd.DataFrame(
            {
                "momentum_rank": [90.0, 60.0, 30.0],
                "value_rank": [20.0, 70.0, 50.0],
            },
            index=["AAA", "BBB", "CCC"],
        )
        combined = combine_rank_columns(frame, {"momentum_rank": 0.7, "value_rank": 0.3})

        self.assertEqual(combined.name, "combined_rank")
        self.assertTrue(np.isfinite(combined.loc["AAA"]))
        self.assertGreater(float(combined.loc["AAA"]), float(combined.loc["CCC"]))

    def test_momentum_quality_proxy_is_ranked_per_column(self) -> None:
        frame = pd.DataFrame(
            {
                "mom_12_1": [0.1, 0.1],
                "mom_6_1": [0.1, 0.1],
                "proximity_52w": [0.9, 0.9],
                # Large scale mismatch should not dominate once per-column ranking is applied.
                "eps_growth_yoy": [10000.0, 100.0],
                "sales_growth_yoy": [0.0, 100.0],
                "f_score": [0.0, 9.0],
            },
            index=["AAA", "BBB"],
        )
        ranked = compute_momentum_rank(
            frame,
            params={
                "mom_12_1_weight": 0.0,
                "mom_6_1_weight": 0.0,
                "proximity_52w_weight": 0.0,
                "quality_growth_weight": 1.0,
            },
        )
        self.assertEqual(ranked.index[0], "BBB")

    def test_low_vol_rank_prefers_lower_volatility_metrics(self) -> None:
        frame = pd.DataFrame(
            {
                "realized_vol_63d": [0.12, 0.45, 0.30],
                "natr": [1.5, 4.5, 3.0],
                "adr": [2.0, 5.0, 3.5],
            },
            index=["LOW", "HIGH", "MID"],
        )
        ranked = compute_low_vol_rank(frame)
        self.assertIn("low_vol_rank", ranked.columns)
        self.assertEqual(ranked.index[0], "LOW")


if __name__ == "__main__":
    unittest.main()
