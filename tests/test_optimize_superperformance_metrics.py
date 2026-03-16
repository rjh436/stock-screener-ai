import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optimize_superperformance as opt


class OptimizeSuperperformanceMetricTests(unittest.TestCase):
    def _synthetic_equity_curve(self, annual_cagr: float = 0.20, years: int = 8):
        dates = pd.date_range("2016-01-31", periods=years * 12, freq="ME")
        monthly_rate = (1.0 + annual_cagr) ** (1.0 / 12.0) - 1.0
        equity = 100000.0
        rows = []
        for dt in dates:
            equity *= (1.0 + monthly_rate)
            rows.append({"Date": dt, "Equity": equity})
        return rows

    def test_calendar_year_returns(self):
        eq = [
            {"Date": "2020-01-02", "Equity": 100.0},
            {"Date": "2020-12-31", "Equity": 120.0},
            {"Date": "2021-01-04", "Equity": 120.0},
            {"Date": "2021-12-31", "Equity": 138.0},
        ]
        yearly = opt._calendar_year_returns_pct(eq)
        self.assertIn(2020, yearly)
        self.assertIn(2021, yearly)
        self.assertAlmostEqual(yearly[2020], 20.0, places=6)
        self.assertAlmostEqual(yearly[2021], 15.0, places=6)

    def test_year_concentration_stats(self):
        eq = [
            {"Date": "2020-01-02", "Equity": 100.0},
            {"Date": "2020-12-31", "Equity": 160.0},
            {"Date": "2021-01-04", "Equity": 160.0},
            {"Date": "2021-12-31", "Equity": 168.0},
            {"Date": "2022-01-03", "Equity": 168.0},
            {"Date": "2022-12-30", "Equity": 151.2},
        ]
        stats = opt._year_concentration_stats(eq)
        self.assertGreater(stats["max_positive_share"], 0.80)
        self.assertEqual(stats["negative_years"], 1)
        self.assertEqual(stats["total_years"], 3)

    def test_rolling_and_stitched_oos_metrics(self):
        eq = self._synthetic_equity_curve(annual_cagr=0.20, years=8)
        worst_24m, median_24m, samples = opt._rolling_window_cagr_stats(eq, window_years=2, step_days=30)
        self.assertTrue(np.isfinite(worst_24m))
        self.assertTrue(np.isfinite(median_24m))
        self.assertGreater(samples, 1)
        self.assertGreater(worst_24m, 10.0)
        self.assertGreater(median_24m, 10.0)

        stitched_cagr, folds, worst_fold = opt._stitched_oos_cagr_pct(eq, train_years=3, test_years=1)
        self.assertTrue(np.isfinite(stitched_cagr))
        self.assertGreater(folds, 1)
        self.assertGreater(stitched_cagr, 10.0)
        self.assertTrue(np.isfinite(worst_fold))


if __name__ == "__main__":
    unittest.main()
