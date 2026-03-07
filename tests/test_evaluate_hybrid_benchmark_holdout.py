import unittest

import pandas as pd

from scripts.evaluate_hybrid_benchmark_holdout import _blend_equity_series, _metrics_from_equity


class EvaluateHybridBenchmarkHoldoutTests(unittest.TestCase):
    def test_blend_equity_series_averages_component_returns(self) -> None:
        idx = pd.to_datetime(["2021-01-01", "2021-01-04", "2021-01-05"])
        etf_eq = pd.Series([100.0, 110.0, 121.0], index=idx)
        stock_eq = pd.Series([100.0, 100.0, 110.0], index=idx)
        blended = _blend_equity_series(etf_eq, stock_eq, etf_weight=0.5)
        self.assertAlmostEqual(float(blended.iloc[0]), 100000.0, places=6)
        self.assertAlmostEqual(float(blended.iloc[1]), 105000.0, places=6)
        self.assertAlmostEqual(float(blended.iloc[2]), 115500.0, places=6)

    def test_metrics_from_equity_reports_cagr_and_drawdown(self) -> None:
        idx = pd.to_datetime(["2021-01-01", "2021-07-01", "2021-12-31"])
        equity = pd.Series([100000.0, 90000.0, 121000.0], index=idx)
        metrics = _metrics_from_equity(equity)
        self.assertAlmostEqual(float(metrics["max_dd_pct"]), 10.0, places=6)
        self.assertGreater(float(metrics["cagr_pct"]), 20.0)


if __name__ == "__main__":
    unittest.main()
