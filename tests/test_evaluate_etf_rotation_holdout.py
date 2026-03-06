import unittest

from scripts.evaluate_etf_rotation_holdout import _window_metrics


class EvaluateEtfRotationHoldoutTests(unittest.TestCase):
    def test_window_metrics_include_audit_fields(self) -> None:
        run = {
            "cagr": 0.1234,
            "max_drawdown_pct": 0.215,
            "avg_annual_turnover_pct": 155.5,
            "total_trades": 42,
            "final_value": 178900.0,
            "audit_report": {
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 0.618,
            },
        }

        metrics = _window_metrics(run)
        self.assertAlmostEqual(float(metrics["cagr_pct"]), 12.34, places=6)
        self.assertAlmostEqual(float(metrics["max_dd_pct"]), 21.5, places=6)
        self.assertEqual(int(metrics["same_day_open_entries"]), 0)
        self.assertAlmostEqual(float(metrics["max_gross_exposure_pct"]), 0.618, places=6)


if __name__ == "__main__":
    unittest.main()
