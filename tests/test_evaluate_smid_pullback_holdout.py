import unittest

from scripts.evaluate_smid_pullback_holdout import (
    _apply_friction_stress,
    _coverage_gate_failures,
    _coverage_ratio,
    _window_metrics,
)
from scripts.evaluate_smid_pullback_validation_matrix import _scenario_bounds


class EvaluateSmidPullbackHoldoutTests(unittest.TestCase):
    def test_window_metrics_include_calmar_and_audit_fields(self) -> None:
        run = {
            "cagr": 0.314,
            "max_drawdown_pct": 0.202,
            "avg_annual_turnover_pct": 912.0,
            "total_trades": 123,
            "final_value": 321000.0,
            "audit_report": {
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 0.98,
            },
        }

        metrics = _window_metrics(run)
        self.assertAlmostEqual(float(metrics["cagr_pct"]), 31.4, places=6)
        self.assertAlmostEqual(float(metrics["max_dd_pct"]), 20.2, places=6)
        self.assertAlmostEqual(float(metrics["calmar"]), 31.4 / 20.2, places=6)
        self.assertEqual(int(metrics["same_day_open_entries"]), 0)
        self.assertAlmostEqual(float(metrics["max_gross_exposure_pct"]), 0.98, places=6)

    def test_coverage_gate_failures_report_both_ratio_and_daily_thresholds(self) -> None:
        failures = _coverage_gate_failures(
            coverage_ratio=0.55,
            coverage_stats={"mean": 0.82, "p10": 0.79},
            min_universe_coverage=0.60,
            min_daily_membership_coverage=0.90,
            min_daily_membership_coverage_p10=0.85,
        )
        self.assertEqual(len(failures), 3)
        self.assertIn("universe_coverage", failures[0])
        self.assertIn("daily_membership_coverage_mean", failures[1])
        self.assertIn("daily_membership_coverage_p10", failures[2])

    def test_apply_friction_stress_scales_and_adds_bps(self) -> None:
        cfg = {
            "friction": {
                "transaction_cost_bps": 15.0,
                "entry_slippage_bps": 10.0,
                "exit_slippage_bps": 12.0,
            }
        }
        stressed = _apply_friction_stress(
            cfg,
            friction_multiplier=2.0,
            extra_transaction_cost_bps=3.0,
            extra_entry_slippage_bps=1.0,
            extra_exit_slippage_bps=2.0,
        )
        friction = dict(stressed["friction"])
        self.assertAlmostEqual(float(friction["transaction_cost_bps"]), 33.0, places=6)
        self.assertAlmostEqual(float(friction["entry_slippage_bps"]), 21.0, places=6)
        self.assertAlmostEqual(float(friction["exit_slippage_bps"]), 26.0, places=6)

    def test_coverage_ratio_handles_requested_symbol_count(self) -> None:
        self.assertAlmostEqual(_coverage_ratio(["A", "B", "C", "D"], ["A", "B"]), 0.5, places=6)
        self.assertEqual(_coverage_ratio([], []), 0.0)

    def test_validation_matrix_scenario_bounds_span_all_splits(self) -> None:
        start, end = _scenario_bounds(
            [
                {"train_start_date": "2018-01-01", "holdout_end_date": "2023-12-31"},
                {"train_start_date": "2016-01-01", "holdout_end_date": "2025-12-31"},
            ]
        )
        self.assertEqual(start, "2016-01-01")
        self.assertEqual(end, "2025-12-31")


if __name__ == "__main__":
    unittest.main()
