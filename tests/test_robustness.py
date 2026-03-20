from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization.robustness import (
    baseline_promotion_status,
    build_summary_metrics,
    promotion_gate_status,
    rejection_reason,
)


class RobustnessHelpersTests(unittest.TestCase):
    def test_build_summary_metrics_computes_trade_rate(self) -> None:
        metrics = build_summary_metrics(
            cagr_pct=18.0,
            max_dd_pct=12.0,
            total_trades=60,
            sample_years=5.0,
        )
        self.assertEqual(60, metrics["total_trades"])
        self.assertAlmostEqual(12.0, metrics["trades_per_year"])
        self.assertAlmostEqual(12.0, metrics["max_dd_pct"])

    def test_promotion_gate_rejects_sparse_profile(self) -> None:
        metrics = build_summary_metrics(
            cagr_pct=25.0,
            max_dd_pct=10.0,
            total_trades=10,
            sample_years=5.0,
        )
        ok, code, reason = promotion_gate_status(metrics, metrics_5y=metrics, metrics_10y={})
        self.assertFalse(ok)
        self.assertEqual(200, code)
        self.assertIn("trade-count floor", reason)

    def test_promotion_gate_allows_robust_profile(self) -> None:
        metrics = build_summary_metrics(
            cagr_pct=22.0,
            max_dd_pct=14.0,
            total_trades=72,
            sample_years=5.0,
            stitched_oos_cagr_pct=9.0,
            stitched_oos_folds=3,
            stitched_oos_worst_fold_return_pct=-12.0,
            worst_24m_cagr_pct=6.0,
            median_24m_cagr_pct=15.0,
            top2_year_pnl_share=0.62,
        )
        ok, code, reason = promotion_gate_status(metrics, metrics_5y=metrics, metrics_10y={})
        self.assertTrue(ok)
        self.assertIsNone(code)
        self.assertEqual("", reason)

    def test_rejection_reason_handles_unknown_and_none(self) -> None:
        self.assertEqual("", rejection_reason(None))
        self.assertEqual("rejection code 999", rejection_reason(999))

    def test_baseline_promotion_status_requires_gate_and_score_edge(self) -> None:
        baseline = build_summary_metrics(
            cagr_pct=18.0,
            max_dd_pct=14.0,
            total_trades=72,
            sample_years=5.0,
        )
        baseline["label"] = "BASELINE"
        baseline["score"] = 20.0

        champion = build_summary_metrics(
            cagr_pct=24.0,
            max_dd_pct=16.0,
            total_trades=84,
            sample_years=5.0,
        )
        champion["label"] = "FAST_PYR"
        champion["score"] = 25.0

        should_apply, reason = baseline_promotion_status(
            champion,
            baseline,
            score_key="score",
            max_drawdown_gate_pct=30.0,
        )
        self.assertTrue(should_apply)
        self.assertEqual("", reason)

        champion["score"] = 19.0
        should_apply, reason = baseline_promotion_status(
            champion,
            baseline,
            score_key="score",
            max_drawdown_gate_pct=30.0,
        )
        self.assertFalse(should_apply)
        self.assertIn("did not beat baseline", reason)


if __name__ == "__main__":
    unittest.main()
