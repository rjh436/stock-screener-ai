from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization.optimizer import calculate_fitness


def _equity_curve_from_yearly_returns(yearly_returns: list[float]) -> list[dict[str, float | str]]:
    equity = 100000.0
    rows: list[dict[str, float | str]] = []
    periods = len(yearly_returns) * 12
    dates = pd.date_range("2016-01-31", periods=periods, freq="ME")
    for idx, dt in enumerate(dates):
        annual_return = yearly_returns[idx // 12]
        monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
        equity *= 1.0 + monthly_return
        rows.append({"Date": dt.isoformat(), "Equity": round(equity, 4)})
    return rows


def _base_result() -> dict:
    return {
        "cagr": 0.22,
        "hit_rate": 58.0,
        "max_drawdown_pct": 0.12,
        "total_trades": 96,
        "active_period_years": 8.0,
        "audit_report": {
            "same_day_open_entries": 0,
            "max_gross_exposure_pct": 1.0,
        },
        "equity_curve": _equity_curve_from_yearly_returns([0.18] * 8),
    }


class OptimizerFitnessTests(unittest.TestCase):
    def test_decimal_drawdown_gate_is_normalized(self) -> None:
        result = _base_result()
        result["max_drawdown_pct"] = 0.30
        self.assertEqual(0.0, calculate_fitness(result))

    def test_sparse_trade_profiles_are_rejected(self) -> None:
        result = _base_result()
        result["total_trades"] = 10
        result["active_period_years"] = 5.0
        self.assertEqual(0.0, calculate_fitness(result))

    def test_same_day_contamination_is_rejected(self) -> None:
        result = _base_result()
        result["audit_report"] = {
            "same_day_open_entries": 1,
            "max_gross_exposure_pct": 1.0,
        }
        self.assertEqual(0.0, calculate_fitness(result))

    def test_catastrophic_stitched_fold_is_rejected(self) -> None:
        result = _base_result()
        result["equity_curve"] = _equity_curve_from_yearly_returns([0.20, 0.20, 0.20, -0.50, 0.25, 0.25, 0.25, 0.25])
        self.assertEqual(0.0, calculate_fitness(result))

    def test_robust_profile_scores_positive(self) -> None:
        result = _base_result()
        score = calculate_fitness(result)
        self.assertGreater(score, 0.0)


if __name__ == "__main__":
    unittest.main()
