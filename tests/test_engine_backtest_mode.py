import unittest

import pandas as pd

from execution.engine import run_backtest


class EngineBacktestModeTests(unittest.TestCase):
    def test_invalid_backtest_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            run_backtest([], None, backtest_mode="foo")

    def test_rank_mode_requires_rank_inputs(self) -> None:
        with self.assertRaises(ValueError):
            run_backtest([], None, backtest_mode="monthly_rank")

    def test_monthly_rank_mode_runs(self) -> None:
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0 + i * 0.4 for i in range(len(dates))],
                "BBB": [95.0 + i * 0.2 for i in range(len(dates))],
            },
            index=dates,
        )
        scores = pd.DataFrame(
            {
                "AAA": [90.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )
        out = run_backtest(
            [],
            None,
            start_cash=100000.0,
            backtest_mode="monthly_rank",
            rank_prices=prices,
            rank_scores=scores,
            rank_target_count=1,
        )
        self.assertIn("final_value", out)
        self.assertIn("cagr", out)


if __name__ == "__main__":
    unittest.main()
