import unittest

import pandas as pd

from scripts.optimize_etf_rotation_local import (
    _expand_universe_specs,
    _filter_symbols_by_history,
    _sample_combinations,
)


def _frame(start: str, periods: int = 5) -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame({"close": range(periods), "open": range(periods)}, index=idx)


class OptimizeEtfRotationLocalTests(unittest.TestCase):
    def test_filter_symbols_by_history_enforces_inception_lag(self) -> None:
        data = {
            "AAA": _frame("2011-01-03"),
            "BBB": _frame("2012-07-02"),
            "CCC": _frame("2010-12-01"),
        }
        filtered = _filter_symbols_by_history(
            data,
            ["AAA", "BBB", "CCC"],
            start_date="2011-01-01",
            max_inception_lag_days=365,
        )
        self.assertEqual(filtered, ["AAA", "CCC"])

    def test_sample_combinations_is_deterministic_and_unique(self) -> None:
        symbols = ["A", "B", "C", "D", "E", "F"]
        sample_a = _sample_combinations(symbols, 3, max_subsets=5, seed=11)
        sample_b = _sample_combinations(symbols, 3, max_subsets=5, seed=11)
        self.assertEqual(sample_a, sample_b)
        self.assertEqual(len(sample_a), 5)
        self.assertEqual(len(sample_a), len(set(sample_a)))

    def test_expand_universe_specs_builds_subset_names_from_pool(self) -> None:
        data = {
            "TQQQ": _frame("2010-02-11"),
            "SOXL": _frame("2010-03-11"),
            "TECL": _frame("2009-02-02"),
            "UPRO": _frame("2009-06-25"),
            "TNA": _frame("2009-02-02"),
            "USD": _frame("2009-02-02"),
            "FAS": _frame("2009-02-02"),
            "CURE": _frame("2011-06-15"),
            "DRN": _frame("2009-07-16"),
            "ERX": _frame("2009-02-02"),
            "YINN": _frame("2009-12-03"),
        }
        specs = _expand_universe_specs(
            universe_names=[],
            candidate_pool_names=["leveraged11"],
            subset_sizes=[4],
            max_subsets=3,
            seed=5,
            data=data,
            start_date="2011-01-01",
            max_inception_lag_days=365,
        )
        self.assertEqual(len(specs), 3)
        for name, symbols, pool in specs:
            self.assertTrue(name.startswith("leveraged11_s4_"))
            self.assertEqual(pool, "leveraged11")
            self.assertEqual(len(symbols), 4)


if __name__ == "__main__":
    unittest.main()
