import unittest

import pandas as pd

from strategies.superperformance import SuperperformanceStrategy


class SuperperformanceEPEntryTimingTests(unittest.TestCase):
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 108.0, "high": 112.0, "low": 105.0, "close": 111.0, "volume": 450000.0, "vol_ma50": 100000.0},
            ]
        )

    def test_ep_force_next_day_overrides_close_mode(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.8,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
            }
        )
        decision = strategy._ep_candidate(self._sample_df(), 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_timing"], "next_day")
        self.assertEqual(decision["signal_mode"], "after_close")

    def test_ep_force_next_day_overrides_open_mode(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "ep_gap_pct": 6.0,
                "ep_entry_mode": "open",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
            }
        )
        decision = strategy._ep_candidate(self._sample_df(), 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_timing"], "next_day")
        self.assertEqual(decision["signal_mode"], "after_close")


if __name__ == "__main__":
    unittest.main()
