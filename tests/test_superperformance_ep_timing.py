import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.superperformance import SuperperformanceStrategy


class SuperperformanceEPEntryTimingTests(unittest.TestCase):
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 108.0, "high": 112.0, "low": 105.0, "close": 111.0, "volume": 450000.0, "vol_ma50": 100000.0},
            ]
        )

    def _low_gap_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 106.5, "high": 110.0, "low": 105.5, "close": 109.5, "volume": 450000.0, "vol_ma50": 100000.0},
            ]
        )

    def _ep_stage2_lag_sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 100000.0,
                    "vol_ma30": 100000.0,
                    "vol_ma50": 100000.0,
                    "sma10": 99.0,
                    "sma20": 98.0,
                    "sma50": 95.0,
                    "sma150": 120.0,
                    "sma200": 130.0,
                    "ret_1m": 20.0,
                    "ret_3m": 25.0,
                    "rs_percentile": 90.0,
                    "adr_pct": 4.0,
                    "eps_growth_yoy": 30.0,
                    "sales_growth_yoy": 25.0,
                },
                {
                    "open": 108.0,
                    "high": 112.0,
                    "low": 107.0,
                    "close": 111.0,
                    "volume": 450000.0,
                    "vol_ma30": 100000.0,
                    "vol_ma50": 100000.0,
                    "sma10": 106.0,
                    "sma20": 103.0,
                    "sma50": 100.0,
                    "sma150": 104.0,
                    "sma200": 128.0,
                    "ret_1m": 20.0,
                    "ret_3m": 25.0,
                    "rs_percentile": 90.0,
                    "adr_pct": 4.5,
                    "eps_growth_yoy": 30.0,
                    "sales_growth_yoy": 25.0,
                },
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

    def test_ep_gap_pct_below_default_is_respected(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.7,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
            }
        )
        decision = strategy._ep_candidate(self._low_gap_sample_df(), 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_timing"], "next_day")

    def test_ep_quality_filters_reject_weak_gap_hold(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.01,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
                "ep_min_clv": 0.0,
                "ep_min_gap_retention": 0.80,
            }
        )
        df = pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 108.0, "high": 108.5, "low": 103.0, "close": 103.8, "volume": 450000.0, "vol_ma50": 100000.0},
            ]
        )

        decision = strategy._ep_candidate(df, 1)
        self.assertIsNone(decision)
        self.assertEqual(strategy._ep_failure_reason(df, 1), "ep:gap_retention=0.47<0.80")
        self.assertEqual(strategy._gate_counts["ep_reject_gap_retention"], 1)

    def test_ep_quality_filters_allow_strong_gap_hold(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.55,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
                "ep_min_clv": 0.65,
                "ep_min_gap_retention": 0.50,
                "ep_require_close_above_open": True,
                "ep_min_close_above_prior_high_pct": 0.005,
            }
        )
        df = pd.DataFrame(
            [
                {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 108.0, "high": 112.0, "low": 107.0, "close": 111.0, "volume": 450000.0, "vol_ma50": 100000.0},
            ]
        )

        decision = strategy._ep_candidate(df, 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_timing"], "next_day")

    def test_check_setup_keeps_ep_on_classic_trend_template_by_default(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "warmup_bars": 1,
                "entry_mode": "ep",
                "trend_template_mode": "classic",
                "rs_gate_min": 70,
                "prior_runup_min_pct": 15,
                "adr_min_pct": 2.5,
                "fundamental_growth_min_pct": 20,
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.7,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
            }
        )
        decision = strategy.check_setup(self._ep_stage2_lag_sample_df(), 1)
        self.assertIsNone(decision)
        self.assertIn("trend_gate classic", strategy.last_reject_reason)

    def test_check_setup_allows_ep_decoupling_from_stage2_lag(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "warmup_bars": 1,
                "entry_mode": "ep",
                "trend_template_mode": "classic",
                "ep_trend_template_mode": "price_above_sma50",
                "rs_gate_min": 70,
                "prior_runup_min_pct": 15,
                "adr_min_pct": 2.5,
                "fundamental_growth_min_pct": 20,
                "ep_gap_pct": 6.0,
                "ep_vol_mult": 3.0,
                "ep_close_near_high_min": 0.7,
                "ep_entry_mode": "close",
                "ep_force_next_day": True,
                "ep_max_stop_pct": 0.2,
            }
        )
        decision = strategy.check_setup(self._ep_stage2_lag_sample_df(), 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "ep")
        self.assertEqual(decision["entry_timing"], "next_day")


if __name__ == "__main__":
    unittest.main()
