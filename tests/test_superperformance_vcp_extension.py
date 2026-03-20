import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.superperformance import SuperperformanceStrategy, VCPDetectionResult


class SuperperformanceVCPExtensionTests(unittest.TestCase):
    def _sample_df(self, close_px: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"open": 98.0, "high": 100.0, "low": 97.0, "close": 99.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"open": 100.0, "high": max(close_px + 0.5, 101.0), "low": 99.0, "close": close_px, "volume": 250000.0, "vol_ma50": 100000.0},
            ]
        )

    def _low_cheat_df(self) -> pd.DataFrame:
        rows = []
        for idx in range(8):
            rows.append(
                {
                    "open": 96.0 + idx,
                    "high": 97.0 + idx,
                    "low": 95.0 + idx,
                    "close": 96.5 + idx,
                    "volume": 100000.0,
                    "vol_ma50": 100000.0,
                    "sma10": 100.0,
                    "sma20": 99.0,
                    "clv": 0.7,
                }
            )
        rows.append(
            {
                "open": 105.0,
                "high": 107.5,
                "low": 104.5,
                "close": 107.2,
                "volume": 120000.0,
                "vol_ma50": 100000.0,
                "sma10": 104.0,
                "sma20": 101.0,
                "clv": 0.9,
            }
        )
        return pd.DataFrame(rows)

    @patch("strategies.superperformance.detect_vcp_breakout")
    def test_close_extension_cap_rejects_overextended_breakout(self, mock_detect) -> None:
        mock_detect.return_value = (
            VCPDetectionResult(
                pivot_price=100.0,
                price_contraction_1=10.0,
                price_contraction_2=5.0,
                volume_contraction_1=1.2,
                volume_contraction_2=0.8,
                breakout_volume_multiple=2.5,
            ),
            "ok",
        )
        strategy = SuperperformanceStrategy(
            {
                "breakout_buffer": 0.0005,
                "vcp_close_extension_max_pct": 0.01,
            }
        )

        decision = strategy._vcp_candidate(self._sample_df(102.0), 1)
        self.assertIsNone(decision)
        self.assertEqual(strategy._last_vcp_failure_reason, "breakout_close_too_extended")

    @patch("strategies.superperformance.detect_vcp_breakout")
    def test_close_extension_cap_preserves_normal_breakout(self, mock_detect) -> None:
        mock_detect.return_value = (
            VCPDetectionResult(
                pivot_price=100.0,
                price_contraction_1=10.0,
                price_contraction_2=5.0,
                volume_contraction_1=1.2,
                volume_contraction_2=0.8,
                breakout_volume_multiple=2.5,
            ),
            "ok",
        )
        strategy = SuperperformanceStrategy(
            {
                "breakout_buffer": 0.0005,
                "vcp_close_extension_max_pct": 0.01,
            }
        )

        decision = strategy._vcp_candidate(self._sample_df(100.9), 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "vcp")

    @patch("strategies.superperformance.detect_vcp_breakout")
    def test_low_cheat_candidate_sets_half_size_open_entry(self, mock_detect) -> None:
        mock_detect.return_value = (
            VCPDetectionResult(
                pivot_price=110.0,
                price_contraction_1=12.0,
                price_contraction_2=5.0,
                volume_contraction_1=1.2,
                volume_contraction_2=0.8,
                breakout_volume_multiple=1.1,
                base_low_price=95.0,
                base_high_price=110.0,
                last_contraction_low_price=103.0,
                base_depth_pct=13.6,
                base_span_bars=18,
            ),
            "ok",
        )
        strategy = SuperperformanceStrategy(
            {
                "breakout_buffer": 0.0,
                "low_cheat_size_scalar": 0.5,
                "low_cheat_enabled": True,
                "low_cheat_right_side_min": 0.65,
                "low_cheat_short_pivot_lookback": 5,
                "low_cheat_max_distance_below_pivot_pct": 0.04,
                "low_cheat_min_gap_to_main_pivot_pct": 0.01,
                "low_cheat_max_stop_pct": 0.06,
                "low_cheat_clv_min": 0.55,
                "low_cheat_volume_max_mult": 1.5,
            }
        )

        decision = strategy._low_cheat_candidate(self._low_cheat_df(), 8)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "low_cheat")
        self.assertEqual(decision["signal_mode"], "open")
        self.assertEqual(decision["entry_timing"], "open")
        self.assertAlmostEqual(decision["size_scalar"], 0.5)
        self.assertAlmostEqual(decision["candidate_meta"]["main_pivot_price"], 110.11, places=2)

    def test_low_cheat_pyramid_adds_on_real_pivot_breakout(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "breakout_buffer": 0.0,
                "low_cheat_breakout_add_fraction": 0.5,
                "low_cheat_breakout_add_volume_mult": 1.25,
            }
        )
        df = pd.DataFrame(
            [
                {"close": 108.0, "volume": 100000.0, "vol_ma50": 100000.0},
                {"close": 111.0, "volume": 160000.0, "vol_ma50": 100000.0},
            ]
        )
        position = {
            "entry_price": 107.0,
            "entry_variant": "low_cheat",
            "main_pivot_price": 110.0,
            "pyramids": 0,
        }

        pyramid = strategy.pyramid(df, 1, position)
        self.assertIsNotNone(pyramid)
        self.assertAlmostEqual(pyramid["add_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
