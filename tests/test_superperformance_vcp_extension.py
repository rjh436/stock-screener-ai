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


if __name__ == "__main__":
    unittest.main()
