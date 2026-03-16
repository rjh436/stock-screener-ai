import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.allocator import RegimeAllocator


class RegimeAllocatorTests(unittest.TestCase):
    def test_disabled_allocator_is_pass_through(self) -> None:
        allocator = RegimeAllocator({"enabled": False})
        decision = allocator.allocate(
            regime_state="GREEN",
            breadth_50=0.8,
            breadth_200=0.7,
            breadth_rs=0.6,
        )
        self.assertEqual(decision.gross_target, 1.0)
        self.assertEqual(decision.offense_score, 1.0)
        self.assertEqual(decision.sleeve_weights, {"breakout": 1.0})

    def test_enabled_allocator_scales_with_state(self) -> None:
        allocator = RegimeAllocator(
            {
                "enabled": True,
                "min_gross": 0.2,
                "max_gross": 1.0,
                "state_thresholds": {"bull": 0.7, "neutral": 0.45},
            }
        )

        bullish = allocator.allocate(
            regime_state="GREEN",
            breadth_50=0.85,
            breadth_200=0.75,
            breadth_rs=0.70,
        )
        defensive = allocator.allocate(
            regime_state="RED",
            breadth_50=0.15,
            breadth_200=0.20,
            breadth_rs=0.20,
        )

        self.assertGreater(bullish.offense_score, defensive.offense_score)
        self.assertGreater(bullish.gross_target, defensive.gross_target)
        self.assertGreaterEqual(bullish.sleeve_weights.get("breakout", 0.0), 0.40)
        self.assertGreaterEqual(defensive.sleeve_weights.get("recovery", 0.0), 0.30)


if __name__ == "__main__":
    unittest.main()
