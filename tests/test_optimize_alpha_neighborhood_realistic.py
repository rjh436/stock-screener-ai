from __future__ import annotations

import unittest

from scripts.optimize_alpha_neighborhood_realistic import _normalize, _resolve_seed_args


class OptimizeAlphaNeighborhoodRealisticTests(unittest.TestCase):
    def test_resolve_seed_args_uses_defaults_only_when_empty(self) -> None:
        self.assertEqual(["a.json"], _resolve_seed_args(["a.json"]))
        resolved = _resolve_seed_args([])
        self.assertGreaterEqual(len(resolved), 1)
        self.assertIn("config/superperformance_alpha_b9.json", resolved)

    def test_normalize_enforces_execution_and_position_caps(self) -> None:
        cfg = _normalize(
            {
                "name": "Alpha",
                "type": "superperformance",
                "max_positions": 99,
                "max_total_exposure_pct_bull": 2.0,
                "max_pos_size_pct": 1.0,
                "bear_cash_mode": "hard",
                "trend_template_mode": "weird",
            }
        )
        self.assertEqual("after_close", cfg["signal_mode"])
        self.assertTrue(cfg["enforce_next_day_exit_execution"])
        self.assertFalse(cfg["allow_margin"])
        self.assertLessEqual(cfg["max_positions"], 6)
        self.assertLessEqual(cfg["max_total_exposure_pct_bull"], 1.0)
        self.assertLessEqual(cfg["max_pos_size_pct"], cfg["max_total_exposure_pct_bull"] / cfg["max_positions"])
        self.assertEqual(0.0, cfg["max_total_exposure_pct_bear"])
        self.assertEqual(0, cfg["bear_max_positions"])
        self.assertEqual("classic", cfg["trend_template_mode"])


if __name__ == "__main__":
    unittest.main()
