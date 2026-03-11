from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_alpha_targeted_grid import _normalize, _one_step_neighbors, _resolve_seed_args, _write_checkpoint


class EvaluateAlphaTargetedGridTests(unittest.TestCase):
    def test_resolve_seed_args_defaults_when_empty(self) -> None:
        self.assertEqual(["a.json"], _resolve_seed_args(["a.json"]))
        resolved = _resolve_seed_args([])
        self.assertGreaterEqual(len(resolved), 1)
        self.assertIn("config/superperformance_alpha_b9.json", resolved)

    def test_one_step_neighbors_generates_distinct_variants(self) -> None:
        seed = _normalize({
            "name": "Alpha",
            "type": "superperformance",
            "rs_gate_min": 85,
            "min_entry_score": 25,
            "vcp_lookback_bars": 30,
            "breakout_buffer": 0.001,
            "max_stop_pct": 0.06,
            "stop_limit_pct": 0.03,
            "ep_gap_pct": 6,
            "ep_close_near_high_min": 0.55,
            "pyramid_threshold": 0.08,
            "pyramid_fraction": 0.5,
            "vcp_gap_chase_max_pct": 0.01,
            "vcp_gap_chase_rs_min": 95,
        })
        neighbors = _one_step_neighbors(seed)
        self.assertGreater(len(neighbors), 0)
        self.assertTrue(any(n.get("rs_gate_min") != 85 for n in neighbors))
        self.assertTrue(any(n.get("min_entry_score") != 25 for n in neighbors))

    def test_normalize_clamps_ranges(self) -> None:
        cfg = _normalize({
            "name": "Alpha",
            "type": "superperformance",
            "max_positions": 99,
            "max_total_exposure_pct_bull": 2.0,
            "max_pos_size_pct": 1.0,
            "market_exposure_mode": "bad",
            "bear_cash_mode": "hard",
        })
        self.assertLessEqual(cfg["max_positions"], 4)
        self.assertLessEqual(cfg["max_total_exposure_pct_bull"], 1.0)
        self.assertLessEqual(cfg["max_pos_size_pct"], cfg["max_total_exposure_pct_bull"] / cfg["max_positions"])
        self.assertEqual("exposure", cfg["market_exposure_mode"])
        self.assertEqual(0.0, cfg["max_total_exposure_pct_bear"])
        self.assertEqual(0, cfg["bear_max_positions"])

    def test_write_checkpoint_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "checkpoint.json"
            _write_checkpoint(path, {"membership_source": "pit"}, [{"name": "b", "score": 1}, {"name": "a", "score": 2}])
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("a", payload["rows"][0]["name"])
            self.assertEqual("pit", payload["membership_source"])


if __name__ == "__main__":
    unittest.main()
