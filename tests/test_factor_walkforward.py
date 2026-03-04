import unittest

import numpy as np
import pandas as pd

from scripts.run_factor_walkforward import (
    _acceptance_snapshot,
    _build_separate_value_momentum_schedule,
    _cross_section_rank_matrix,
)


class FactorWalkforwardTests(unittest.TestCase):
    def test_cross_section_rank_matrix(self) -> None:
        vals = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        valid = np.array([[True, True, True], [True, True, True]], dtype=bool)
        out = _cross_section_rank_matrix(vals, valid_mask=valid, higher_is_better=True, min_names=2)

        self.assertEqual(out.shape, vals.shape)
        self.assertGreater(float(out[0, 2]), float(out[0, 0]))
        self.assertGreater(float(out[1, 0]), float(out[1, 1]))

    def test_build_separate_schedule_outputs_weights(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        mom = pd.DataFrame(
            {
                "AAA": [90.0] * len(idx),
                "BBB": [80.0] * len(idx),
                "CCC": [70.0] * len(idx),
            },
            index=idx,
        )
        val = pd.DataFrame(
            {
                "AAA": [60.0] * len(idx),
                "BBB": [75.0] * len(idx),
                "DDD": [85.0] * len(idx),
            },
            index=idx,
        )

        sched = _build_separate_value_momentum_schedule(
            mom,
            val,
            rebalance_freq="M",
            momentum_count=2,
            value_count=2,
            hold_buffer_mult=1.25,
            momentum_weight=0.5,
            value_weight=0.5,
        )
        self.assertFalse(sched.empty)
        for _, row in sched.iterrows():
            total = float(pd.to_numeric(row, errors="coerce").fillna(0.0).sum())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_acceptance_snapshot_checks_same_day_contamination(self) -> None:
        full = {"cagr": 0.12, "max_drawdown_pct": 0.25, "audit_report": {"max_gross_exposure_pct": 1.0, "same_day_open_entries": 2}}
        stitched = {"stitched_cagr_pct": 11.0}
        snap = _acceptance_snapshot(full, stitched, stitched)
        self.assertFalse(bool(snap.get("same_day_contamination_ok")))
        self.assertEqual(int(snap.get("same_day_contamination_count", 0)), 2)


if __name__ == "__main__":
    unittest.main()
