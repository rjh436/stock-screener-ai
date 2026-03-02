import unittest

import numpy as np

from execution.engine import (
    _audit_track_gross_exposure,
    _audit_track_same_day_open_entry,
    _audit_track_stale_position_event,
    _entry_timing_label,
    _finalize_backtest_audit_report,
    _init_backtest_audit_report,
)


class EngineAuditReportTests(unittest.TestCase):
    def test_entry_timing_label_marks_same_day_open(self) -> None:
        label = _entry_timing_label(
            entry_i=10,
            prev_i=10,
            curr_i=11,
            signal_mode="open",
        )
        self.assertEqual(label, "same_day_open")

    def test_entry_timing_label_marks_next_day_open(self) -> None:
        label = _entry_timing_label(
            entry_i=11,
            prev_i=10,
            curr_i=11,
            signal_mode="open",
        )
        self.assertEqual(label, "next_day_open")

    def test_same_day_open_raises_and_stale_events_are_counted(self) -> None:
        audit = _init_backtest_audit_report()
        dates = np.array(
            [
                np.datetime64("2026-02-20"),
                np.datetime64("2026-02-23"),
            ],
            dtype="datetime64[ns]",
        )

        with self.assertRaises(ValueError) as context:
            _audit_track_same_day_open_entry(audit, symbol="abcd", day_idx=0, all_dates=dates)
        self.assertIn("CRITICAL", str(context.exception))
        _audit_track_stale_position_event(audit, symbol="wxyz", day_idx=1, all_dates=dates)

        final = _finalize_backtest_audit_report(audit)
        self.assertEqual(final["same_day_open_entries"], 0)
        self.assertEqual(final["same_day_open_symbol_count"], 0)
        self.assertEqual(final["stale_position_days"], 1)
        self.assertEqual(final["stale_position_symbol_count"], 1)
        self.assertNotIn("ABCD", final["same_day_open_symbols"])
        self.assertIn("WXYZ", final["stale_position_symbols"])

    def test_gross_exposure_audit_tracks_daily_and_max(self) -> None:
        audit = _init_backtest_audit_report()
        dates = np.array(
            [
                np.datetime64("2026-02-20"),
                np.datetime64("2026-02-23"),
            ],
            dtype="datetime64[ns]",
        )

        _audit_track_gross_exposure(
            audit,
            day_idx=0,
            all_dates=dates,
            gross_exposure=100000.0,
            mtm_equity=120000.0,
        )
        _audit_track_gross_exposure(
            audit,
            day_idx=1,
            all_dates=dates,
            gross_exposure=180000.0,
            mtm_equity=150000.0,
        )

        final = _finalize_backtest_audit_report(audit)
        self.assertEqual(len(final["gross_exposure_daily"]), 2)
        self.assertAlmostEqual(final["max_gross_exposure_notional"], 180000.0)
        self.assertAlmostEqual(final["max_gross_exposure_pct"], 1.2)
        self.assertEqual(final["max_gross_exposure_date"], "2026-02-23")


if __name__ == "__main__":
    unittest.main()
