from __future__ import annotations

import os
import sys
import unittest
from unittest import mock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimization.walkforward import (
    WalkforwardContext,
    prepare_walkforward_context,
    parse_friction_values,
    replay_ranked_candidates,
    run_walkforward_batch_reports,
    walkforward_gate_status,
    walkforward_summary,
)


def _report(*, full_dd: float = 28.0, oos36: float = 18.0, oos60: float = 20.0, cash_only: bool = True) -> dict:
    return {
        "scenarios": [
            {
                "scenario": "10x10",
                "full": {
                    "cagr_pct": 19.5,
                    "max_dd_pct": full_dd,
                    "trades": 144,
                    "audit": {
                        "same_day_open_entries": 0,
                        "max_gross_exposure_pct": 1.0,
                    },
                },
                "stitched_36_12": {
                    "stitched_cagr_pct": oos36,
                    "worst_fold_return_pct": -8.0,
                    "fold_count": 7,
                },
                "stitched_60_12": {
                    "stitched_cagr_pct": oos60,
                    "worst_fold_return_pct": -6.0,
                    "fold_count": 5,
                },
                "acceptance": {
                    "cash_only_ok": cash_only,
                    "meets_min_robust_target": min(oos36, oos60) >= 16.0,
                    "meets_stretch_target": min(oos36, oos60) >= 20.0,
                },
            }
        ]
    }


class WalkforwardHelpersTests(unittest.TestCase):
    def test_parse_friction_values_handles_strings_and_sequences(self) -> None:
        self.assertEqual((10.0, 20.0), parse_friction_values("10,20"))
        self.assertEqual((5.0, 15.0), parse_friction_values([5, 15]))

    def test_walkforward_summary_extracts_primary_fields(self) -> None:
        summary = walkforward_summary(_report())
        self.assertEqual("10x10", summary["scenario"])
        self.assertAlmostEqual(19.5, summary["full_cagr_pct"])
        self.assertEqual(144, summary["full_trades"])
        self.assertAlmostEqual(18.0, summary["stitched_36_12_cagr_pct"])
        self.assertTrue(summary["cash_only_ok"])

    def test_walkforward_gate_accepts_robust_report(self) -> None:
        ok, reason, summary = walkforward_gate_status(_report())
        self.assertTrue(ok)
        self.assertEqual("", reason)
        self.assertAlmostEqual(20.0, summary["stitched_60_12_cagr_pct"])

    def test_walkforward_gate_rejects_full_drawdown_breach(self) -> None:
        ok, reason, _ = walkforward_gate_status(_report(full_dd=39.4))
        self.assertFalse(ok)
        self.assertIn("max drawdown", reason)

    def test_walkforward_gate_rejects_36_12_shortfall(self) -> None:
        ok, reason, _ = walkforward_gate_status(_report(oos36=14.8, oos60=22.8))
        self.assertFalse(ok)
        self.assertIn("36/12", reason)

    def test_walkforward_gate_rejects_cash_only_failure(self) -> None:
        ok, reason, _ = walkforward_gate_status(_report(cash_only=False))
        self.assertFalse(ok)
        self.assertIn("cash-only", reason)

    @mock.patch("optimization.walkforward.run_walkforward_report_with_context")
    @mock.patch("optimization.walkforward.prepare_walkforward_context")
    def test_replay_ranked_candidates_demotes_failed_replays(
        self,
        mock_prepare_context: mock.Mock,
        mock_run_with_context: mock.Mock,
    ) -> None:
        good_report = _report(oos36=18.0, oos60=21.0, full_dd=28.0)
        bad_report = _report(oos36=14.5, oos60=20.0, full_dd=39.0)
        mock_prepare_context.return_value = WalkforwardContext(
            start_date="2016-01-01",
            end_date="2026-03-15",
            symbols=("AAA", "BBB"),
            source="pit_snapshot_window",
            membership_source="pit_snapshot_timeline",
            universe_limit_mode="sample",
            universe_sample_seed=42,
            prepared_data_source="live_fetch",
            prepared=object(),
            global_data={},
            membership_by_day=[],
            windows_36_12=(("2019-01-01", "2020-01-01"),),
            windows_60_12=(("2021-01-01", "2022-01-01"),),
        )
        mock_run_with_context.side_effect = [bad_report, good_report]
        rows = [
            {"name": "A", "config": {"name": "A"}, "score": 100.0},
            {"name": "B", "config": {"name": "B"}, "score": 90.0},
            {"name": "C", "config": {"name": "C"}, "score": 80.0},
        ]
        ranked = replay_ranked_candidates(rows, top_n=2, friction_values=(10.0,))
        mock_prepare_context.assert_called_once()
        self.assertEqual(2, mock_run_with_context.call_count)
        self.assertEqual("B", ranked[0]["name"])
        self.assertTrue(ranked[0]["walkforward_ok"])
        self.assertEqual("C", ranked[1]["name"])
        self.assertFalse(ranked[2]["walkforward_ok"])
        self.assertLess(ranked[2]["score"], -1_000_000.0)

    @mock.patch("optimization.walkforward.run_walkforward_report_with_context")
    @mock.patch("optimization.walkforward.prepare_walkforward_context")
    def test_run_walkforward_batch_reports_reuses_context(
        self,
        mock_prepare_context: mock.Mock,
        mock_run_with_context: mock.Mock,
    ) -> None:
        context = WalkforwardContext(
            start_date="2016-01-01",
            end_date="2026-03-15",
            symbols=("AAA",),
            source="pit_snapshot_window",
            membership_source="pit_snapshot_timeline",
            universe_limit_mode="sample",
            universe_sample_seed=42,
            prepared_data_source="live_fetch",
            prepared=object(),
            global_data={},
            membership_by_day=[],
            windows_36_12=(("2019-01-01", "2020-01-01"),),
            windows_60_12=(("2021-01-01", "2022-01-01"),),
        )
        mock_prepare_context.return_value = context
        mock_run_with_context.side_effect = [
            {"config_path": "a.json", "scenarios": []},
            {"config_path": "b.json", "scenarios": []},
        ]
        reports = run_walkforward_batch_reports(
            [{"name": "A"}, {"name": "B"}],
            config_paths=["a.json", "b.json"],
        )
        mock_prepare_context.assert_called_once()
        self.assertEqual(2, mock_run_with_context.call_count)
        self.assertEqual(["a.json", "b.json"], [r["config_path"] for r in reports])

    @mock.patch("optimization.walkforward.build_russell3000_membership_by_day")
    @mock.patch("optimization.walkforward.fetch_data_pack")
    @mock.patch("optimization.walkforward.load_prepared_cache")
    @mock.patch("optimization.walkforward.get_universe_symbols_pit_window_with_meta")
    def test_prepare_walkforward_context_uses_prepared_cache_when_available(
        self,
        mock_get_universe: mock.Mock,
        mock_load_cache: mock.Mock,
        mock_fetch_pack: mock.Mock,
        mock_build_membership: mock.Mock,
    ) -> None:
        mock_get_universe.return_value = (["AAA", "BBB"], "pit_snapshot_window")
        mock_fetch_pack.return_value = {"SPY": object()}
        mock_load_cache.return_value = mock.Mock(
            enriched={"AAA": object()},
            all_dates=np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
        )
        mock_build_membership.return_value = ([{"AAA"}, {"AAA"}], "pit_snapshot_timeline")
        context = prepare_walkforward_context(
            start_date="2024-01-01",
            end_date="2024-12-31",
            prepared_cache_path="data/cache_indicators.pkl",
        )
        mock_load_cache.assert_called_once()
        mock_fetch_pack.assert_called_once()
        self.assertEqual("pit_snapshot_window", context.source)
        self.assertEqual("pit_snapshot_timeline", context.membership_source)
        self.assertTrue(str(context.prepared_data_source).endswith("data/cache_indicators.pkl"))
        self.assertEqual(("AAA",), context.symbols)


if __name__ == "__main__":
    unittest.main()
