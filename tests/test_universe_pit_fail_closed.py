import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_with_meta,
)


class UniversePitFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "RUSSELL3000_PIT_DIR": os.environ.get("RUSSELL3000_PIT_DIR"),
            "RUSSELL3000_PIT_MEMBERSHIP_CSV": os.environ.get("RUSSELL3000_PIT_MEMBERSHIP_CSV"),
        }

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_missing_pit_raises_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["RUSSELL3000_PIT_DIR"] = tmpdir
            os.environ["RUSSELL3000_PIT_MEMBERSHIP_CSV"] = ""
            with self.assertRaises(RuntimeError) as context:
                get_universe_symbols_pit_with_meta("RUSSELL3000", "2024-01-15")
            self.assertIn("CRITICAL", str(context.exception))

    def test_incomplete_daily_membership_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pit_dir = Path(tmpdir)
            csv_path = pit_dir / "r3k_2024-01-31.csv"
            csv_path.write_text("ticker\nAAPL\nMSFT\n", encoding="utf-8")
            os.environ["RUSSELL3000_PIT_DIR"] = str(pit_dir)
            os.environ["RUSSELL3000_PIT_MEMBERSHIP_CSV"] = ""

            all_dates = [
                np.datetime64("2024-01-15"),
                np.datetime64("2024-02-01"),
            ]
            membership, source = build_russell3000_membership_by_day(all_dates)
            self.assertEqual(source, "incomplete")
            self.assertEqual(membership, [])

            partial_membership, partial_source = build_russell3000_membership_by_day(
                all_dates,
                allow_missing_days=True,
            )
            self.assertEqual(partial_source, "pit_snapshot_timeline")
            self.assertEqual(len(partial_membership), 2)
            self.assertIsNone(partial_membership[0])
            self.assertIsNotNone(partial_membership[1])


if __name__ == "__main__":
    unittest.main()
