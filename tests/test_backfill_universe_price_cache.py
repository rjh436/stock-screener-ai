import unittest

from scripts.backfill_universe_price_cache import _select_target_symbols


class BackfillUniversePriceCacheTests(unittest.TestCase):
    def test_select_target_symbols_dedupes_requested_modes(self) -> None:
        report = {
            "missing_symbols": ["AAA", "BBB", "CCC"],
            "incomplete_symbols": ["BBB", "DDD"],
            "stale_symbols": ["EEE", "AAA"],
        }
        selected = _select_target_symbols(report, ["missing", "incomplete", "stale"])
        self.assertEqual(selected, ["AAA", "BBB", "CCC", "DDD", "EEE"])


if __name__ == "__main__":
    unittest.main()
