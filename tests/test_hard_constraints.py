import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.universe import get_universe_symbols
from execution.engine import run_backtest, _audit_track_same_day_open_entry
import numpy as np
import datetime

class TestHardConstraints(unittest.TestCase):
    def test_pit_failure(self):
        """Ensure requesting Russell 3000 without PIT metadata explicitly fails"""
        with self.assertRaises(RuntimeError) as context:
            get_universe_symbols("RUSSELL3000")
        self.assertIn("CRITICAL", str(context.exception))
        self.assertIn("Non-PIT", str(context.exception))

    def test_same_day_entry_failure(self):
        """Ensure same day entry tracking raises explicit ValueError"""
        audit_report = {}
        all_dates = np.array([np.datetime64('2026-02-23T00:00:00')])
        with self.assertRaises(ValueError) as context:
            _audit_track_same_day_open_entry(
                audit_report, 
                symbol="AAPL", 
                day_idx=0, 
                all_dates=all_dates
            )
        self.assertIn("CRITICAL", str(context.exception))
        self.assertIn("Same-day entry contamination", str(context.exception))

if __name__ == '__main__':
    unittest.main()
