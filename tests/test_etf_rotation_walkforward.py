import unittest

import numpy as np
import pandas as pd

from scripts.run_etf_rotation_walkforward import _build_rotation_scores


class EtfRotationWalkforwardTests(unittest.TestCase):
    def test_rotation_scores_favor_stronger_trending_symbol_and_filter_weak_symbol(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="B")
        leader = pd.Series(np.linspace(100.0, 220.0, len(dates)), index=dates)
        laggard = pd.Series(np.linspace(100.0, 80.0, len(dates)), index=dates)
        close_px = pd.DataFrame({"LEAD": leader, "LAG": laggard}, index=dates)

        cfg = {
            "min_price": 5.0,
            "min_rank_names": 2,
            "lookbacks": {"21": 0.5, "63": 0.5},
            "absolute_momentum_lookback": 63,
            "absolute_momentum_min": 0.0,
            "trend_ma_days": 50,
            "fast_ma_days": 20,
            "require_fast_above_trend": True,
            "short_term_reversal_days": 0,
            "volatility_lookback_days": 0,
        }

        scores = _build_rotation_scores(close_px, cfg)
        last = scores.iloc[-1]
        self.assertTrue(np.isfinite(last["LEAD"]))
        self.assertTrue(np.isnan(last["LAG"]))
        self.assertGreater(float(last["LEAD"]), 50.0)


if __name__ == "__main__":
    unittest.main()
