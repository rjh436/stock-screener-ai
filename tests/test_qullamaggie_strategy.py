from __future__ import annotations

import unittest

import pandas as pd

from strategies.qullamaggie import QullamaggieStrategy


def _base_df() -> pd.DataFrame:
    rows = []
    for i in range(220):
        rows.append(
            {
                "close": 95.0 + (i * 0.1),
                "open": 94.5 + (i * 0.1),
                "high": 96.0 + (i * 0.1),
                "low": 94.0 + (i * 0.1),
                "volume": 100000.0,
                "vol_ma10": 120000.0,
                "vol_ma20": 120000.0,
                "vol_ma30": 300000.0,
                "vol_ma50": 140000.0,
                "sma20": 100.0,
                "sma50": 98.0,
                "sma150": 92.0,
                "sma200": 90.0,
                "spy_close": 500.0,
                "spy_sma200": 450.0,
                "high_52w": 110.0,
                "rs_rating": 95.0,
                "momentum_rank": 97.0,
                "adr_pct": 0.04,
                "bb_width": 0.12,
                "high_20_prev": 105.0,
                "range_pct_5": 2.0,
                "range_pct_10": 4.0,
                "range_pct_20": 7.0,
                "range_pct_40": 12.0,
                "clv": 0.7,
                "gap_pct": 0.02,
            }
        )
    return pd.DataFrame(rows)


class QullamaggieStrategyTests(unittest.TestCase):
    def test_breakout_candidate_builds_near_pivot(self) -> None:
        df = _base_df()
        df.iloc[-1] = {
            **df.iloc[-1].to_dict(),
            "close": 104.7,
            "open": 104.1,
            "high": 104.95,
            "low": 103.8,
            "high_20_prev": 105.0,
            "clv": 0.8,
            "volume": 95000.0,
            "vol_ma10": 110000.0,
            "vol_ma20": 120000.0,
            "vol_ma50": 150000.0,
        }
        strat = QullamaggieStrategy({"name": "QM", "type": "qullamaggie", "entry_mode": "breakout"})
        decision = strat.entry(df, len(df) - 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "vcp")
        self.assertGreater(decision["trigger_price"], df.iloc[-1]["close"])
        self.assertLess(decision["stop_price"], decision["trigger_price"])

    def test_ep_candidate_builds_for_gap_day(self) -> None:
        df = _base_df()
        df.iloc[-1] = {
            **df.iloc[-1].to_dict(),
            "close": 112.0,
            "open": 108.0,
            "high": 113.0,
            "low": 107.0,
            "gap_pct": 0.12,
            "volume": 450000.0,
            "vol_ma50": 120000.0,
            "sma20": 103.0,
            "clv": 0.92,
            "high_52w": 113.0,
        }
        strat = QullamaggieStrategy({"name": "QM", "type": "qullamaggie", "entry_mode": "ep"})
        decision = strat.entry(df, len(df) - 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "ep")
        self.assertGreaterEqual(decision["trigger_price"], df.iloc[-1]["high"])

    def test_rejects_when_market_not_in_uptrend(self) -> None:
        df = _base_df()
        df.iloc[-1] = {
            **df.iloc[-1].to_dict(),
            "close": 104.7,
            "open": 104.1,
            "high": 104.95,
            "low": 103.8,
            "high_20_prev": 105.0,
            "spy_close": 430.0,
            "spy_sma200": 450.0,
        }
        strat = QullamaggieStrategy({"name": "QM", "type": "qullamaggie", "entry_mode": "breakout"})
        self.assertIsNone(strat.entry(df, len(df) - 1))


if __name__ == "__main__":
    unittest.main()
