import unittest

import numpy as np
import pandas as pd

from scripts.run_event_momentum_walkforward import (
    _base_valid_mask,
    _build_available_event_mask,
    _build_event_momentum_scores,
    _build_fundamental_matrix,
    _market_cap_coverage_stats,
)


class EventMomentumWalkforwardTests(unittest.TestCase):
    def test_build_fundamental_matrix_forward_fills_from_available_dates(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=5)
        fundamentals = {
            "AAA": pd.DataFrame(
                {"shares_outstanding": [50.0, 60.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
            )
        }
        matrix = _build_fundamental_matrix(dates, ["AAA", "BBB"], fundamentals, "shares_outstanding")
        self.assertTrue(np.isnan(matrix[0, 0]))
        self.assertEqual(float(matrix[1, 0]), 50.0)
        self.assertEqual(float(matrix[2, 0]), 50.0)
        self.assertEqual(float(matrix[3, 0]), 60.0)
        self.assertTrue(np.isnan(matrix[:, 1]).all())

    def test_base_valid_mask_can_enforce_market_cap_band(self) -> None:
        close = np.array([[10.0, 10.0]], dtype=np.float32)
        vol_ma20 = np.array([[1_000_000.0, 1_000_000.0]], dtype=np.float32)
        shares_outstanding = np.array([[50_000_000.0, 800_000_000.0]], dtype=np.float32)
        membership_mask = np.ones_like(close, dtype=bool)
        features = {
            "close": close,
            "vol_ma20": vol_ma20,
            "shares_outstanding": shares_outstanding,
            "membership_mask": membership_mask,
        }
        cfg = {
            "min_price": 8.0,
            "min_adv20": 1.0,
            "max_adv20": 0.0,
            "min_market_cap": 300_000_000.0,
            "max_market_cap": 5_000_000_000.0,
            "min_history_bars": 1,
        }
        valid = _base_valid_mask(features, cfg)
        self.assertTrue(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1]))

    def test_available_event_mask_marks_event_day_and_freshness_window(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        fundamentals = {
            "AAA": pd.DataFrame(
                {"eps_growth_yoy": [25.0, 30.0]},
                index=pd.to_datetime(["2024-01-03", "2024-01-06"]),
            ),
        }
        membership_mask = np.ones((len(dates), 2), dtype=bool)
        mask = _build_available_event_mask(
            dates,
            ["AAA", "BBB"],
            fundamentals,
            freshness_days=2,
            membership_mask=membership_mask,
        )
        self.assertTrue(mask[2, 0])  # 2024-01-03
        self.assertTrue(mask[3, 0])  # freshness day 2
        self.assertFalse(mask[4, 0])
        self.assertTrue(mask[5, 0])  # 2024-01-08 from weekend event
        self.assertFalse(mask[:, 1].any())

    def test_market_cap_coverage_stats_reports_band_counts(self) -> None:
        close = np.array(
            [
                [10.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
            ],
            dtype=np.float32,
        )
        shares_outstanding = np.array(
            [
                [50_000_000.0, 400_000_000.0, np.nan],
                [50_000_000.0, 400_000_000.0, np.nan],
            ],
            dtype=np.float32,
        )
        stats = _market_cap_coverage_stats(
            {"close": close, "shares_outstanding": shares_outstanding},
            {"min_market_cap": 300_000_000.0, "max_market_cap": 5_000_000_000.0},
        )
        self.assertEqual(stats["share_nonnull_symbols"], 2)
        self.assertEqual(stats["market_cap_band_symbols"], 1)
        self.assertEqual(stats["max_daily_market_cap_band_count"], 1)
        self.assertAlmostEqual(float(stats["avg_daily_market_cap_band_count"]), 1.0, places=6)

    def test_event_scores_require_explicit_event_mask_and_decay_forward(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=40)
        symbols = ["AAA", "BBB"]
        shape = (len(dates), len(symbols))

        close = np.full(shape, 20.0, dtype=np.float32)
        open_ = np.full(shape, 19.5, dtype=np.float32)
        high = np.full(shape, 20.5, dtype=np.float32)
        low = np.full(shape, 19.0, dtype=np.float32)
        volume = np.full(shape, 2_000_000.0, dtype=np.float32)
        vol_ma20 = np.full(shape, 1_000_000.0, dtype=np.float32)
        sma20 = np.full(shape, 19.0, dtype=np.float32)
        sma50 = np.tile(np.linspace(17.0, 18.5, len(dates), dtype=np.float32).reshape(-1, 1), (1, len(symbols)))
        sma200 = np.full(shape, 15.0, dtype=np.float32)
        high_52w = np.full(shape, 21.0, dtype=np.float32)
        eps_yoy = np.full(shape, np.nan, dtype=np.float32)
        sales_yoy = np.full(shape, np.nan, dtype=np.float32)
        eps_accel = np.full(shape, np.nan, dtype=np.float32)
        revenue_accel = np.full(shape, np.nan, dtype=np.float32)

        # Seed history then create a new event for AAA only.
        eps_yoy[:20, :] = 10.0
        sales_yoy[:20, :] = 8.0
        eps_accel[:20, :] = 0.0
        revenue_accel[:20, :] = 0.0

        event_idx = 25
        close[event_idx, 0] = 22.0
        high[event_idx, 0] = 22.5
        low[event_idx, 0] = 20.5
        volume[event_idx, 0] = 4_000_000.0
        eps_yoy[event_idx:, 0] = 40.0
        sales_yoy[event_idx:, 0] = 30.0
        eps_accel[event_idx:, 0] = 15.0
        revenue_accel[event_idx:, 0] = 10.0

        # BBB never gets a fresh fundamental event.
        eps_yoy[20:, 1] = 40.0
        sales_yoy[20:, 1] = 30.0
        eps_accel[20:, 1] = 15.0
        revenue_accel[20:, 1] = 10.0
        event_mask = np.zeros(shape, dtype=bool)
        event_mask[event_idx, 0] = True

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vol_ma20": vol_ma20,
            "sma20": sma20,
            "sma50": sma50,
            "sma200": sma200,
            "high_52w": high_52w,
            "eps_yoy": eps_yoy,
            "sales_yoy": sales_yoy,
            "eps_accel": eps_accel,
            "revenue_accel": revenue_accel,
            "shares_outstanding": np.full(shape, np.nan, dtype=np.float32),
            "event_mask": event_mask,
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 8.0,
            "min_adv20": 1.0,
            "max_adv20": 0.0,
            "min_history_bars": 10,
            "trend_required": True,
            "trend_slope_lookback": 5,
            "require_above_sma20": True,
            "require_both_growth": True,
            "min_eps_yoy": 15.0,
            "min_sales_yoy": 10.0,
            "min_event_return": 0.03,
            "min_volume_surge": 1.5,
            "min_close_strength": 0.5,
            "min_event_names": 1,
            "hold_days": 3,
            "score_decay": 0.9,
        }

        scores = _build_event_momentum_scores(features, cfg)
        self.assertIn("AAA", scores.columns)
        aaa = scores["AAA"].dropna()
        self.assertGreaterEqual(len(aaa), 3)
        self.assertGreater(float(aaa.iloc[0]), float(aaa.iloc[1]))
        self.assertGreater(float(aaa.iloc[1]), float(aaa.iloc[2]))
        if "BBB" in scores.columns:
            self.assertTrue(pd.to_numeric(scores["BBB"], errors="coerce").dropna().empty)
