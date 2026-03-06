import unittest

import numpy as np
import pandas as pd

from scripts.run_price_volume_event_walkforward import (
    _base_valid_mask,
    _build_price_volume_event_scores,
)


class PriceVolumeEventWalkforwardTests(unittest.TestCase):
    def test_base_valid_mask_filters_price_and_dollar_volume(self) -> None:
        close = np.array([[4.0, 10.0, 15.0]], dtype=np.float32)
        vol_ma20 = np.array([[3_000_000.0, 100_000.0, 500_000.0]], dtype=np.float32)
        membership_mask = np.ones_like(close, dtype=bool)
        features = {
            "close": close,
            "vol_ma20": vol_ma20,
            "membership_mask": membership_mask,
        }
        cfg = {
            "min_price": 5.0,
            "max_price": 20.0,
            "min_adv20": 2_000_000.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
        }
        valid = _base_valid_mask(features, cfg)
        self.assertFalse(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1]))
        self.assertTrue(bool(valid[0, 2]))

    def test_scores_require_gap_or_day_trigger_and_decay_forward(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=40)
        symbols = ["AAA", "BBB"]
        shape = (len(dates), len(symbols))

        open_ = np.full(shape, 20.0, dtype=np.float32)
        high = np.full(shape, 20.5, dtype=np.float32)
        low = np.full(shape, 19.5, dtype=np.float32)
        close = np.full(shape, 20.0, dtype=np.float32)
        volume = np.full(shape, 2_000_000.0, dtype=np.float32)
        vol_ma20 = np.full(shape, 1_000_000.0, dtype=np.float32)
        sma20 = np.full(shape, 19.0, dtype=np.float32)
        sma50 = np.tile(np.linspace(18.0, 19.0, len(dates), dtype=np.float32).reshape(-1, 1), (1, len(symbols)))
        sma200 = np.full(shape, 15.0, dtype=np.float32)
        high_52w = np.full(shape, 21.0, dtype=np.float32)

        event_idx = 25
        # Seed prior close history.
        close[:event_idx, 0] = np.linspace(18.0, 21.0, event_idx, dtype=np.float32)
        close[:event_idx, 1] = np.linspace(18.0, 20.0, event_idx, dtype=np.float32)
        open_[:event_idx, :] = close[:event_idx, :]

        # AAA triggers a strong event day.
        prev_close = close[event_idx - 1, 0]
        open_[event_idx, 0] = prev_close * 1.05
        close[event_idx, 0] = prev_close * 1.09
        high[event_idx, 0] = close[event_idx, 0] * 1.01
        low[event_idx, 0] = open_[event_idx, 0] * 0.995
        volume[event_idx, 0] = 5_000_000.0

        # BBB stays quiet and should not score.
        open_[event_idx, 1] = close[event_idx - 1, 1] * 1.01
        close[event_idx, 1] = close[event_idx - 1, 1] * 1.015

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
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 5.0,
            "min_adv20": 1.0,
            "max_adv20": 0.0,
            "min_history_bars": 10,
            "trigger_mode": "gap_or_day_return",
            "min_gap_pct": 0.03,
            "min_day_return": 0.05,
            "min_volume_surge": 1.5,
            "min_close_strength": 0.5,
            "require_green_day": True,
            "trend_required": True,
            "trend_slope_lookback": 5,
            "require_above_sma20": True,
            "min_momentum_20": 0.0,
            "min_momentum_63": 0.0,
            "min_proximity_52w": 0.0,
            "min_event_names": 1,
            "hold_days": 3,
            "score_decay": 0.9,
        }

        scores = _build_price_volume_event_scores(features, cfg)
        self.assertIn("AAA", scores.columns)
        aaa = scores["AAA"].dropna()
        self.assertGreaterEqual(len(aaa), 3)
        self.assertGreater(float(aaa.iloc[0]), float(aaa.iloc[1]))
        self.assertGreater(float(aaa.iloc[1]), float(aaa.iloc[2]))
        if "BBB" in scores.columns:
            self.assertTrue(pd.to_numeric(scores["BBB"], errors="coerce").dropna().empty)

    def test_gap_only_trigger_rejects_day_return_without_gap(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=30)
        shape = (len(dates), 1)
        close = np.full(shape, 20.0, dtype=np.float32)
        open_ = np.full(shape, 20.0, dtype=np.float32)
        high = np.full(shape, 20.6, dtype=np.float32)
        low = np.full(shape, 19.8, dtype=np.float32)
        volume = np.full(shape, 2_000_000.0, dtype=np.float32)
        vol_ma20 = np.full(shape, 1_000_000.0, dtype=np.float32)
        sma20 = np.full(shape, 19.0, dtype=np.float32)
        sma50 = np.tile(np.linspace(18.0, 18.5, len(dates), dtype=np.float32).reshape(-1, 1), (1, 1))
        sma200 = np.full(shape, 15.0, dtype=np.float32)
        high_52w = np.full(shape, 21.0, dtype=np.float32)

        event_idx = 22
        close[:event_idx, 0] = np.linspace(18.0, 20.0, event_idx, dtype=np.float32)
        open_[:event_idx, 0] = close[:event_idx, 0]
        # Big day return from intraday move, but no opening gap.
        open_[event_idx, 0] = close[event_idx - 1, 0]
        close[event_idx, 0] = close[event_idx - 1, 0] * 1.08
        volume[event_idx, 0] = 4_000_000.0

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": ["AAA"],
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
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 5.0,
            "min_adv20": 1.0,
            "max_adv20": 0.0,
            "min_history_bars": 10,
            "trigger_mode": "gap_only",
            "min_gap_pct": 0.03,
            "min_day_return": 0.05,
            "min_volume_surge": 1.5,
            "min_close_strength": 0.5,
            "require_green_day": True,
            "trend_required": True,
            "trend_slope_lookback": 5,
            "require_above_sma20": True,
            "min_momentum_20": 0.0,
            "min_momentum_63": 0.0,
            "min_proximity_52w": 0.0,
            "min_event_names": 1,
            "hold_days": 3,
            "score_decay": 0.9,
        }
        scores = _build_price_volume_event_scores(features, cfg)
        self.assertTrue(scores.empty or "AAA" not in scores.columns or scores["AAA"].dropna().empty)
