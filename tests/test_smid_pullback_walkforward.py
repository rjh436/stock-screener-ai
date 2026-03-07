import unittest

import numpy as np
import pandas as pd

from scripts.run_smid_pullback_walkforward import (
    _base_valid_mask,
    _build_smid_pullback_scores,
)


class SmidPullbackWalkforwardTests(unittest.TestCase):
    def test_base_valid_mask_filters_price_liquidity_and_history(self) -> None:
        close = np.array([[1.5, 10.0, 10.0]], dtype=np.float32)
        vol_ma20 = np.array([[1_000_000.0, 50_000.0, 200_000.0]], dtype=np.float32)
        membership_mask = np.ones_like(close, dtype=bool)
        features = {
            "close": close,
            "vol_ma20": vol_ma20,
            "membership_mask": membership_mask,
        }
        cfg = {
            "min_price": 2.0,
            "max_price": 80.0,
            "min_adv20": 1_000_000.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
        }
        valid = _base_valid_mask(features, cfg)
        self.assertFalse(bool(valid[0, 0]))
        self.assertFalse(bool(valid[0, 1]))
        self.assertTrue(bool(valid[0, 2]))

    def test_base_valid_mask_skips_max_price_when_disabled(self) -> None:
        close = np.array([[120.0, 10.0]], dtype=np.float32)
        vol_ma20 = np.array([[20_000.0, 200_000.0]], dtype=np.float32)
        membership_mask = np.ones_like(close, dtype=bool)
        features = {
            "close": close,
            "vol_ma20": vol_ma20,
            "membership_mask": membership_mask,
        }
        cfg = {
            "min_price": 2.0,
            "max_price": 0.0,
            "min_adv20": 1_000_000.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
        }
        valid = _base_valid_mask(features, cfg)
        self.assertTrue(bool(valid[0, 0]))
        self.assertTrue(bool(valid[0, 1]))

    def test_base_valid_mask_uses_nominal_price_proxy_when_requested(self) -> None:
        close = np.array([[60.0, 60.0]], dtype=np.float32)
        raw_close = np.array([[120.0, 60.0]], dtype=np.float32)
        vol_ma20 = np.array([[20_000.0, 20_000.0]], dtype=np.float32)
        membership_mask = np.ones_like(close, dtype=bool)
        features = {
            "close": close,
            "raw_close": raw_close,
            "vol_ma20": vol_ma20,
            "membership_mask": membership_mask,
        }
        cfg = {
            "min_price": 2.0,
            "max_price": 80.0,
            "min_adv20": 1_000_000.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
            "price_filter_mode": "nominal",
        }
        valid = _base_valid_mask(features, cfg)
        self.assertFalse(bool(valid[0, 0]))
        self.assertTrue(bool(valid[0, 1]))

    def test_scores_prefer_stronger_pullback_candidate(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        symbols = ["AAA", "BBB", "CCC"]
        shape = (len(dates), len(symbols))

        close = np.array(
            [
                [20.0, 20.0, 20.0],
                [21.0, 21.0, 21.0],
                [22.0, 22.0, 22.0],
            ],
            dtype=np.float32,
        )
        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "open": close.copy(),
            "close": close,
            "vol_ma20": np.full(shape, 100_000.0, dtype=np.float32),
            "sma20": np.array(
                [
                    [19.8, 19.2, 19.8],
                    [20.6, 20.0, 20.6],
                    [21.5, 21.3, 21.5],
                ],
                dtype=np.float32,
            ),
            "sma50": np.full(shape, 18.5, dtype=np.float32),
            "sma200": np.full(shape, 15.0, dtype=np.float32),
            "high52w": np.array(
                [
                    [24.0, 24.0, 24.0],
                    [24.0, 24.0, 24.0],
                    [24.5, 25.0, 24.5],
                ],
                dtype=np.float32,
            ),
            "ret_3m": np.array(
                [
                    [40.0, 35.0, 10.0],
                    [45.0, 38.0, 10.0],
                    [50.0, 40.0, 10.0],
                ],
                dtype=np.float32,
            ),
            "ret_6m": np.array(
                [
                    [60.0, 50.0, 10.0],
                    [65.0, 55.0, 10.0],
                    [70.0, 55.0, 10.0],
                ],
                dtype=np.float32,
            ),
            "adr_pct": np.array(
                [
                    [4.0, 3.0, 1.0],
                    [4.5, 3.2, 1.0],
                    [5.0, 3.5, 1.0],
                ],
                dtype=np.float32,
            ),
            "rs_rating": np.array(
                [
                    [96.0, 90.0, 50.0],
                    [97.0, 91.0, 50.0],
                    [98.0, 92.0, 50.0],
                ],
                dtype=np.float32,
            ),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 2.0,
            "max_price": 80.0,
            "min_adv20": 1.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
            "min_rank_names": 2,
            "min_rs_rating": 90.0,
            "min_ret_3m": 35.0,
            "min_ret_6m": 50.0,
            "min_pct_off_high": -0.18,
            "max_pct_off_high": -0.03,
            "min_dist_sma20": -0.04,
            "max_dist_sma20": 0.10,
            "min_adr_pct": 3.0,
            "rs_weight": 1.0,
            "ret_3m_weight": 0.8,
            "ret_6m_weight": 0.5,
            "dist_sma20_weight": 0.7,
            "pct_off_high_weight": 0.4,
            "adr_weight": 0.2,
        }

        scores = _build_smid_pullback_scores(features, cfg)
        self.assertIn("AAA", scores.columns)
        self.assertIn("BBB", scores.columns)
        last = pd.to_numeric(scores.iloc[-1], errors="coerce").dropna()
        self.assertGreater(float(last["AAA"]), float(last["BBB"]))
        self.assertNotIn("CCC", last.index)


if __name__ == "__main__":
    unittest.main()
