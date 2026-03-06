import unittest

import numpy as np
import pandas as pd

from scripts.run_sector_stock_momentum_walkforward import (
    _base_stock_valid_mask,
    _build_sector_filtered_target_schedule,
)


class SectorStockMomentumWalkforwardTests(unittest.TestCase):
    def test_base_stock_valid_mask_filters_price_liquidity_history_and_unknown_sector(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        close_px = pd.DataFrame(
            {
                "TECH": [12.0, 12.1, 12.2, 12.3, 12.5, 13.0],
                "CHEAP": [4.0, 4.2, 4.4, 4.6, 4.8, 5.0],
                "THIN": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
                "UNKNOWN": [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
            },
            index=dates,
        )
        volume_px = pd.DataFrame(
            {
                "TECH": [300_000.0, 300_000.0, 300_000.0, 300_000.0, 300_000.0, 300_000.0],
                "CHEAP": [500_000.0, 500_000.0, 500_000.0, 500_000.0, 500_000.0, 500_000.0],
                "THIN": [10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0],
                "UNKNOWN": [300_000.0, 300_000.0, 300_000.0, 300_000.0, 300_000.0, 300_000.0],
            },
            index=dates,
        )
        sector_map = {"TECH": "Technology", "CHEAP": "Technology", "THIN": "Healthcare"}
        cfg = {
            "min_price": 10.0,
            "min_adv20": 2_000_000.0,
            "max_adv20": 0.0,
            "min_history_bars": 1,
        }

        valid = _base_stock_valid_mask(close_px, volume_px, sector_map, cfg)
        self.assertTrue(bool(valid.iloc[-1]["TECH"]))
        self.assertFalse(bool(valid.iloc[-1]["CHEAP"]))
        self.assertFalse(bool(valid.iloc[-1]["THIN"]))
        self.assertFalse(bool(valid.iloc[-1]["UNKNOWN"]))

    def test_sector_filtered_schedule_selects_only_winning_sector_names(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=3)
        stock_scores = pd.DataFrame(
            {
                "TECH1": [80.0, 85.0, 90.0],
                "TECH2": [70.0, 75.0, 80.0],
                "HEALTH1": [95.0, 96.0, 97.0],
            },
            index=dates,
        )
        stock_valid_mask = pd.DataFrame(True, index=dates, columns=stock_scores.columns)
        sector_target_schedule = pd.DataFrame(index=dates, columns=["XLK", "XLV"], dtype=float)
        sector_target_schedule.iloc[-1] = [1.0, np.nan]
        sector_name_by_symbol = {
            "TECH1": "Technology",
            "TECH2": "Technology",
            "HEALTH1": "Healthcare",
        }
        sector_name_by_etf = {"XLK": "Technology", "XLV": "Healthcare"}
        stock_cfg = {
            "target_count": 2,
            "hold_buffer_mult": 1.1,
            "conviction_weighted": False,
            "conviction_power": 1.0,
            "gross_exposure": 1.0,
            "min_rank_names": 1,
            "max_stocks_per_sector": 0,
        }

        schedule = _build_sector_filtered_target_schedule(
            stock_scores=stock_scores,
            stock_valid_mask=stock_valid_mask,
            sector_target_schedule=sector_target_schedule,
            sector_name_by_symbol=sector_name_by_symbol,
            sector_name_by_etf=sector_name_by_etf,
            stock_cfg=stock_cfg,
        )
        last = schedule.iloc[-1].dropna()
        self.assertEqual(set(last.index), {"TECH1", "TECH2"})
        self.assertAlmostEqual(float(last.sum()), 1.0, places=6)

    def test_sector_filtered_schedule_respects_max_stocks_per_sector(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=2)
        stock_scores = pd.DataFrame(
            {
                "TECH1": [80.0, 90.0],
                "TECH2": [79.0, 89.0],
                "HEALTH1": [78.0, 88.0],
                "HEALTH2": [77.0, 87.0],
            },
            index=dates,
        )
        stock_valid_mask = pd.DataFrame(True, index=dates, columns=stock_scores.columns)
        sector_target_schedule = pd.DataFrame(index=dates, columns=["XLK", "XLV"], dtype=float)
        sector_target_schedule.iloc[-1] = [0.5, 0.5]
        sector_name_by_symbol = {
            "TECH1": "Technology",
            "TECH2": "Technology",
            "HEALTH1": "Healthcare",
            "HEALTH2": "Healthcare",
        }
        sector_name_by_etf = {"XLK": "Technology", "XLV": "Healthcare"}
        stock_cfg = {
            "target_count": 4,
            "hold_buffer_mult": 1.1,
            "conviction_weighted": False,
            "conviction_power": 1.0,
            "gross_exposure": 1.0,
            "min_rank_names": 1,
            "max_stocks_per_sector": 1,
        }

        schedule = _build_sector_filtered_target_schedule(
            stock_scores=stock_scores,
            stock_valid_mask=stock_valid_mask,
            sector_target_schedule=sector_target_schedule,
            sector_name_by_symbol=sector_name_by_symbol,
            sector_name_by_etf=sector_name_by_etf,
            stock_cfg=stock_cfg,
        )
        last = schedule.iloc[-1].dropna()
        self.assertEqual(len(last), 2)
        self.assertIn("TECH1", last.index)
        self.assertIn("HEALTH1", last.index)


if __name__ == "__main__":
    unittest.main()
