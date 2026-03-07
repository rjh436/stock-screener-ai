import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from data import corporate_actions
from data.corporate_actions import (
    build_nominal_price_frame,
    build_split_adjustment_factor,
    load_split_series,
    save_split_series,
)


class CorporateActionsTests(unittest.TestCase):
    def test_build_split_adjustment_factor_applies_future_splits_to_prior_bars(self) -> None:
        idx = pd.to_datetime(["2020-08-28", "2020-08-31", "2020-09-01"])
        splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
        factor = build_split_adjustment_factor(idx, splits)
        self.assertAlmostEqual(float(factor.iloc[0]), 4.0, places=6)
        self.assertAlmostEqual(float(factor.iloc[1]), 1.0, places=6)
        self.assertAlmostEqual(float(factor.iloc[2]), 1.0, places=6)

    def test_build_nominal_price_frame_multiplies_adjusted_ohlc_by_split_factor(self) -> None:
        idx = pd.to_datetime(["2020-08-28", "2020-08-31"])
        frame = pd.DataFrame(
            {
                "open": [126.0, 127.58],
                "high": [126.5, 131.0],
                "low": [124.0, 126.0],
                "close": [124.8075, 129.04],
            },
            index=idx,
        )
        splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
        raw = build_nominal_price_frame(frame, symbol="AAPL", split_series=splits, allow_fetch=False)
        self.assertAlmostEqual(float(raw.loc[pd.Timestamp("2020-08-28"), "raw_close"]), 499.23, places=2)
        self.assertAlmostEqual(float(raw.loc[pd.Timestamp("2020-08-31"), "raw_close"]), 129.04, places=2)

    def test_save_and_load_split_series_round_trip_with_meta_row(self) -> None:
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with mock.patch.object(corporate_actions, "CACHE_DIR", cache_dir):
                splits = pd.Series([4.0], index=pd.to_datetime(["2020-08-31"]))
                save_split_series("AAPL", splits, fetched_ok=True)
                series = load_split_series("AAPL", allow_fetch=False)
        self.assertIsNotNone(series)
        self.assertAlmostEqual(float(series.loc[pd.Timestamp("2020-08-31")]), 4.0, places=6)

    def test_load_split_series_ignores_legacy_cache_without_meta(self) -> None:
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            with mock.patch.object(corporate_actions, "CACHE_DIR", cache_dir):
                legacy = pd.DataFrame({"split_ratio": [4.0]}, index=pd.to_datetime(["2020-08-31"]))
                legacy.index.name = "date"
                legacy.to_parquet(cache_dir / "AAPL.parquet")
                series = load_split_series("AAPL", allow_fetch=False)
        self.assertIsNone(series)


if __name__ == "__main__":
    unittest.main()
