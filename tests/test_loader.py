import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import clean_dataframe


class LoaderTests(unittest.TestCase):
    def test_clean_dataframe_drops_weekend_rows(self) -> None:
        dates = pd.bdate_range("2021-10-25", periods=20).tolist()
        dates.insert(5, pd.Timestamp("2021-11-21"))
        df = pd.DataFrame(
            {
                "open": [10.0 + i for i in range(len(dates))],
                "high": [10.5 + i for i in range(len(dates))],
                "low": [9.5 + i for i in range(len(dates))],
                "close": [10.1 + i for i in range(len(dates))],
                "volume": [1000 for _ in range(len(dates))],
            },
            index=pd.to_datetime(dates),
        )

        out = clean_dataframe(df)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertNotIn("2021-11-21", list(out.index.strftime("%Y-%m-%d")))


if __name__ == "__main__":
    unittest.main()
