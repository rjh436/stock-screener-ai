import unittest

import numpy as np
import pandas as pd

from execution.engine import (
    PreparedBacktestData,
    _SymbolArrays,
    _normalize_daily_dataframe_index,
    _normalize_prepared_calendar,
)


class ExecutionEngineTests(unittest.TestCase):
    def test_normalize_daily_dataframe_index_drops_weekends(self) -> None:
        df = pd.DataFrame(
            {
                "close": [10.0, 11.0, 12.0],
            },
            index=pd.to_datetime(
                [
                    "2021-11-19 06:00:00",
                    "2021-11-21 06:00:00",
                    "2021-11-22 06:00:00",
                ]
            ),
        )

        out = _normalize_daily_dataframe_index(df)
        self.assertEqual(list(out.index.strftime("%Y-%m-%d")), ["2021-11-19", "2021-11-22"])

    def test_normalize_prepared_calendar_drops_weekend_rows_from_cached_symbol_arrays(self) -> None:
        idx = np.array(
            pd.to_datetime(["2021-11-19", "2021-11-21", "2021-11-22"]).values,
            dtype="datetime64[ns]",
        )
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=pd.to_datetime(idx))
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sym = _SymbolArrays(
            df=df,
            index=idx.copy(),
            gidx=np.array([0, 1, 2], dtype=np.int32),
            open=arr.copy(),
            high=arr.copy(),
            low=arr.copy(),
            close=arr.copy(),
            volume=arr.copy(),
            rsi14=arr.copy(),
            atr14=arr.copy(),
            natr=arr.copy(),
            volma50=arr.copy(),
            sma10=arr.copy(),
            sma20=arr.copy(),
            sma50=arr.copy(),
            sma150=arr.copy(),
            sma200=arr.copy(),
            sma200slope=arr.copy(),
            high52w=arr.copy(),
            low52w=arr.copy(),
            bbwidth=arr.copy(),
            spyclose=arr.copy(),
            spysma20=arr.copy(),
            spysma50=arr.copy(),
            spysma200=arr.copy(),
            rsrating=arr.copy(),
            momrank=arr.copy(),
            adr_pct=arr.copy(),
            prev_high=arr.copy(),
            highest10_1=arr.copy(),
            gap_pct=arr.copy(),
            clv=arr.copy(),
            trend_mask=np.array([True, True, True]),
            adx=arr.copy(),
            rs_ratio=arr.copy(),
            rs_ratio_sma50=arr.copy(),
        )
        prepared = PreparedBacktestData(
            enriched={"AAA": sym},
            all_dates=idx.copy(),
        )

        out = _normalize_prepared_calendar(prepared)
        self.assertEqual(list(pd.DatetimeIndex(out.all_dates).strftime("%Y-%m-%d")), ["2021-11-19", "2021-11-22"])
        self.assertEqual(list(pd.DatetimeIndex(out.enriched["AAA"].index).strftime("%Y-%m-%d")), ["2021-11-19", "2021-11-22"])
        self.assertEqual(out.enriched["AAA"].gidx.tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
