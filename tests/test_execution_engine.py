import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from execution.engine import (
    PreparedBacktestData,
    _SymbolArrays,
    _evaluate_exit_state_machine,
    _normalize_daily_dataframe_index,
    _normalize_prepared_calendar,
)


class ExecutionEngineTests(unittest.TestCase):
    def _make_symbol_arrays(self, close_vals, sma50_vals) -> _SymbolArrays:
        idx = np.array(pd.to_datetime(["2021-11-19", "2021-11-22", "2021-11-23"]).values, dtype="datetime64[ns]")
        df = pd.DataFrame(
            {
                "open": close_vals,
                "high": [v + 0.5 for v in close_vals],
                "low": [v - 0.5 for v in close_vals],
                "close": close_vals,
                "volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
                "atr14": [1.0, 1.0, 1.0],
                "natr": [1.0, 1.0, 1.0],
                "volma50": [1.0, 1.0, 1.0],
                "sma10": close_vals,
                "sma20": close_vals,
                "sma50": sma50_vals,
                "sma150": sma50_vals,
                "sma200": sma50_vals,
                "sma200slope": [1.0, 1.0, 1.0],
                "high52w": [20.0, 20.0, 20.0],
                "low52w": [5.0, 5.0, 5.0],
                "bbwidth": [1.0, 1.0, 1.0],
                "spyclose": [1.0, 1.0, 1.0],
                "spysma20": [1.0, 1.0, 1.0],
                "spysma50": [1.0, 1.0, 1.0],
                "spysma200": [1.0, 1.0, 1.0],
                "rsrating": [95.0, 95.0, 95.0],
                "momrank": [95.0, 95.0, 95.0],
                "adr_pct": [2.0, 2.0, 2.0],
                "prev_high": [13.0, 13.0, 13.0],
                "highest10_1": [13.0, 13.0, 13.0],
                "gap_pct": [0.0, 0.0, 0.0],
                "clv": [0.8, 0.8, 0.8],
                "adx": [30.0, 30.0, 30.0],
                "rs_ratio": [1.0, 1.0, 1.0],
                "rs_ratio_sma50": [1.0, 1.0, 1.0],
            },
            index=pd.to_datetime(idx),
        )
        arr = lambda vals: np.array(vals, dtype=np.float64)
        return _SymbolArrays(
            df=df,
            index=idx.copy(),
            gidx=np.array([0, 1, 2], dtype=np.int32),
            open=arr(close_vals),
            high=arr([v + 0.5 for v in close_vals]),
            low=arr([v - 0.5 for v in close_vals]),
            close=arr(close_vals),
            volume=arr([1_000_000.0, 1_000_000.0, 1_000_000.0]),
            rsi14=arr([60.0, 60.0, 60.0]),
            atr14=arr([1.0, 1.0, 1.0]),
            natr=arr([1.0, 1.0, 1.0]),
            volma50=arr([1.0, 1.0, 1.0]),
            sma10=arr(close_vals),
            sma20=arr(close_vals),
            sma50=arr(sma50_vals),
            sma150=arr(sma50_vals),
            sma200=arr(sma50_vals),
            sma200slope=arr([1.0, 1.0, 1.0]),
            high52w=arr([20.0, 20.0, 20.0]),
            low52w=arr([5.0, 5.0, 5.0]),
            bbwidth=arr([1.0, 1.0, 1.0]),
            spyclose=arr([1.0, 1.0, 1.0]),
            spysma20=arr([1.0, 1.0, 1.0]),
            spysma50=arr([1.0, 1.0, 1.0]),
            spysma200=arr([1.0, 1.0, 1.0]),
            rsrating=arr([95.0, 95.0, 95.0]),
            momrank=arr([95.0, 95.0, 95.0]),
            adr_pct=arr([2.0, 2.0, 2.0]),
            prev_high=arr([13.0, 13.0, 13.0]),
            highest10_1=arr([13.0, 13.0, 13.0]),
            gap_pct=arr([0.0, 0.0, 0.0]),
            clv=arr([0.8, 0.8, 0.8]),
            trend_mask=np.array([True, True, True]),
            adx=arr([30.0, 30.0, 30.0]),
            rs_ratio=arr([1.0, 1.0, 1.0]),
            rs_ratio_sma50=arr([1.0, 1.0, 1.0]),
        )

    def test_strategy_exit_confirm_bars_requires_second_soft_exit_signal(self) -> None:
        sym_data = self._make_symbol_arrays([12.0, 11.0, 10.0], [10.0, 11.5, 11.5])
        pos = {
            "entry_price": 12.0,
            "stop_price": 9.0,
            "entry_day_idx": 0,
            "partial_taken": False,
            "strategy_exit_signal_streak": 0,
        }
        params = {
            "exit_sma_fast": "sma50",
            "exit_sma_slow": "sma50",
            "time_stop_days": 7,
            "strategy_exit_confirm_bars": 2,
        }

        should_exit, _, reason, _ = _evaluate_exit_state_machine(
            sym="AAA",
            pos=pos,
            params=params,
            sym_data=sym_data,
            loc=1,
            day_idx=1,
            current_open=11.0,
            current_low=10.5,
            current_close=11.0,
            cash=0.0,
            trades_list=[],
        )
        self.assertFalse(should_exit)
        self.assertIsNone(reason)
        self.assertEqual(pos["strategy_exit_signal_streak"], 1)

        should_exit, _, reason, _ = _evaluate_exit_state_machine(
            sym="AAA",
            pos=pos,
            params=params,
            sym_data=sym_data,
            loc=2,
            day_idx=2,
            current_open=10.0,
            current_low=9.5,
            current_close=10.0,
            cash=0.0,
            trades_list=[],
        )
        self.assertTrue(should_exit)
        self.assertEqual(reason, "STRATEGY_EXIT")

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
