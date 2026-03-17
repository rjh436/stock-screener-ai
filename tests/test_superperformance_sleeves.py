import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from strategies.superperformance import SuperperformanceStrategy


class SuperperformanceSleeveTests(unittest.TestCase):
    def test_continuation_sleeve_signal(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "multi_sleeve_enabled": True,
                "enabled_sleeves": ["continuation"],
                "warmup_bars": 1,
                "continuation_rs_min": 70,
                "continuation_runup_min_pct": 10,
                "continuation_require_recent_high_pullback": False,
                "continuation_require_support_touch": False,
                "continuation_require_reclaim_breakout": False,
            }
        )
        df = pd.DataFrame(
            [
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "sma20": 98.0,
                    "sma50": 95.0,
                    "sma150": 90.0,
                    "sma200": 85.0,
                    "volume": 100000.0,
                    "vol_ma50": 100000.0,
                    "ret_1m": 12.0,
                    "ret_3m": 20.0,
                    "rs_percentile": 80.0,
                },
                {
                    "open": 101.0,
                    "high": 103.0,
                    "low": 99.5,
                    "close": 102.0,
                    "sma20": 100.0,
                    "sma50": 98.0,
                    "sma150": 92.0,
                    "sma200": 88.0,
                    "volume": 90000.0,
                    "vol_ma50": 110000.0,
                    "ret_1m": 15.0,
                    "ret_3m": 22.0,
                    "rs_percentile": 82.0,
                },
            ]
        )

        decision = strategy.check_setup(df, 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "continuation")
        self.assertEqual(decision["sleeve"], "continuation")

    def test_recovery_sleeve_signal(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "multi_sleeve_enabled": True,
                "enabled_sleeves": ["recovery"],
                "warmup_bars": 1,
                "recovery_rs_min": 60,
            }
        )

        rows = []
        for idx in range(12):
            if idx < 6:
                close = 95.0
            else:
                close = 106.0
            rows.append(
                {
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "sma20": 99.0,
                    "sma50": 100.0,
                    "sma150": 95.0,
                    "sma200": 90.0,
                    "volume": 150000.0,
                    "vol_ma50": 120000.0,
                    "rs_percentile": 75.0,
                    "ret_1m": 8.0,
                    "ret_3m": 12.0,
                }
            )
        df = pd.DataFrame(rows)

        decision = strategy.check_setup(df, 11)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "recovery")
        self.assertEqual(decision["sleeve"], "recovery")

    def test_continuation_requires_recent_high_pullback(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "multi_sleeve_enabled": True,
                "enabled_sleeves": ["continuation"],
                "warmup_bars": 1,
                "continuation_rs_min": 70,
                "continuation_runup_min_pct": 5,
                "continuation_pullback_max_pct": 0.05,
                "continuation_recent_high_lookback_bars": 40,
                "continuation_recent_high_max_age_bars": 10,
                "continuation_recent_high_pullback_min_pct": 0.02,
                "continuation_recent_high_pullback_max_pct": 0.20,
                "continuation_require_support_touch": False,
                "continuation_require_reclaim_breakout": False,
                "continuation_require_up_close": False,
            }
        )

        rows = []
        for idx in range(25):
            close = 90.0 + idx
            rows.append(
                {
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "sma20": close - 1.0,
                    "sma50": close - 4.0,
                    "sma150": close - 8.0,
                    "sma200": close - 10.0,
                    "volume": 100000.0,
                    "vol_ma50": 110000.0,
                    "ret_1m": 12.0,
                    "ret_3m": 20.0,
                    "rs_percentile": 85.0,
                }
            )

        # Replace final bar with a clean pullback after a recent high.
        rows[-1]["close"] = 110.0
        rows[-1]["open"] = 109.5
        rows[-1]["high"] = 111.0
        rows[-1]["low"] = 108.5
        rows[-1]["sma20"] = 109.0
        rows[-1]["sma50"] = 105.0
        rows[-1]["sma150"] = 100.0
        rows[-1]["sma200"] = 95.0
        rows[-1]["ret_1m"] = 10.0
        rows[-1]["ret_3m"] = 18.0
        df = pd.DataFrame(rows)

        decision = strategy.check_setup(df, len(df) - 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "continuation")

        # No pullback from recent highs should fail the continuation sleeve.
        df2 = df.copy()
        df2.iloc[-1, df2.columns.get_loc("close")] = 115.0
        df2.iloc[-1, df2.columns.get_loc("open")] = 114.5
        df2.iloc[-1, df2.columns.get_loc("high")] = 116.0
        df2.iloc[-1, df2.columns.get_loc("low")] = 113.5
        df2.iloc[-1, df2.columns.get_loc("sma20")] = 114.0
        df2.iloc[-1, df2.columns.get_loc("sma50")] = 110.0
        df2.iloc[-1, df2.columns.get_loc("sma150")] = 104.0
        df2.iloc[-1, df2.columns.get_loc("sma200")] = 98.0
        decision2 = strategy.check_setup(df2, len(df2) - 1)
        self.assertIsNone(decision2)

    def test_continuation_default_requires_reclaim_breakout(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "multi_sleeve_enabled": True,
                "enabled_sleeves": ["continuation"],
                "warmup_bars": 1,
                "continuation_rs_min": 70,
                "continuation_runup_min_pct": 5,
                "continuation_recent_high_lookback_bars": 30,
                "continuation_recent_high_max_age_bars": 20,
                "continuation_recent_high_pullback_min_pct": 0.02,
                "continuation_recent_high_pullback_max_pct": 0.20,
                "continuation_support_touch_lookback_bars": 8,
                "continuation_support_touch_tolerance_pct": 0.02,
                "continuation_pullback_max_pct": 0.08,
                "continuation_reclaim_lookback_bars": 5,
                "continuation_reclaim_buffer_pct": 0.0,
            }
        )

        rows = []
        base_close = 100.0
        for idx in range(16):
            close = base_close + idx
            rows.append(
                {
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "sma20": close - 2.0,
                    "sma50": close - 5.0,
                    "sma150": close - 10.0,
                    "sma200": close - 12.0,
                    "volume": 90000.0,
                    "vol_ma50": 110000.0,
                    "ret_1m": 12.0,
                    "ret_3m": 20.0,
                    "rs_percentile": 85.0,
                    "high_52w": close + 3.0,
                }
            )

        # Put an older high in the lookback window, then a pullback and reclaim.
        rows[-7].update({"close": 121.0, "open": 120.0, "high": 122.0, "low": 119.0, "sma20": 114.0, "sma50": 110.0})
        rows[-3].update({"close": 112.0, "open": 112.5, "high": 113.0, "low": 109.5, "sma20": 111.0, "sma50": 108.0})
        rows[-2].update({"close": 111.0, "open": 111.5, "high": 112.0, "low": 108.8, "sma20": 110.6, "sma50": 107.8})
        # Final bar reclaims above recent short-term highs but remains below older high.
        rows[-1].update({"close": 116.0, "open": 114.0, "high": 116.5, "low": 113.5, "sma20": 111.2, "sma50": 108.2})
        df = pd.DataFrame(rows)

        decision = strategy.check_setup(df, len(df) - 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "continuation")

        # Without reclaim above prior pivot highs, continuation must fail.
        df_bad = df.copy()
        df_bad.iloc[-1, df_bad.columns.get_loc("close")] = 112.2
        df_bad.iloc[-1, df_bad.columns.get_loc("open")] = 111.8
        df_bad.iloc[-1, df_bad.columns.get_loc("high")] = 113.0
        df_bad.iloc[-1, df_bad.columns.get_loc("low")] = 111.0
        decision_bad = strategy.check_setup(df_bad, len(df_bad) - 1)
        self.assertIsNone(decision_bad)

    def test_continuation_breakout_fallback_signal(self) -> None:
        strategy = SuperperformanceStrategy(
            {
                "multi_sleeve_enabled": True,
                "enabled_sleeves": ["continuation"],
                "warmup_bars": 1,
                "continuation_breakout_enabled": True,
                "continuation_require_recent_high_pullback": True,
                "continuation_require_support_touch": False,
                "continuation_require_reclaim_breakout": False,
                "continuation_breakout_lookback_bars": 10,
                "continuation_breakout_rs_min": 75,
                "continuation_breakout_runup_min_pct": 8,
                "continuation_breakout_volume_min_mult": 1.0,
                "continuation_breakout_buffer_pct": 0.0,
            }
        )

        rows = []
        for idx in range(25):
            close = 100.0 + idx
            rows.append(
                {
                    "open": close - 0.8,
                    "high": close + 0.8,
                    "low": close - 1.0,
                    "close": close,
                    "sma20": close - 2.0,
                    "sma50": close - 4.0,
                    "sma150": close - 9.0,
                    "sma200": close - 12.0,
                    "volume": 120000.0,
                    "vol_ma50": 100000.0,
                    "ret_1m": 10.0,
                    "ret_3m": 18.0,
                    "rs_percentile": 85.0,
                    "high_52w": close + 2.0,
                }
            )

        # New pivot breakout bar; this should fail pullback continuation but pass
        # breakout fallback.
        rows[-1]["open"] = rows[-2]["close"] + 0.5
        rows[-1]["close"] = rows[-2]["close"] + 2.0
        rows[-1]["high"] = rows[-1]["close"] + 0.7
        rows[-1]["low"] = rows[-1]["open"] - 0.8
        rows[-1]["volume"] = 150000.0
        rows[-1]["vol_ma50"] = 100000.0
        rows[-1]["high_52w"] = rows[-1]["close"] + 2.0
        df = pd.DataFrame(rows)

        decision = strategy.check_setup(df, len(df) - 1)
        self.assertIsNotNone(decision)
        self.assertEqual(decision["entry_type"], "continuation")
        self.assertEqual(decision["sleeve"], "continuation")

    def test_adaptive_green_breakout_gates_relax(self) -> None:
        base_params = {
            "multi_sleeve_enabled": False,
            "warmup_bars": 1,
            "entry_mode": "ep",
            "rs_percentile_min": 80,
            "prior_runup_min_pct": 30,
            "adr_min_pct": 3.5,
            "fundamental_growth_min_pct": 20,
            "ep_gap_pct": 6,
            "ep_vol_mult": 3.0,
            "ep_close_near_high_min": 0.7,
            "ep_entry_mode": "close",
            "ep_force_next_day": True,
        }
        rows = [
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100000.0,
                "vol_ma50": 100000.0,
                "sma10": 99.0,
                "sma20": 98.0,
                "sma50": 95.0,
                "sma150": 90.0,
                "sma200": 85.0,
                "ret_1m": 20.0,
                "ret_3m": 20.0,
                "rs_percentile": 75.0,
                "adr_pct": 2.8,
                "eps_growth_yoy": 30.0,
                "sales_growth_yoy": 25.0,
                "spy_close": 500.0,
                "spy_sma200": 450.0,
            },
            {
                "open": 108.0,
                "high": 110.0,
                "low": 106.0,
                "close": 109.0,
                "volume": 400000.0,
                "vol_ma50": 100000.0,
                "sma10": 105.0,
                "sma20": 103.0,
                "sma50": 100.0,
                "sma150": 95.0,
                "sma200": 90.0,
                "ret_1m": 20.0,
                "ret_3m": 20.0,
                "rs_percentile": 75.0,
                "adr_pct": 2.8,
                "eps_growth_yoy": 30.0,
                "sales_growth_yoy": 25.0,
                "spy_close": 505.0,
                "spy_sma200": 451.0,
            },
        ]
        df = pd.DataFrame(rows)

        strat_static = SuperperformanceStrategy(dict(base_params))
        decision_static = strat_static.check_setup(df, 1)
        self.assertIsNone(decision_static)

        adaptive_params = dict(base_params)
        adaptive_params.update(
            {
                "adaptive_breakout_gates_enabled": True,
                "adaptive_green_rs_min": 70,
                "adaptive_green_runup_min_pct": 15,
                "adaptive_green_adr_min_pct": 2.5,
            }
        )
        strat_adaptive = SuperperformanceStrategy(adaptive_params)
        decision_adaptive = strat_adaptive.check_setup(df, 1)
        self.assertIsNotNone(decision_adaptive)
        self.assertEqual(decision_adaptive["entry_type"], "ep")

    def test_adaptive_green_adv50_floor_relaxes_liquidity_gate(self) -> None:
        base_params = {
            "multi_sleeve_enabled": False,
            "warmup_bars": 1,
            "entry_mode": "ep",
            "rs_gate_min": 70,
            "prior_runup_min_pct": 15,
            "adr_min_pct": 2.5,
            "fundamental_growth_min_pct": 20,
            "min_avg_dollar_volume_50": 15000000,
            "ep_gap_pct": 6,
            "ep_vol_mult": 3.0,
            "ep_close_near_high_min": 0.7,
            "ep_entry_mode": "close",
            "ep_force_next_day": True,
        }
        rows = [
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100000.0,
                "vol_ma50": 100000.0,
                "sma10": 99.0,
                "sma20": 98.0,
                "sma50": 95.0,
                "sma150": 90.0,
                "sma200": 85.0,
                "ret_1m": 20.0,
                "ret_3m": 20.0,
                "rs_percentile": 75.0,
                "adr_pct": 2.8,
                "eps_growth_yoy": 30.0,
                "sales_growth_yoy": 25.0,
                "spy_close": 500.0,
                "spy_sma200": 450.0,
            },
            {
                "open": 108.0,
                "high": 110.0,
                "low": 106.0,
                "close": 109.0,
                "volume": 400000.0,
                "vol_ma50": 100000.0,
                "sma10": 105.0,
                "sma20": 103.0,
                "sma50": 100.0,
                "sma150": 95.0,
                "sma200": 90.0,
                "ret_1m": 20.0,
                "ret_3m": 20.0,
                "rs_percentile": 75.0,
                "adr_pct": 2.8,
                "eps_growth_yoy": 30.0,
                "sales_growth_yoy": 25.0,
                "spy_close": 505.0,
                "spy_sma200": 451.0,
            },
        ]
        df = pd.DataFrame(rows)

        strat_static = SuperperformanceStrategy(dict(base_params))
        decision_static = strat_static.check_setup(df, 1)
        self.assertIsNone(decision_static)
        self.assertIn("adv50", strat_static._last_reject_reason)

        adaptive_params = dict(base_params)
        adaptive_params.update(
            {
                "adaptive_breakout_gates_enabled": True,
                "adaptive_green_min_avg_dollar_volume_50": 5000000,
            }
        )
        strat_adaptive = SuperperformanceStrategy(adaptive_params)
        decision_adaptive = strat_adaptive.check_setup(df, 1)
        self.assertIsNotNone(decision_adaptive)
        self.assertEqual(decision_adaptive["entry_type"], "ep")


if __name__ == "__main__":
    unittest.main()
