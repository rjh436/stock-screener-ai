import unittest

import numpy as np
import pandas as pd

from scripts.run_etf_rotation_walkforward import (
    _build_allocator_state,
    _build_allocator_target_schedule,
    _build_exposure_scalar,
    _build_rotation_scores,
)


class EtfRotationWalkforwardTests(unittest.TestCase):
    def test_rotation_scores_favor_stronger_trending_symbol_and_filter_weak_symbol(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="B")
        leader = pd.Series(np.linspace(100.0, 220.0, len(dates)), index=dates)
        laggard = pd.Series(np.linspace(100.0, 80.0, len(dates)), index=dates)
        close_px = pd.DataFrame({"LEAD": leader, "LAG": laggard}, index=dates)

        cfg = {
            "min_price": 5.0,
            "min_rank_names": 2,
            "lookbacks": {"21": 0.5, "63": 0.5},
            "absolute_momentum_lookback": 63,
            "absolute_momentum_min": 0.0,
            "trend_ma_days": 50,
            "fast_ma_days": 20,
            "require_fast_above_trend": True,
            "short_term_reversal_days": 0,
            "volatility_lookback_days": 0,
        }

        scores = _build_rotation_scores(close_px, cfg)
        last = scores.iloc[-1]
        self.assertTrue(np.isfinite(last["LEAD"]))
        self.assertTrue(np.isnan(last["LAG"]))
        self.assertGreater(float(last["LEAD"]), 50.0)

    def test_allocator_state_switches_between_risk_on_and_risk_off(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="B")
        risk_on = np.linspace(100.0, 180.0, 200)
        risk_off = np.linspace(180.0, 120.0, 60)
        spy = pd.Series(np.concatenate([risk_on, risk_off]), index=dates)
        qqq = pd.Series(np.concatenate([risk_on, risk_off]), index=dates)
        vix = pd.Series(np.concatenate([np.full(200, 16.0), np.full(60, 32.0)]), index=dates)
        close_px = pd.DataFrame(
            {
                "AGG1": spy * 1.1,
                "AGG2": spy * 1.2,
            },
            index=dates,
        )
        global_data = {
            "SPY": pd.DataFrame({"close": spy}, index=dates),
            "QQQ": pd.DataFrame({"close": qqq}, index=dates),
            "VIX": pd.DataFrame({"close": vix}, index=dates),
        }
        cfg = {
            "allocator": {
                "regime_symbol": "SPY",
                "confirm_symbol": "QQQ",
                "vix_symbol": "VIX",
                "regime_ma_days": 50,
                "breadth_ma_days": 50,
                "breadth_symbols": ["AGG1", "AGG2"],
                "risk_on_breadth": 0.5,
                "risk_off_breadth": 0.5,
                "vix_risk_on_max": 20.0,
                "vix_risk_off_min": 28.0,
            },
            "sleeves": {"aggressive": {"symbols": ["AGG1", "AGG2"]}},
        }

        state = _build_allocator_state(close_px, global_data, cfg)
        self.assertEqual(state.iloc[150], "risk_on")
        self.assertEqual(state.iloc[-1], "risk_off")

    def test_allocator_target_schedule_selects_active_sleeve_only(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="B")
        up = pd.Series(np.linspace(100.0, 180.0, len(dates)), index=dates)
        flat = pd.Series(np.linspace(100.0, 102.0, len(dates)), index=dates)
        down = pd.Series(np.linspace(120.0, 90.0, len(dates)), index=dates)
        data = {
            "SPY": pd.DataFrame({"close": up, "open": up}, index=dates),
            "QQQ": pd.DataFrame({"close": up, "open": up}, index=dates),
            "VIX": pd.DataFrame({"close": pd.Series(np.full(len(dates), 16.0), index=dates)}, index=dates),
            "TQQQ": pd.DataFrame({"close": up * 1.5, "open": up * 1.5}, index=dates),
            "SOXL": pd.DataFrame({"close": up * 1.4, "open": up * 1.4}, index=dates),
            "QQQ_NEU": pd.DataFrame({"close": flat, "open": flat}, index=dates),
            "SPY_NEU": pd.DataFrame({"close": flat * 1.01, "open": flat * 1.01}, index=dates),
            "SHY": pd.DataFrame({"close": flat, "open": flat}, index=dates),
            "IEF": pd.DataFrame({"close": down, "open": down}, index=dates),
        }
        cfg = {
            "rebalance_freq": "W",
            "global_symbols": ["SPY", "QQQ", "VIX"],
            "allocator": {
                "regime_symbol": "SPY",
                "confirm_symbol": "QQQ",
                "vix_symbol": "VIX",
                "regime_ma_days": 50,
                "breadth_ma_days": 50,
                "breadth_symbols": ["TQQQ", "SOXL"],
                "risk_on_breadth": 0.5,
                "risk_off_breadth": 0.25,
                "vix_risk_on_max": 20.0,
                "vix_risk_off_min": 28.0,
                "risk_on_sleeve": "aggressive",
                "neutral_sleeve": "neutral",
                "risk_off_sleeve": "defensive",
            },
            "sleeves": {
                "aggressive": {
                    "symbols": ["TQQQ", "SOXL"],
                    "target_count": 1,
                    "lookbacks": {"21": 1.0},
                    "absolute_momentum_lookback": 21,
                    "absolute_momentum_min": 0.0,
                    "trend_ma_days": 20,
                    "fast_ma_days": 10,
                    "require_fast_above_trend": False,
                },
                "neutral": {
                    "symbols": ["QQQ_NEU", "SPY_NEU"],
                    "target_count": 1,
                    "lookbacks": {"21": 1.0},
                    "absolute_momentum_lookback": 21,
                    "absolute_momentum_min": -1.0,
                    "trend_ma_days": 20,
                    "fast_ma_days": 10,
                    "require_fast_above_trend": False,
                },
                "defensive": {
                    "symbols": ["SHY", "IEF"],
                    "target_count": 1,
                    "lookbacks": {"21": 1.0},
                    "absolute_momentum_lookback": 21,
                    "absolute_momentum_min": -1.0,
                    "trend_ma_days": 20,
                    "fast_ma_days": 10,
                    "require_fast_above_trend": False,
                },
            },
        }

        close_px, open_px, target_schedule, state = _build_allocator_target_schedule(data, cfg)
        self.assertFalse(close_px.empty)
        self.assertFalse(open_px.empty)
        self.assertFalse(target_schedule.empty)
        self.assertEqual(state.iloc[-1], "risk_on")
        last = target_schedule.iloc[-1].dropna()
        self.assertTrue(all(sym in {"TQQQ", "SOXL"} for sym in last.index))
        self.assertAlmostEqual(float(last.sum()), 1.0, places=6)

    def test_exposure_scalar_reduces_gross_under_proxy_drawdown(self) -> None:
        dates = pd.date_range("2024-01-01", periods=260, freq="B")
        up = np.linspace(100.0, 150.0, 220)
        down = np.linspace(150.0, 110.0, 40)
        proxy = pd.Series(np.concatenate([up, down]), index=dates)
        close_px = pd.DataFrame({"AAA": proxy}, index=dates)
        cfg = {
            "exposure_control": {
                "base_gross_exposure": 1.0,
                "proxy_symbols": ["AAA"],
                "drawdown_brake_start_pct": 0.10,
                "drawdown_brake_full_pct": 0.20,
                "drawdown_min_gross_exposure": 0.35,
            }
        }

        scalar = _build_exposure_scalar(close_px, {}, cfg)
        self.assertAlmostEqual(float(scalar.iloc[150]), 1.0, places=6)
        self.assertLess(float(scalar.iloc[-1]), 1.0)
        self.assertGreaterEqual(float(scalar.iloc[-1]), 0.35)

    def test_exposure_scalar_applies_vol_target_cap(self) -> None:
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        proxy = pd.Series(100.0, index=dates)
        proxy.iloc[1::2] = 110.0
        proxy.iloc[2::2] = 90.0
        close_px = pd.DataFrame({"AAA": proxy.ffill()}, index=dates)
        cfg = {
            "exposure_control": {
                "base_gross_exposure": 1.0,
                "proxy_symbols": ["AAA"],
                "vol_target_annual": 0.25,
                "vol_lookback_days": 10,
                "min_gross_exposure": 0.20,
                "max_gross_exposure": 1.0,
            }
        }

        scalar = _build_exposure_scalar(close_px, {}, cfg)
        self.assertLess(float(scalar.iloc[-1]), 1.0)
        self.assertGreaterEqual(float(scalar.iloc[-1]), 0.20)


if __name__ == "__main__":
    unittest.main()
