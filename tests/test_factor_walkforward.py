import unittest

import numpy as np
import pandas as pd

from scripts.run_factor_walkforward import (
    _acceptance_snapshot,
    _build_value_proxy_scores,
    _build_low_vol_scores,
    _build_momentum_low_vol_schedule,
    _build_momentum_quality_scores,
    _build_separate_value_momentum_schedule,
    _cross_section_rank_matrix,
    _daily_membership_price_coverage,
    _stitch_test_windows_warm,
)


class FactorWalkforwardTests(unittest.TestCase):
    def test_cross_section_rank_matrix(self) -> None:
        vals = np.array(
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
        valid = np.array([[True, True, True], [True, True, True]], dtype=bool)
        out = _cross_section_rank_matrix(vals, valid_mask=valid, higher_is_better=True, min_names=2)

        self.assertEqual(out.shape, vals.shape)
        self.assertGreater(float(out[0, 2]), float(out[0, 0]))
        self.assertGreater(float(out[1, 0]), float(out[1, 1]))

    def test_build_separate_schedule_outputs_weights(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        mom = pd.DataFrame(
            {
                "AAA": [90.0] * len(idx),
                "BBB": [80.0] * len(idx),
                "CCC": [70.0] * len(idx),
            },
            index=idx,
        )
        val = pd.DataFrame(
            {
                "AAA": [60.0] * len(idx),
                "BBB": [75.0] * len(idx),
                "DDD": [85.0] * len(idx),
            },
            index=idx,
        )

        sched = _build_separate_value_momentum_schedule(
            mom,
            val,
            rebalance_freq="M",
            momentum_count=2,
            value_count=2,
            hold_buffer_mult=1.25,
            momentum_weight=0.5,
            value_weight=0.5,
        )
        self.assertFalse(sched.empty)
        for _, row in sched.iterrows():
            total = float(pd.to_numeric(row, errors="coerce").fillna(0.0).sum())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_build_momentum_low_vol_schedule_outputs_weights(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        mom = pd.DataFrame(
            {
                "AAA": [95.0] * len(idx),
                "BBB": [85.0] * len(idx),
                "CCC": [75.0] * len(idx),
            },
            index=idx,
        )
        low_vol = pd.DataFrame(
            {
                "DDD": [90.0] * len(idx),
                "EEE": [80.0] * len(idx),
                "AAA": [70.0] * len(idx),
            },
            index=idx,
        )

        sched = _build_momentum_low_vol_schedule(
            mom,
            low_vol,
            rebalance_freq="M",
            momentum_count=2,
            low_vol_count=2,
            hold_buffer_mult=1.25,
            momentum_weight=0.75,
            low_vol_weight=0.25,
        )
        self.assertFalse(sched.empty)
        for _, row in sched.iterrows():
            total = float(pd.to_numeric(row, errors="coerce").fillna(0.0).sum())
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_build_momentum_low_vol_schedule_adapts_to_risk_scalar(self) -> None:
        idx = pd.date_range("2024-01-01", periods=80, freq="B")
        mom = pd.DataFrame(
            {
                "AAA": [95.0] * len(idx),
                "BBB": [85.0] * len(idx),
                "CCC": [75.0] * len(idx),
            },
            index=idx,
        )
        low_vol = pd.DataFrame(
            {
                "DDD": [90.0] * len(idx),
                "EEE": [80.0] * len(idx),
                "AAA": [70.0] * len(idx),
            },
            index=idx,
        )
        # Risk-on through first half, risk-off through second half.
        risk = pd.Series(1.0, index=idx)
        risk.iloc[len(idx) // 2 :] = 0.5

        sched = _build_momentum_low_vol_schedule(
            mom,
            low_vol,
            rebalance_freq="M",
            momentum_count=2,
            low_vol_count=2,
            hold_buffer_mult=1.25,
            momentum_weight=0.8,
            low_vol_weight=0.2,
            risk_scalar_by_date=risk,
            adaptive_low_vol_max_weight=0.6,
        )
        self.assertFalse(sched.empty)
        low_vol_symbols = {"DDD", "EEE"}
        wts = []
        for dt, row in sched.iterrows():
            row_num = pd.to_numeric(row, errors="coerce").fillna(0.0)
            total = float(row_num.sum())
            self.assertAlmostEqual(total, 1.0, places=6)
            lv_w = float(row_num.loc[row_num.index.intersection(low_vol_symbols)].sum())
            wts.append((pd.Timestamp(dt), lv_w))

        risk_on = [w for dt, w in wts if dt < idx[len(idx) // 2]]
        risk_off = [w for dt, w in wts if dt >= idx[len(idx) // 2]]
        self.assertTrue(risk_on and risk_off)
        self.assertGreater(float(np.mean(risk_off)), float(np.mean(risk_on)))

    def test_momentum_quality_scores_rank_quality_components_per_column(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        symbols = ["AAA", "BBB"]
        shape = (len(dates), len(symbols))
        close = np.full(shape, 100.0, dtype=np.float32)
        high = np.full(shape, 100.0, dtype=np.float32)
        vol = np.full(shape, 2_000_000.0, dtype=np.float32)
        eps = np.full(shape, np.nan, dtype=np.float32)
        sales = np.full(shape, np.nan, dtype=np.float32)
        inst = np.full(shape, np.nan, dtype=np.float32)

        # Give equal momentum/proximity and only quality differences on final day.
        close[:, 0] = np.linspace(80.0, 100.0, len(dates))
        close[:, 1] = np.linspace(80.0, 100.0, len(dates))
        high[:, :] = 100.0
        eps[-1, 0] = 10000.0
        eps[-1, 1] = 100.0
        sales[-1, 0] = 0.0
        sales[-1, 1] = 100.0
        inst[-1, 0] = 0.0
        inst[-1, 1] = 100.0

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "close": close,
            "high_52w": high,
            "vol_ma20": vol,
            "eps_yoy": eps,
            "sales_yoy": sales,
            "inst": inst,
            "natr": np.full(shape, 2.0, dtype=np.float32),
            "adr": np.full(shape, 2.0, dtype=np.float32),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 10.0,
            "min_adv20": 1.0,
            "min_history_bars": 252,
            "drop_bottom_dv_frac": 0.0,
            "min_rank_names": 2,
            "mom_12_1_weight": 0.0,
            "mom_6_1_weight": 0.0,
            "mom_1_0_penalty_weight": 0.0,
            "proximity_52w_weight": 0.0,
            "quality_growth_weight": 1.0,
        }
        scores = _build_momentum_quality_scores(features, cfg)
        last = pd.to_numeric(scores.iloc[-1], errors="coerce").dropna().sort_values(ascending=False)
        self.assertEqual(last.index[0], "BBB")

    def test_momentum_quality_scores_can_apply_one_month_reversal_penalty(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        symbols = ["CHASED", "STEADY"]
        shape = (len(dates), len(symbols))
        close = np.full(shape, 100.0, dtype=np.float32)
        high = np.full(shape, 200.0, dtype=np.float32)
        vol = np.full(shape, 2_000_000.0, dtype=np.float32)

        # Keep longer momentum broadly similar, but spike CHASED in the last month.
        close[:, 0] = np.linspace(80.0, 100.0, len(dates))
        close[:, 1] = np.linspace(80.0, 100.0, len(dates))
        close[-21:, 0] = np.linspace(100.0, 140.0, 21)
        close[-21:, 1] = np.linspace(100.0, 102.0, 21)

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "close": close,
            "high_52w": high,
            "vol_ma20": vol,
            "eps_yoy": np.zeros(shape, dtype=np.float32),
            "sales_yoy": np.zeros(shape, dtype=np.float32),
            "inst": np.zeros(shape, dtype=np.float32),
            "natr": np.full(shape, 2.0, dtype=np.float32),
            "adr": np.full(shape, 2.0, dtype=np.float32),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 10.0,
            "min_adv20": 1.0,
            "min_history_bars": 252,
            "drop_bottom_dv_frac": 0.0,
            "min_rank_names": 2,
            "mom_12_1_weight": 0.0,
            "mom_6_1_weight": 0.0,
            "mom_1_0_penalty_weight": 1.0,
            "proximity_52w_weight": 0.0,
            "quality_growth_weight": 0.0,
        }
        scores = _build_momentum_quality_scores(features, cfg)
        last = pd.to_numeric(scores.iloc[-1], errors="coerce").dropna().sort_values(ascending=False)
        self.assertEqual(last.index[0], "STEADY")

    def test_momentum_quality_scores_can_use_sector_relative_momentum(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        symbols = ["A1", "A2", "B1", "B2"]
        shape = (len(dates), len(symbols))
        close = np.full(shape, np.nan, dtype=np.float32)
        vol = np.full(shape, 2_000_000.0, dtype=np.float32)

        close[:, 0] = np.linspace(50.0, 200.0, len(dates))   # A1 strongest in sector A
        close[:, 1] = np.linspace(50.0, 180.0, len(dates))   # A2 weaker in sector A
        close[:, 2] = np.linspace(100.0, 120.0, len(dates))  # B1 strongest in sector B
        close[:, 3] = np.linspace(100.0, 110.0, len(dates))  # B2 weaker in sector B

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "close": close,
            "high_52w": np.maximum.accumulate(close, axis=0),
            "vol_ma20": vol,
            "eps_yoy": np.zeros(shape, dtype=np.float32),
            "sales_yoy": np.zeros(shape, dtype=np.float32),
            "inst": np.zeros(shape, dtype=np.float32),
            "natr": np.full(shape, 2.0, dtype=np.float32),
            "adr": np.full(shape, 2.0, dtype=np.float32),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        base_cfg = {
            "min_price": 10.0,
            "min_adv20": 1.0,
            "min_history_bars": 252,
            "drop_bottom_dv_frac": 0.0,
            "min_rank_names": 4,
            "mom_12_1_weight": 1.0,
            "mom_6_1_weight": 0.0,
            "mom_1_0_penalty_weight": 0.0,
            "proximity_52w_weight": 0.0,
            "quality_growth_weight": 0.0,
        }
        sectors = {"A1": "SEC_A", "A2": "SEC_A", "B1": "SEC_B", "B2": "SEC_B"}

        plain = _build_momentum_quality_scores(features, dict(base_cfg))
        sector_rel = _build_momentum_quality_scores(
            features,
            {**base_cfg, "use_sector_relative_momentum": True, "sector_relative_min_group_size": 2},
            symbol_group_map=sectors,
        )
        plain_last = pd.to_numeric(plain.iloc[-1], errors="coerce")
        sector_last = pd.to_numeric(sector_rel.iloc[-1], errors="coerce")

        self.assertGreater(float(plain_last["A2"]), float(plain_last["B1"]))
        self.assertGreater(float(sector_last["B1"]), float(sector_last["A2"]))
        self.assertGreater(float(sector_last["B1"]), float(plain_last["B1"]))

    def test_build_low_vol_scores_prefers_lower_realized_vol(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=220)
        symbols = ["LOW", "HIGH"]
        shape = (len(dates), len(symbols))
        close = np.full(shape, np.nan, dtype=np.float32)
        vol = np.full(shape, 2_000_000.0, dtype=np.float32)

        low_path = 100.0 + np.linspace(0.0, 5.0, len(dates))
        high_path = 100.0 + np.sin(np.linspace(0.0, 60.0, len(dates))) * 8.0
        close[:, 0] = low_path
        close[:, 1] = high_path

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "close": close,
            "high_52w": np.maximum.accumulate(close, axis=0),
            "vol_ma20": vol,
            "eps_yoy": np.full(shape, 10.0, dtype=np.float32),
            "sales_yoy": np.full(shape, 10.0, dtype=np.float32),
            "inst": np.full(shape, 50.0, dtype=np.float32),
            "natr": np.array([[1.0, 4.0]] * len(dates), dtype=np.float32),
            "adr": np.array([[1.0, 4.0]] * len(dates), dtype=np.float32),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 10.0,
            "min_adv20": 1.0,
            "min_history_bars": 100,
            "drop_bottom_dv_frac": 0.0,
            "min_rank_names": 2,
            "low_vol_lookback_days": 63,
            "low_volatility_weight": 0.7,
            "low_natr_weight": 0.2,
            "low_adr_weight": 0.1,
        }
        scores = _build_low_vol_scores(features, cfg)
        last = pd.to_numeric(scores.iloc[-1], errors="coerce")
        self.assertGreater(float(last["LOW"]), float(last["HIGH"]))

    def test_daily_membership_price_coverage_stats(self) -> None:
        features = {
            "membership_mask": np.array(
                [
                    [True, True, False],
                    [True, True, True],
                    [False, True, True],
                ],
                dtype=bool,
            ),
            "close": np.array(
                [
                    [10.0, 20.0, np.nan],
                    [10.0, np.nan, 30.0],
                    [np.nan, 20.0, 30.0],
                ],
                dtype=np.float64,
            ),
        }
        stats = _daily_membership_price_coverage(features)
        self.assertTrue(np.isfinite(float(stats["mean"])))
        self.assertAlmostEqual(float(stats["mean"]), (1.0 + (2.0 / 3.0) + 1.0) / 3.0, places=6)
        self.assertGreaterEqual(float(stats["max"]), float(stats["median"]))

    def test_value_proxy_scores_can_use_earnings_yield(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=300)
        symbols = ["CHEAP", "EXPENSIVE"]
        shape = (len(dates), len(symbols))

        close = np.full(shape, 100.0, dtype=np.float32)
        close[:, 1] = 120.0
        eps_ttm = np.full(shape, 8.0, dtype=np.float32)
        eps_ttm[:, 1] = 2.0

        features = {
            "dates": pd.DatetimeIndex(dates),
            "symbols": symbols,
            "close": close,
            "high_52w": np.maximum.accumulate(close, axis=0),
            "vol_ma20": np.full(shape, 2_000_000.0, dtype=np.float32),
            "eps_yoy": np.zeros(shape, dtype=np.float32),
            "sales_yoy": np.zeros(shape, dtype=np.float32),
            "inst": np.zeros(shape, dtype=np.float32),
            "eps_ttm": eps_ttm,
            "revenue_ttm": np.full(shape, np.nan, dtype=np.float32),
            "net_income_ttm": np.full(shape, np.nan, dtype=np.float32),
            "net_margin_ttm": np.full(shape, np.nan, dtype=np.float32),
            "natr": np.full(shape, 2.0, dtype=np.float32),
            "adr": np.full(shape, 2.0, dtype=np.float32),
            "membership_mask": np.ones(shape, dtype=bool),
        }
        cfg = {
            "min_price": 10.0,
            "min_adv20": 1.0,
            "min_history_bars": 252,
            "drop_bottom_dv_frac": 0.0,
            "min_rank_names": 2,
            "value_rev_weight": 0.0,
            "value_quality_weight": 0.0,
            "value_stability_weight": 0.0,
            "value_earnings_yield_weight": 1.0,
            "value_margin_weight": 0.0,
            "value_min_earnings_yield": -1.0,
        }
        scores = _build_value_proxy_scores(features, cfg)
        last = pd.to_numeric(scores.iloc[-1], errors="coerce")
        self.assertGreater(float(last["CHEAP"]), float(last["EXPENSIVE"]))

    def test_acceptance_snapshot_checks_same_day_contamination(self) -> None:
        full = {"cagr": 0.12, "max_drawdown_pct": 0.25, "audit_report": {"max_gross_exposure_pct": 1.0, "same_day_open_entries": 2}}
        stitched = {"stitched_cagr_pct": 11.0}
        snap = _acceptance_snapshot(full, stitched, stitched)
        self.assertFalse(bool(snap.get("same_day_contamination_ok")))
        self.assertEqual(int(snap.get("same_day_contamination_count", 0)), 2)

    def test_stitch_windows_warm_uses_equity_curve_slice(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=120)
        equity = []
        val = 100_000.0
        for i, dt in enumerate(dates):
            val *= (1.0005 if i < 60 else 1.0010)
            equity.append({"Date": dt.isoformat(), "Equity": float(val)})
        full_run = {"equity_curve": equity}
        windows = [
            ("2024-03-01", "2024-06-01"),
            ("2024-06-01", "2024-09-01"),
        ]
        stitched = _stitch_test_windows_warm(full_run=full_run, windows=windows, test_months=12)
        self.assertEqual(int(stitched.get("fold_count", 0)), 2)
        self.assertTrue(np.isfinite(float(stitched.get("stitched_cagr_pct", np.nan))))

    def test_acceptance_snapshot_prefers_warm_oos_when_configured(self) -> None:
        full = {"cagr": 0.10, "max_drawdown_pct": 0.30, "audit_report": {"max_gross_exposure_pct": 1.0, "same_day_open_entries": 0}}
        cold = {"stitched_cagr_pct": 4.0}
        warm = {"stitched_cagr_pct": 7.0}
        snap = _acceptance_snapshot(
            full,
            stitched_36_12=cold,
            stitched_60_12=cold,
            stitched_60_12_warm=warm,
            gate_cfg={
                "use_warm_restart_for_oos_gate": True,
                "oos_gate_primary_window": "60_12",
                "oos60_gate_20x20_cagr_pct": 6.0,
            },
        )
        self.assertEqual(str(snap.get("oos_gate_mode")), "warm_restart")
        self.assertTrue(bool(snap.get("meets_20x20_gate")))

    def test_acceptance_snapshot_supports_primary_window_selection(self) -> None:
        full = {"cagr": 0.10, "max_drawdown_pct": 0.30, "audit_report": {"max_gross_exposure_pct": 1.0, "same_day_open_entries": 0}}
        stitched_36 = {"stitched_cagr_pct": 6.0}
        stitched_60 = {"stitched_cagr_pct": 3.0}
        stitched_24 = {"stitched_cagr_pct": 8.0}
        snap = _acceptance_snapshot(
            full,
            stitched_36_12=stitched_36,
            stitched_60_12=stitched_60,
            stitched_24_12=stitched_24,
            stitched_24_12_warm=stitched_24,
            stitched_60_12_warm=stitched_60,
            gate_cfg={
                "use_warm_restart_for_oos_gate": True,
                "oos_gate_primary_window": "24_12",
                "oos60_gate_20x20_cagr_pct": 6.0,
            },
        )
        self.assertEqual(str(snap.get("oos_gate_primary_window")), "24_12")
        self.assertEqual(float(snap.get("oos_60_12_cagr_pct")), 8.0)
        self.assertTrue(bool(snap.get("meets_20x20_gate")))


if __name__ == "__main__":
    unittest.main()
