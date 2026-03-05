import unittest

import pandas as pd

from execution.rebalance_engine import (
    apply_hold_buffer,
    run_periodic_rebalance,
    select_target_portfolio,
)


class RebalanceEngineTests(unittest.TestCase):
    def test_hold_buffer_keeps_existing_ranked_names(self) -> None:
        scores = pd.Series(
            [100, 95, 90, 85, 80, 75],
            index=["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"],
        )
        existing = ["EEE", "ZZZ"]
        keepers = apply_hold_buffer(existing, scores, target_count=4, hold_buffer_mult=1.5)

        self.assertIn("EEE", keepers)
        self.assertNotIn("ZZZ", keepers)

        selected = select_target_portfolio(scores, target_count=4, existing_symbols=existing, hold_buffer_mult=1.5)
        self.assertEqual(len(selected), 4)
        self.assertIn("EEE", selected)

    def test_run_periodic_rebalance_smoke(self) -> None:
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100 + (i * 0.4) for i in range(len(dates))],
                "BBB": [95 + (i * 0.2) for i in range(len(dates))],
                "CCC": [90 + (i * 0.1) for i in range(len(dates))],
                "DDD": [85 + ((-1) ** i) * 0.2 for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [90.0] * len(dates),
                "BBB": [80.0] * len(dates),
                "CCC": [70.0] * len(dates),
                "DDD": [60.0] * len(dates),
            },
            index=dates,
        )

        out = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=2,
            transaction_cost_bps=2.0,
            turnover_budget=0.50,
            start_cash=100000.0,
        )

        self.assertIn("final_value", out)
        self.assertIn("cagr", out)
        self.assertIn("max_drawdown_pct", out)
        self.assertGreater(len(out.get("equity_curve", [])), 10)
        self.assertGreaterEqual(out.get("total_trades", 0), 1)

    def test_missing_price_days_do_not_mark_position_to_zero(self) -> None:
        dates = pd.date_range("2024-01-01", periods=66, freq="B")
        aaa = [100.0 + (i * 0.3) for i in range(len(dates))]
        # Simulate a temporary data hole after the position is established.
        for i in range(25, 35):
            aaa[i] = float("nan")
        prices = pd.DataFrame(
            {
                "AAA": aaa,
                "BBB": [90.0 + (i * 0.1) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [95.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )

        out = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=2.0,
            turnover_budget=1.0,
            start_cash=100000.0,
        )

        # Without last-known-price valuation this path can show an extreme synthetic drawdown.
        self.assertLess(float(out.get("max_drawdown_pct", 1.0)), 0.25)
        self.assertGreater(float(out.get("final_value", 0.0)), 80000.0)

    def test_risk_scalar_allows_cash_buffer(self) -> None:
        dates = pd.date_range("2024-01-01", periods=66, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0 + (i * 0.2) for i in range(len(dates))],
                "BBB": [95.0 + (i * 0.1) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [90.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )
        risk = pd.Series(0.5, index=dates)

        out = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            risk_scalar_by_date=risk,
        )

        audit = out.get("audit_report", {}) or {}
        self.assertLessEqual(float(audit.get("max_gross_exposure_pct", 1.0)), 0.60)

    def test_hard_stop_improves_capital_preservation(self) -> None:
        dates = pd.date_range("2024-01-01", periods=84, freq="B")
        # Price rises first, then suffers a large decline.
        aaa = [100.0 + (i * 0.4) for i in range(30)] + [112.0 - (i * 1.1) for i in range(54)]
        prices = pd.DataFrame(
            {
                "AAA": aaa,
                "BBB": [90.0 + (i * 0.05) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [95.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )

        no_stop = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=2.0,
            turnover_budget=1.0,
            start_cash=100000.0,
        )
        with_stop = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=2.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            hard_stop_pct=0.10,
        )

        self.assertGreater(float(with_stop.get("final_value", 0.0)), float(no_stop.get("final_value", 0.0)))

    def test_default_execution_uses_t_plus_one_and_audits_same_day(self) -> None:
        dates = pd.date_range("2024-01-01", periods=90, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0 + (i * 0.2) for i in range(len(dates))],
                "BBB": [100.0 - (i * 0.05) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [90.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )

        t1 = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
        )
        same_day = int((t1.get("audit_report", {}) or {}).get("same_day_open_entries", 0) or 0)
        self.assertEqual(same_day, 0)
        for row in t1.get("rebalance_log", []):
            sig = pd.Timestamp(row.get("signal_date"))
            exe = pd.Timestamp(row.get("date"))
            self.assertLess(sig, exe)

        t0 = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            execution_lag_days=0,
        )
        same_day0 = int((t0.get("audit_report", {}) or {}).get("same_day_open_entries", 0) or 0)
        self.assertGreaterEqual(same_day0, 1)

    def test_run_periodic_rebalance_supports_daily_and_weekly_frequency(self) -> None:
        dates = pd.date_range("2024-01-01", periods=40, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0 + (i * 0.25) for i in range(len(dates))],
                "BBB": [95.0 + (i * 0.10) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [90.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )

        daily = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="D",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
        )
        weekly = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="W",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
        )

        self.assertGreater(len(daily.get("rebalance_log", [])), len(weekly.get("rebalance_log", [])))
        self.assertEqual(int((daily.get("audit_report", {}) or {}).get("same_day_open_entries", 0) or 0), 0)
        self.assertEqual(int((weekly.get("audit_report", {}) or {}).get("same_day_open_entries", 0) or 0), 0)

    def test_execution_prices_can_differ_from_mark_prices(self) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        close_prices = pd.DataFrame(
            {
                "AAA": [100.0 + i for i in range(len(dates))],
            },
            index=dates,
        )
        open_prices = pd.DataFrame(
            {
                "AAA": [90.0 + i for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame({"AAA": [100.0] * len(dates)}, index=dates)

        out = run_periodic_rebalance(
            close_prices,
            ranks,
            execution_prices=open_prices,
            rebalance_freq="D",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            execution_lag_days=1,
        )

        rebalance_log = out.get("rebalance_log", [])
        self.assertTrue(rebalance_log)
        first_orders = rebalance_log[0].get("orders", [])
        self.assertTrue(first_orders)
        self.assertEqual(float(first_orders[0]["price"]), float(open_prices.iloc[1, 0]))

    def test_stale_price_exit_liquidates_unquoted_position(self) -> None:
        dates = pd.date_range("2024-01-01", periods=70, freq="B")
        aaa = [100.0 + (i * 0.2) for i in range(len(dates))]
        # Force prolonged quote outage after initial position entry.
        for i in range(28, len(dates)):
            aaa[i] = float("nan")

        prices = pd.DataFrame(
            {
                "AAA": aaa,
                "BBB": [95.0 + (i * 0.05) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [95.0] * len(dates),
                "BBB": [80.0] * len(dates),
            },
            index=dates,
        )

        out = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=1,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            hard_stop_pct=None,
            trend_ma_days=None,
            time_stop_days=None,
            max_stale_price_days=3,
        )
        # Buy then stale-price forced sell.
        self.assertGreaterEqual(int(out.get("total_trades", 0) or 0), 2)

    def test_conviction_weighted_targets_overweight_higher_rank(self) -> None:
        dates = pd.date_range("2024-01-01", periods=66, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": [100.0 + (i * 0.1) for i in range(len(dates))],
                "BBB": [100.0 + (i * 0.1) for i in range(len(dates))],
            },
            index=dates,
        )
        ranks = pd.DataFrame(
            {
                "AAA": [95.0] * len(dates),
                "BBB": [70.0] * len(dates),
            },
            index=dates,
        )
        out = run_periodic_rebalance(
            prices,
            ranks,
            rebalance_freq="M",
            target_count=2,
            transaction_cost_bps=0.0,
            turnover_budget=1.0,
            start_cash=100000.0,
            conviction_weighted=True,
            conviction_power=1.0,
        )
        logs = out.get("rebalance_log", [])
        self.assertTrue(logs)
        first_weights = logs[0].get("weights", {})
        self.assertGreater(float(first_weights.get("AAA", 0.0)), float(first_weights.get("BBB", 0.0)))


if __name__ == "__main__":
    unittest.main()
