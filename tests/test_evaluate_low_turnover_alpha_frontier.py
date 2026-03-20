from __future__ import annotations

import json
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path

import pandas as pd

from scripts.evaluate_low_turnover_alpha_frontier import (
    _apply_realistic_costs,
    _limit_prepared_universe,
    _load_frontier_rows,
    _pack_metrics,
    _resolve_config_args,
    _score,
    _write_partial_out,
)


class EvaluateLowTurnoverAlphaFrontierTests(unittest.TestCase):
    def test_load_frontier_rows_reads_config_objects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "frontier.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"config": {"name": "Alpha 1", "type": "superperformance"}}),
                        json.dumps({"config": {"name": "Alpha 2", "type": "superperformance"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _load_frontier_rows(path)
            self.assertEqual(2, len(rows))
            self.assertEqual("Alpha 1", rows[0]["name"])
            self.assertEqual("Alpha 2", rows[1]["name"])

    def test_apply_realistic_costs_overrides_friction(self) -> None:
        class Args:
            transaction_cost_bps = 2.0
            entry_slippage_bps = 10.0
            exit_slippage_bps = 10.0

        cfg = _apply_realistic_costs({"name": "Alpha", "transaction_cost_bps": 0.0}, Args())
        self.assertEqual(2.0, cfg["transaction_cost_bps"])
        self.assertEqual(10.0, cfg["entry_slippage_bps"])
        self.assertEqual(10.0, cfg["exit_slippage_bps"])
        self.assertFalse(cfg["allow_margin"])
        self.assertFalse(cfg["log_regime_skips"])

    def test_score_penalizes_excess_trades(self) -> None:
        base = {
            "metrics_5y": {
                "cagr_pct": 25.0,
                "max_dd_pct": 20.0,
                "sample_years": 5.0,
                "total_trades": 420,
                "trades_per_year": 84.0,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 18.0,
                "stitched_oos_folds": 2,
                "stitched_oos_worst_fold_return_pct": -5.0,
                "worst_24m_cagr_pct": 9.0,
                "median_24m_cagr_pct": 20.0,
                "top2_year_pnl_share": 0.60,
            },
            "metrics_10y": {
                "cagr_pct": 20.0,
                "max_dd_pct": 25.0,
                "sample_years": 10.0,
                "total_trades": 900,
                "trades_per_year": 90.0,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 14.0,
                "stitched_oos_folds": 5,
                "stitched_oos_worst_fold_return_pct": -8.0,
                "worst_24m_cagr_pct": 6.0,
                "median_24m_cagr_pct": 18.0,
                "top2_year_pnl_share": 0.65,
            },
        }
        high_turnover = {
            "metrics_5y": {
                **base["metrics_5y"],
                "total_trades": 900,
                "trades_per_year": 180.0,
            },
            "metrics_10y": {
                **base["metrics_10y"],
                "total_trades": 2200,
                "trades_per_year": 220.0,
            },
        }
        self.assertGreater(_score(base, 150.0), _score(high_turnover, 150.0))

    def test_score_rejects_sparse_trade_histories(self) -> None:
        sparse = {
            "metrics_5y": {
                "cagr_pct": 35.0,
                "max_dd_pct": 12.0,
                "sample_years": 5.0,
                "total_trades": 12,
                "trades_per_year": 2.4,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 10.0,
                "stitched_oos_folds": 2,
                "stitched_oos_worst_fold_return_pct": -4.0,
                "worst_24m_cagr_pct": 8.0,
                "median_24m_cagr_pct": 16.0,
                "top2_year_pnl_share": 0.55,
            },
            "metrics_10y": {
                "cagr_pct": 32.0,
                "max_dd_pct": 15.0,
                "sample_years": 10.0,
                "total_trades": 18,
                "trades_per_year": 1.8,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 9.0,
                "stitched_oos_folds": 5,
                "stitched_oos_worst_fold_return_pct": -10.0,
                "worst_24m_cagr_pct": 5.0,
                "median_24m_cagr_pct": 15.0,
                "top2_year_pnl_share": 0.50,
            },
        }
        self.assertLess(_score(sparse, 150.0), -1_000_000.0)

    def test_score_rejects_fragile_stitched_oos_profile(self) -> None:
        fragile = {
            "metrics_5y": {
                "cagr_pct": 28.0,
                "max_dd_pct": 18.0,
                "sample_years": 5.0,
                "total_trades": 120,
                "trades_per_year": 24.0,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 12.0,
                "stitched_oos_folds": 2,
                "stitched_oos_worst_fold_return_pct": -5.0,
                "worst_24m_cagr_pct": 6.0,
                "median_24m_cagr_pct": 16.0,
                "top2_year_pnl_share": 0.60,
            },
            "metrics_10y": {
                "cagr_pct": 24.0,
                "max_dd_pct": 20.0,
                "sample_years": 10.0,
                "total_trades": 220,
                "trades_per_year": 22.0,
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
                "stitched_oos_cagr_pct": 4.0,
                "stitched_oos_folds": 5,
                "stitched_oos_worst_fold_return_pct": -42.0,
                "worst_24m_cagr_pct": -12.0,
                "median_24m_cagr_pct": 10.0,
                "top2_year_pnl_share": 0.92,
            },
        }
        self.assertLess(_score(fragile, 150.0), -1_000_000.0)

    def test_pack_metrics_adds_robustness_fields(self) -> None:
        equity = 100000.0
        equity_curve = []
        for dt in pd.date_range("2016-01-31", periods=96, freq="ME"):
            equity *= 1.015
            equity_curve.append({"Date": dt.isoformat(), "Equity": round(equity, 4)})
        result = {
            "cagr": 0.20,
            "max_drawdown_pct": 0.18,
            "total_trades": 96,
            "final_value": 430000.0,
            "audit_report": {
                "same_day_open_entries": 0,
                "max_gross_exposure_pct": 1.0,
            },
            "equity_curve": equity_curve,
        }
        metrics = _pack_metrics(result, "2016-01-01", "2023-12-31")
        self.assertIn("stitched_oos_cagr_pct", metrics)
        self.assertIn("stitched_oos_worst_fold_return_pct", metrics)
        self.assertIn("top2_year_pnl_share", metrics)
        self.assertGreater(metrics["sample_years"], 7.9)
        self.assertGreater(metrics["stitched_oos_folds"], 0)

    def test_resolve_config_args_uses_defaults_only_when_empty(self) -> None:
        self.assertEqual(["a.json"], _resolve_config_args(["a.json"]))
        resolved = _resolve_config_args([])
        self.assertGreaterEqual(len(resolved), 1)
        self.assertIn("config/superperformance_alpha_b4.json", resolved)

    def test_write_partial_out_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "partial.json"
            _write_partial_out(
                out_path,
                "pit",
                {"limit": 0, "mode": "sample", "sample_seed": 42, "prepared_symbols": 2, "selected_symbols": 2},
                2.0,
                10.0,
                10.0,
                [
                    {"name": "B", "score": 2.0},
                    {"name": "A", "score": 3.0},
                ],
            )
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual("A", payload["rows"][0]["name"])
            self.assertEqual("pit", payload["membership_source"])

    def test_limit_prepared_universe_samples_deterministically(self) -> None:
        Prepared = namedtuple("Prepared", ["enriched", "all_dates"])
        prepared = Prepared(
            enriched={"A": object(), "B": object(), "C": object(), "D": object()},
            all_dates=[1, 2, 3],
        )
        limited, info = _limit_prepared_universe(prepared, 2, "sample", 7)
        self.assertEqual(2, len(limited.enriched))
        self.assertEqual(4, info["prepared_symbols"])
        self.assertEqual(2, info["selected_symbols"])
        again, again_info = _limit_prepared_universe(prepared, 2, "sample", 7)
        self.assertEqual(sorted(limited.enriched.keys()), sorted(again.enriched.keys()))
        self.assertEqual(info, again_info)


if __name__ == "__main__":
    unittest.main()
