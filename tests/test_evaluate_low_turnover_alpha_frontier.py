from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_low_turnover_alpha_frontier import (
    _apply_realistic_costs,
    _load_frontier_rows,
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
            "metrics_5y": {"cagr_pct": 25.0, "max_dd_pct": 20.0, "trades_per_year": 80.0, "same_day_open_entries": 0},
            "metrics_10y": {"cagr_pct": 20.0, "max_dd_pct": 25.0, "trades_per_year": 90.0, "same_day_open_entries": 0},
        }
        high_turnover = {
            "metrics_5y": {"cagr_pct": 25.0, "max_dd_pct": 20.0, "trades_per_year": 180.0, "same_day_open_entries": 0},
            "metrics_10y": {"cagr_pct": 20.0, "max_dd_pct": 25.0, "trades_per_year": 220.0, "same_day_open_entries": 0},
        }
        self.assertGreater(_score(base, 150.0), _score(high_turnover, 150.0))

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


if __name__ == "__main__":
    unittest.main()
