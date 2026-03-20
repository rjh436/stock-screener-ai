from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import autonomous_tuner, autonomous_walkforward


class AutonomousPromotionGateTests(unittest.TestCase):
    @patch.dict(os.environ, {"APEX_START_DATE": "2020-01-01", "APEX_END_DATE": "2025-01-01"}, clear=False)
    def test_autonomous_tuner_rejects_sparse_candidate(self) -> None:
        metrics = {
            "cagr": 42.0,
            "dd": 12.0,
            "trades": 10,
            "pf": 1.8,
            "win_loss_ratio": 4.0,
        }
        self.assertFalse(autonomous_tuner._is_promotable(metrics))

    @patch.dict(os.environ, {"APEX_START_DATE": "2020-01-01", "APEX_END_DATE": "2025-01-01"}, clear=False)
    def test_autonomous_tuner_accepts_robust_candidate(self) -> None:
        metrics = {
            "cagr": 32.0,
            "dd": 14.0,
            "trades": 64,
            "pf": 1.8,
            "win_loss_ratio": 4.0,
        }
        self.assertTrue(autonomous_tuner._is_promotable(metrics))

    @patch.dict(
        os.environ,
        {
            "APEX_WF_TRAIN_START": "2018-01-01",
            "APEX_WF_TRAIN_END": "2023-01-01",
            "APEX_WF_MIN_IS_CAGR": "15",
            "APEX_WF_MAX_IS_DD": "32",
            "APEX_WF_MIN_IS_PF": "1.25",
            "APEX_WF_MIN_IS_TRADES": "120",
            "APEX_WF_OOS_MAX_DD": "35",
        },
        clear=False,
    )
    def test_walkforward_rejects_sparse_in_sample(self) -> None:
        in_sample = {"cagr": 28.0, "dd": 14.0, "pf": 1.7, "trades": 18}
        oos = [
            {
                "label": "oos_1",
                "start": "2023-01-01",
                "end": "2024-01-01",
                "cagr_pct": 12.0,
                "max_drawdown_pct": 16.0,
                "profit_factor": 1.4,
                "total_trades": 28,
                "max_dd_req": 35.0,
                "pass": True,
            }
        ]
        known_gate = {"pass": True}
        self.assertFalse(autonomous_walkforward._promotable(in_sample, oos, known_gate, "penalty"))

    @patch.dict(
        os.environ,
        {
            "APEX_WF_TRAIN_START": "2006-01-01",
            "APEX_WF_TRAIN_END": "2018-12-31",
            "APEX_WF_MIN_IS_CAGR": "15",
            "APEX_WF_MAX_IS_DD": "32",
            "APEX_WF_MIN_IS_PF": "1.25",
            "APEX_WF_MIN_IS_TRADES": "120",
            "APEX_WF_OOS_MAX_DD": "35",
        },
        clear=False,
    )
    def test_walkforward_accepts_shared_gate_and_oos_passes(self) -> None:
        in_sample = {"cagr": 22.0, "dd": 18.0, "pf": 1.6, "trades": 180}
        oos = [
            {
                "label": "oos_2019_2021",
                "start": "2019-01-01",
                "end": "2021-12-31",
                "cagr_pct": 11.0,
                "max_drawdown_pct": 19.0,
                "profit_factor": 1.3,
                "total_trades": 30,
                "max_dd_req": 35.0,
                "pass": True,
            },
            {
                "label": "oos_2022_now",
                "start": "2022-01-01",
                "end": "2025-01-01",
                "cagr_pct": 16.0,
                "max_drawdown_pct": 21.0,
                "profit_factor": 1.4,
                "total_trades": 36,
                "max_dd_req": 35.0,
                "pass": True,
            },
        ]
        known_gate = {"pass": True}
        self.assertTrue(autonomous_walkforward._promotable(in_sample, oos, known_gate, "penalty"))


if __name__ == "__main__":
    unittest.main()
