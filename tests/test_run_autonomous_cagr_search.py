from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_autonomous_cagr_search import _dedupe_configs, _load_frontier_candidates


class RunAutonomousCagrSearchTests(unittest.TestCase):
    def test_dedupe_configs_keeps_unique(self) -> None:
        configs = [{"name": "A", "x": 1}, {"name": "A", "x": 1}, {"name": "B", "x": 2}]
        out = _dedupe_configs(configs)
        self.assertEqual(2, len(out))
        self.assertEqual("A", out[0]["name"])
        self.assertEqual("B", out[1]["name"])

    def test_load_frontier_candidates_sorts_by_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "frontier.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"score": 2.0, "config": {"name": "B"}}),
                        json.dumps({"score": 5.0, "config": {"name": "A"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = _load_frontier_candidates(path, 5)
            self.assertEqual("A", out[0]["name"])
            self.assertEqual("B", out[1]["name"])


if __name__ == "__main__":
    unittest.main()
