from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.repo_hygiene_audit import Finding, _check_tracked_scratch_files, _score


class RepoHygieneAuditTests(unittest.TestCase):
    def test_detects_tracked_backup_and_scratch_files(self) -> None:
        findings: list[Finding] = []
        tracked = [
            "app.py.20251206_151927.bak",
            "execution/engine.py.bak",
            "config/generated_strategies.json.bak_before_restore",
            "debug_entry.py",
            "_tmp_test_root.txt",
            "Run_Screener copy.command",
            "codebase_audit.txt",
        ]
        _check_tracked_scratch_files(findings, tracked)
        categories = {f.category for f in findings}
        self.assertIn("tracked_backups", categories)
        self.assertIn("tracked_scratch", categories)

    def test_score_penalizes_new_hygiene_findings(self) -> None:
        findings = [
            Finding(
                severity="medium",
                category="tracked_backups",
                message="Tracked backup files are present in git and obscure the canonical source tree.",
                evidence=["app.py.20251206_151927.bak"],
            ),
            Finding(
                severity="medium",
                category="tracked_scratch",
                message="Tracked scratch/debug/duplicate-launcher files are present in git.",
                evidence=["debug_entry.py"],
            ),
        ]
        score = _score(findings)
        self.assertLess(score["score"], 100)
        self.assertEqual(2, score["counts"]["medium"])


if __name__ == "__main__":
    unittest.main()
