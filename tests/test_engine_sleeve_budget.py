import unittest

from execution.engine import _Candidate, _apply_sleeve_budgets


class EngineSleeveBudgetTests(unittest.TestCase):
    def test_apply_sleeve_budgets_prefers_weighted_mix(self) -> None:
        candidates = [
            _Candidate("A", 100.0, 95.0, 99.0, "S", 1, sleeve="breakout"),
            _Candidate("B", 100.0, 95.0, 98.0, "S", 1, sleeve="breakout"),
            _Candidate("C", 100.0, 95.0, 97.0, "S", 1, sleeve="continuation"),
            _Candidate("D", 100.0, 95.0, 96.0, "S", 1, sleeve="recovery"),
            _Candidate("E", 100.0, 95.0, 95.0, "S", 1, sleeve="recovery"),
        ]

        selected = _apply_sleeve_budgets(
            candidates,
            sleeve_weights={"breakout": 0.5, "continuation": 0.25, "recovery": 0.25},
            open_slots=4,
        )

        self.assertEqual(len(selected), 4)
        sleeves = [c.sleeve for c in selected]
        self.assertIn("breakout", sleeves)
        self.assertIn("continuation", sleeves)
        self.assertIn("recovery", sleeves)
        scores = [c.score for c in selected]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_apply_sleeve_budgets_can_run_strict(self) -> None:
        candidates = [
            _Candidate("A", 100.0, 95.0, 99.0, "S", 1, sleeve="continuation"),
            _Candidate("B", 100.0, 95.0, 98.0, "S", 1, sleeve="continuation"),
            _Candidate("C", 100.0, 95.0, 97.0, "S", 1, sleeve="continuation"),
        ]

        selected = _apply_sleeve_budgets(
            candidates,
            sleeve_weights={"breakout": 1.0},
            open_slots=3,
            allow_overfill=False,
        )

        self.assertEqual(len(selected), 0)


if __name__ == "__main__":
    unittest.main()
