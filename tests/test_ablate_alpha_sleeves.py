import unittest

import pandas as pd

from scripts.ablate_alpha_sleeves import _blend_curves, _curve_metrics


class AblateAlphaSleevesTests(unittest.TestCase):
    def test_blend_curves_and_metrics(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        c1 = pd.Series([100000, 101000, 102000, 103000, 104000], index=idx)
        c2 = pd.Series([100000, 100500, 101500, 102500, 103500], index=idx)

        blended = _blend_curves({"a": c1, "b": c2}, {"a": 0.6, "b": 0.4})
        self.assertFalse(blended.empty)
        self.assertAlmostEqual(float(blended.iloc[0]), 100000.0, places=6)

        metrics = _curve_metrics(blended)
        self.assertGreater(metrics["final_value"], 100000.0)
        self.assertGreater(metrics["cagr_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
