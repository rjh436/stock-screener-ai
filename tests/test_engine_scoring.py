import unittest

from execution.engine import _score_row_dual_core


class EngineScoringTests(unittest.TestCase):
    def test_missing_fundamentals_use_neutral_or_capped_proxy(self) -> None:
        score_neutral, tech_neutral, fund_neutral, has_fund_neutral = _score_row_dual_core(
            rsi14=70.0,
            bb_width=0.15,
            natr=2.0,
            close_px=100.0,
            high_52w=105.0,
            weights={"rsi_factor": 1.0},
            rs_rating=90.0,
            eps_growth_qoq=float("nan"),
            eps_growth_yoy=float("nan"),
            sales_growth_yoy=float("nan"),
            institutional_sponsorship=float("nan"),
            gap_pct=1.0,
            volume=1_000_000.0,
            vol_ma50=900_000.0,
            technical_weight=0.60,
            fundamental_weight=0.40,
            return_components=True,
        )
        self.assertFalse(has_fund_neutral)
        self.assertEqual(fund_neutral, 50.0)
        self.assertGreater(score_neutral, 0.0)
        self.assertGreater(tech_neutral, fund_neutral)

        score_proxy, _, fund_proxy, has_fund_proxy = _score_row_dual_core(
            rsi14=70.0,
            bb_width=0.15,
            natr=2.0,
            close_px=100.0,
            high_52w=105.0,
            weights={"rsi_factor": 1.0},
            rs_rating=90.0,
            eps_growth_qoq=float("nan"),
            eps_growth_yoy=float("nan"),
            sales_growth_yoy=float("nan"),
            institutional_sponsorship=float("nan"),
            gap_pct=6.0,
            volume=3_000_000.0,
            vol_ma50=1_000_000.0,
            technical_weight=0.60,
            fundamental_weight=0.40,
            return_components=True,
        )
        self.assertFalse(has_fund_proxy)
        self.assertEqual(fund_proxy, 60.0)
        self.assertGreater(score_proxy, score_neutral)


if __name__ == "__main__":
    unittest.main()
