import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.fundamental_loader import _extract_quarter_record
from scripts.refresh_missing_fundamental_shares import find_tickers_missing_shares


class _FakeQuery:
    def __init__(self, facts):
        self._facts = list(facts)
        self._statement_type = None
        self._period_type = None
        self._concept = None

    def by_statement_type(self, value):
        self._statement_type = value
        return self

    def by_period_type(self, value):
        self._period_type = value
        return self

    def by_concept(self, value):
        self._concept = value
        return self

    def to_dict(self, orient):
        assert orient == "records"
        out = []
        for row in self._facts:
            if self._statement_type and row.get("statement_type") != self._statement_type:
                continue
            if self._period_type and row.get("period_type") != self._period_type:
                continue
            if self._concept and row.get("concept") != self._concept:
                continue
            out.append(dict(row))
        return out


class _FakeXBRL:
    def __init__(self, facts):
        self._facts = list(facts)
        self.facts = list(facts)

    def query(self):
        return _FakeQuery(self._facts)


class _FakeFiling:
    def __init__(self, facts):
        self.form = "10-Q"
        self.filing_date = "2024-05-02"
        self.period_of_report = "2024-03-31"
        self._xbrl = _FakeXBRL(facts)

    def xbrl(self):
        return self._xbrl


class FundamentalLoaderTests(unittest.TestCase):
    def test_extract_quarter_record_persists_shares_outstanding(self) -> None:
        facts = [
            {
                "concept": "Revenues",
                "statement_type": "IncomeStatement",
                "period_type": "duration",
                "value": 1000.0,
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
            },
            {
                "concept": "NetIncomeLoss",
                "statement_type": "IncomeStatement",
                "period_type": "duration",
                "value": 120.0,
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
            },
            {
                "concept": "EarningsPerShareDiluted",
                "statement_type": "IncomeStatement",
                "period_type": "duration",
                "value": 1.25,
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
            },
            {
                "concept": "WeightedAverageNumberOfDilutedSharesOutstanding",
                "statement_type": "EntityInformation",
                "period_type": "duration",
                "value": 96.0,
                "period_start": "2024-01-01",
                "period_end": "2024-03-31",
            },
            {
                "concept": "EntityCommonStockSharesOutstanding",
                "statement_type": "EntityInformation",
                "period_type": "instant",
                "value": 98.0,
                "period_end": "2024-03-31",
            },
        ]

        row = _extract_quarter_record(_FakeFiling(facts), request_pause_sec=0.0)
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["quarter_end"], "2024-03-31")
        self.assertEqual(row["filing_date"], "2024-05-02")
        self.assertEqual(row["revenue"], 1000.0)
        self.assertEqual(row["net_income"], 120.0)
        self.assertEqual(row["eps"], 1.25)
        self.assertEqual(row["shares_outstanding"], 96.0)

    def test_find_tickers_missing_shares_detects_missing_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "ticker=AAA").mkdir()
            (root / "ticker=BBB").mkdir()
            (root / "ticker=CCC").mkdir()

            pd.DataFrame({"revenue": [1.0]}).to_parquet(root / "ticker=AAA" / "fundamentals.parquet")
            pd.DataFrame({"shares_outstanding": [42.0]}).to_parquet(root / "ticker=BBB" / "fundamentals.parquet")
            pd.DataFrame({"shares_outstanding": [None]}).to_parquet(root / "ticker=CCC" / "fundamentals.parquet")

            missing = find_tickers_missing_shares(root)
            self.assertEqual(missing, ["AAA", "CCC"])


if __name__ == "__main__":
    unittest.main()
