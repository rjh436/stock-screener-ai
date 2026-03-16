import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.custom_universe import list_cached_symbols, load_symbol_file


class CustomUniverseTests(unittest.TestCase):
    def test_list_cached_symbols_normalizes_and_excludes_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aapl.parquet").write_text("", encoding="utf-8")
            (root / "MSFT.parquet").write_text("", encoding="utf-8")
            (root / "$VIX.parquet").write_text("", encoding="utf-8")
            (root / "README.txt").write_text("", encoding="utf-8")

            syms = list_cached_symbols(root)
            self.assertEqual(syms, ["AAPL", "MSFT"])

    def test_load_symbol_file_accepts_text_and_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "symbols.txt"
            text_path.write_text("aapl\nMSFT\nmsft\n", encoding="utf-8")
            self.assertEqual(load_symbol_file(text_path), ["AAPL", "MSFT"])

            csv_path = Path(tmp) / "symbols.csv"
            csv_path.write_text("AAPL,MSFT\nNVDA\n", encoding="utf-8")
            self.assertEqual(load_symbol_file(csv_path), ["AAPL", "MSFT", "NVDA"])
