from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
_DEFAULT_EXCLUDE_PREFIXES = ("$",)


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def list_cached_symbols(
    cache_dir: Optional[str | Path] = None,
    *,
    exclude_prefixes: Sequence[str] = _DEFAULT_EXCLUDE_PREFIXES,
    max_symbols: Optional[int] = None,
) -> List[str]:
    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
    if not root.exists():
        return []

    excluded = tuple(_normalize_symbol(x) for x in (exclude_prefixes or ()))
    symbols: List[str] = []
    seen = set()
    for path in sorted(root.glob("*.parquet")):
        sym = _normalize_symbol(path.stem)
        if not sym:
            continue
        if excluded and any(sym.startswith(prefix) for prefix in excluded):
            continue
        if sym in seen:
            continue
        seen.add(sym)
        symbols.append(sym)
        if max_symbols is not None and max_symbols > 0 and len(symbols) >= int(max_symbols):
            break
    return sorted(symbols)


def load_symbol_file(path: str | Path) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Symbol file not found: {file_path}")

    rows = file_path.read_text(encoding="utf-8").splitlines()
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = [",".join(row) for row in reader if row]

    symbols: List[str] = []
    seen = set()
    for raw in rows:
        parts: Iterable[str]
        if "," in raw:
            parts = raw.split(",")
        else:
            parts = [raw]
        for part in parts:
            sym = _normalize_symbol(part)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            symbols.append(sym)
    return symbols
