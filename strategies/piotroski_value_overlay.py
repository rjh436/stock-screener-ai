from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class PiotroskiValueOverlay(BaseStrategy):
    """Apply a Piotroski-style quality filter to value candidates."""

    def __init__(self, params):
        super().__init__(params)
        self.min_f_score = float(self.params.get("min_f_score", 7.0) or 7.0)
        self.min_eps_growth_yoy = float(self.params.get("min_eps_growth_yoy", 0.0) or 0.0)
        self.min_sales_growth_yoy = float(self.params.get("min_sales_growth_yoy", 0.0) or 0.0)

    def filter_symbols(self, frame: pd.DataFrame, symbols: Iterable[str], date_idx: object) -> List[str]:
        if frame is None or frame.empty:
            return []

        if isinstance(frame.index, pd.MultiIndex):
            lvl0 = pd.to_datetime(frame.index.get_level_values(0), errors="coerce")
            target = pd.Timestamp(date_idx)
            eligible = sorted({d for d in lvl0 if pd.notna(d) and d <= target})
            if not eligible:
                return []
            snap = frame.xs(eligible[-1], level=0, drop_level=True)
        elif {"date", "symbol"}.issubset(set(frame.columns)):
            local = frame.copy()
            local["date"] = pd.to_datetime(local["date"], errors="coerce")
            target = pd.Timestamp(date_idx)
            local = local.loc[local["date"] <= target]
            if local.empty:
                return []
            snap = local.loc[local["date"] == local["date"].max()].set_index("symbol")
        else:
            snap = frame.copy()
            if "symbol" in snap.columns:
                snap = snap.set_index("symbol")

        keep: List[str] = []
        symbol_set = {str(s) for s in symbols}
        for sym in symbol_set:
            if sym not in snap.index:
                continue
            row = snap.loc[sym]
            f_score = float(pd.to_numeric(row.get("f_score", np.nan), errors="coerce"))
            eps_yoy = float(pd.to_numeric(row.get("eps_growth_yoy", np.nan), errors="coerce"))
            sales_yoy = float(pd.to_numeric(row.get("sales_growth_yoy", np.nan), errors="coerce"))

            f_score_ok = np.isfinite(f_score) and f_score >= self.min_f_score
            growth_ok = (
                (not np.isfinite(eps_yoy) or eps_yoy >= self.min_eps_growth_yoy)
                and (not np.isfinite(sales_yoy) or sales_yoy >= self.min_sales_growth_yoy)
            )

            if f_score_ok or growth_ok:
                keep.append(sym)
        return keep

    # Event-driven hooks are intentionally no-op for this overlay.
    def entry(self, df: pd.DataFrame, i: int):
        return None

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        return False
