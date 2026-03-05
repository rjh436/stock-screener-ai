from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from strategies.base import BaseStrategy
from execution.rank_scoring import compute_low_vol_rank
from execution.rebalance_engine import select_target_portfolio


class LowVolatilityStrategy(BaseStrategy):
    """
    Rank/rebalance low-volatility sleeve.

    This strategy is consumed by periodic factor portfolio construction.
    """

    def __init__(self, params: Dict):
        super().__init__(params)
        self._target_count = int(self.params.get("target_count", 12) or 12)
        self._hold_buffer_mult = float(self.params.get("hold_buffer_mult", 1.25) or 1.25)

    def check_setup(
        self,
        frame: pd.DataFrame,
        date_idx: object,
        existing_symbols: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        ranked = compute_low_vol_rank(frame, date_idx=date_idx, params=self.params)
        selected = select_target_portfolio(
            ranked["low_vol_rank"],
            target_count=self._target_count,
            existing_symbols=existing_symbols,
            hold_buffer_mult=self._hold_buffer_mult,
        )
        if not selected:
            return {"selected": [], "weights": {}, "ranked": ranked}
        w = 1.0 / float(len(selected))
        weights = {sym: w for sym in selected}
        return {"selected": selected, "weights": weights, "ranked": ranked}

    # Event-driven hooks are intentionally no-op for this periodic sleeve.
    def entry(self, df: pd.DataFrame, i: int):
        return None

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        return False

    def pyramid(self, *args, **kwargs):
        return None
