from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import pandas as pd

from strategies.base import BaseStrategy
from strategies.piotroski_value_overlay import PiotroskiValueOverlay
from execution.rank_scoring import compute_momentum_rank, compute_value_rank
from execution.rebalance_engine import select_target_portfolio


class SeparateValueMomentumStrategy(BaseStrategy):
    """
    Build separate value and momentum sleeves, then combine at portfolio level.

    This class is used by factor scripts for periodic target construction.
    """

    def __init__(self, params: Dict):
        super().__init__(params)
        self.value_count = int(self.params.get("value_target_count", 12) or 12)
        self.momentum_count = int(self.params.get("momentum_target_count", 12) or 12)
        self.hold_buffer_mult = float(self.params.get("hold_buffer_mult", 1.25) or 1.25)
        self.value_weight = float(self.params.get("value_weight", 0.5) or 0.5)
        self.momentum_weight = float(self.params.get("momentum_weight", 0.5) or 0.5)
        self.overlay = PiotroskiValueOverlay(params)

    def build_target(
        self,
        frame: pd.DataFrame,
        date_idx: object,
        existing_symbols: Optional[Iterable[str]] = None,
    ) -> Dict[str, object]:
        value_rank = compute_value_rank(frame, date_idx=date_idx, params=self.params)
        momentum_rank = compute_momentum_rank(frame, date_idx=date_idx, params=self.params)

        existing = list(existing_symbols or [])
        value_sel = select_target_portfolio(
            value_rank.get("value_rank", pd.Series(dtype=float)),
            target_count=self.value_count,
            existing_symbols=existing,
            hold_buffer_mult=self.hold_buffer_mult,
        )
        # Optional quality filter on value candidates.
        value_filtered = self.overlay.filter_symbols(frame, value_sel, date_idx)
        if value_filtered:
            value_sel = value_filtered[: self.value_count]

        momentum_sel = select_target_portfolio(
            momentum_rank.get("momentum_rank", pd.Series(dtype=float)),
            target_count=self.momentum_count,
            existing_symbols=existing,
            hold_buffer_mult=self.hold_buffer_mult,
        )

        weights: Dict[str, float] = {}
        if value_sel:
            w = self.value_weight / float(len(value_sel))
            for sym in value_sel:
                weights[sym] = weights.get(sym, 0.0) + w
        if momentum_sel:
            w = self.momentum_weight / float(len(momentum_sel))
            for sym in momentum_sel:
                weights[sym] = weights.get(sym, 0.0) + w

        total = sum(weights.values())
        if total > 0:
            weights = {k: (v / total) for k, v in weights.items()}

        return {
            "value_selected": value_sel,
            "momentum_selected": momentum_sel,
            "weights": weights,
            "value_rank": value_rank,
            "momentum_rank": momentum_rank,
        }

    # Event-driven hooks are intentionally no-op for this periodic sleeve.
    def entry(self, df: pd.DataFrame, i: int):
        return None

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        return False

    def pyramid(self, *args, **kwargs):
        return None
