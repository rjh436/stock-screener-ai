from typing import Dict, Optional
import pandas as pd

from .base import BaseStrategy
from .minervini_sepa import MinerviniSEPAStrategy


class GenericStrategy(BaseStrategy):
    """
    Compatibility wrapper for legacy scripts that expect GenericStrategy.
    Delegates entry/exit to MinerviniSEPAStrategy.
    """

    def __init__(self, params: Dict):
        self.params = params or {}
        self._name = self.params.get("name", "GenericStrategy")
        self._delegate = MinerviniSEPAStrategy(self.params)
        super().__init__(self.params)

    @property
    def name(self) -> str:
        return self._name

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        return self._delegate.entry(df, i)

    def exit(
        self,
        df: pd.DataFrame,
        i: int,
        entry_i: int,
        entry_price: float,
        stop_price: float,
    ) -> bool:
        return self._delegate.exit(df, i, entry_i, entry_price, stop_price)
