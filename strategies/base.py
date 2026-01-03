from abc import ABC, abstractmethod
from typing import Dict, Optional
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, params: Dict):
        self.params = params

    @abstractmethod
    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        """
        Check entry conditions for the bar at index `i`.
        Returns a dict with {'entry_price', 'stop_price'} or None.
        """
        pass

    @abstractmethod
    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        """
        Check exit conditions for the bar at index `i`.
        Returns True if the position should be closed. Strategies may also return
        (should_exit, updated_stop_price) to tighten stops dynamically.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
