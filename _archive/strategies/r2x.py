from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyR2X(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "rsi_threshold_low": 5.0,
            "rsi_threshold_high": 10.0,
            "time_stop": 10
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        row = df.iloc[i]
        
        rsi_low = self.params["rsi_threshold_low"]
        rsi_high = self.params["rsi_threshold_high"]

        # Regime
        if not (row["close"] > row["sma200"]):
            return None

        atr10 = row["atr10"]
        close0 = row["close"]
        if atr10 <= 1e-4 or close0 <= 0 or pd.isna(atr10):
            return None

        atr_pct = (atr10 / max(close0, 1e-6)) * 100.0
        threshold = rsi_low if atr_pct > 2.5 else rsi_high
        
        if row["rsi2"] >= threshold:
            return None

        stop_price = close0 - 2 * atr10
        if stop_price <= 0:
            return None

        return {"entry_price": close0, "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        if row["low"] < stop_price:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
