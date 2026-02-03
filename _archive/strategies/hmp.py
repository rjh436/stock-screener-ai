from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyHMP(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "rsi_limit": 35,
            "time_stop": 10
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 50: return None
        row = df.iloc[i]
        
        rsi_lim = self.params["rsi_limit"]

        # Strong uptrend
        if not (row["ema50"] > row["ema200"]):
            return None
        if not (row["close"] > row["ema50"]):
            return None

        # Short-term oversold
        if row["rsi5"] > rsi_lim:
            return None

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            return None

        stop_price = row["ema50"] - atr
        if stop_price <= 0:
            return None

        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        if row["low"] < stop_price:
            return True

        # Trail on EMA50
        if row["close"] < row["ema50"]:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
