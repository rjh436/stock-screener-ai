from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyMindful(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "time_stop": 9
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 10: return None
        window = df.iloc[i - 9 : i + 1]
        row = df.iloc[i]

        if (window["close"] < window["sma20"]).any():
            return None

        if not (window["high"] > window["kc_upper"]).any():
            return None

        atr14 = row["atr14"]
        if atr14 <= 1e-4 or pd.isna(atr14):
            return None

        sma20 = row["sma20"]
        if pd.isna(sma20):
            return None

        entry_price = float(sma20)
        stop_price = entry_price - 2 * atr14
        if entry_price <= 0 or stop_price <= 0:
            return None

        return {"entry_price": entry_price, "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        target_price = entry_price + 2 * (entry_price - stop_price)

        if row["high"] >= target_price:
            return True

        if row["low"] <= stop_price:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
