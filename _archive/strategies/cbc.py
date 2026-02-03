from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyCBC(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "vol_mult": 1.3,
            "rsi_reset": 20,
            "time_stop": 15
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 21: return None
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        prev_idx = i - 1
        
        vol_mult = self.params["vol_mult"]
        rsi_reset = self.params["rsi_reset"]

        # Volatility contraction yesterday
        if df.iloc[prev_idx]["atr14"] >= df.iloc[prev_idx]["atr14_ma20"]:
            return None

        high20_prev = df.iloc[prev_idx]["highest20"]
        if pd.isna(high20_prev):
            return None

        # Yesterday near resistance
        if not (prev["close"] >= 0.98 * high20_prev and prev["close"] <= high20_prev):
            return None

        # RSI reset
        if df.iloc[prev_idx]["rsi2"] > rsi_reset:
            return None

        # Breakout
        if not (row["close"] > high20_prev):
            return None

        # Volume spike
        if not (row["volume"] >= vol_mult * df.iloc[prev_idx]["vol_ma20"]):
            return None

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            return None

        lowest20_prev = df.iloc[prev_idx]["lowest20"]
        if pd.isna(lowest20_prev):
            return None

        stop_price = max(lowest20_prev, high20_prev - atr)
        if stop_price <= 0:
            return None

        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        if row["low"] < stop_price:
            return True

        if row["close"] < row["sma20"]:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
