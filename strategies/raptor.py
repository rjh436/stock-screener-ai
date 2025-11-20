from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyRaptor(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "rsi_limit": 10,
            "time_stop": 20
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 25: return None
        row = df.iloc[i]
        
        rsi_lim = self.params["rsi_limit"]

        # Regime uptrend
        if not (row["close"] > row["ema100"]):
            return None
        if not (row["ema100"] > df.iloc[i - 5]["ema100"]):
            return None

        # ATR squeeze
        if not (row["atr14"] < row["atr14_ma20"]):
            return None

        # Price at or below lower BB
        if not (row["close"] <= row["bb_lower"]):
            return None

        # RSI pullback
        if row["rsi2"] > rsi_lim:
            return None

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            return None

        lowest5_prev = df.iloc[i - 1]["lowest5"]
        if pd.isna(lowest5_prev):
            return None

        stop_price = lowest5_prev
        if stop_price <= 0:
            return None

        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        if row["low"] < stop_price:
            return True

        # Profit target: close above middle BB
        if row["close"] > row["bb_mid"]:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
