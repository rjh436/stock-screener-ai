from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyConnorsRSI(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "crsi_limit": 15,
            "sma_trend": 200,
            "stop_atr_mult": 2.0,
            "exit_sma": 20,
            "time_stop": 10
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 20: return None
        row = df.iloc[i]
        
        # Dynamic Params
        crsi_lim = self.params["crsi_limit"]
        sma_trend = self.params["sma_trend"]
        stop_mult = self.params["stop_atr_mult"]

        # Trend Filter
        sma_col = f"sma{sma_trend}"
        if sma_col not in row or not (row["close"] > row[sma_col]):
            return None
            
        # Trigger
        if row["crsi"] > crsi_lim:
            return None

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            return None

        stop_price = row["close"] - (stop_mult * atr)
        if stop_price <= 0:
            return None

        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]
        exit_sma = self.params["exit_sma"]
        sma_col = f"sma{exit_sma}"

        if row["low"] < stop_price:
            return True

        # Mean reversion exit
        if sma_col in row and row["close"] > row[sma_col]:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
