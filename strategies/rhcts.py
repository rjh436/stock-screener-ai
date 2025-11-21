from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyRHCTS(BaseStrategy):
    def __init__(self, params: Dict = None):
        # Default parameters
        defaults = {
            "vol_threshold": 1_000_000,
            "rsi_limit": 70,
            "crsi_limit": 20,
            "atr_multiplier": 1.0,
            "time_stop": 10
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 6:
            return None

        row = df.iloc[i]
        prev = df.iloc[i - 1]

        # Dynamic Params
        vol_thresh = self.params["vol_threshold"]
        rsi_lim = self.params["rsi_limit"]
        crsi_lim = self.params["crsi_limit"]
        atr_mult = self.params["atr_multiplier"]

        # Pre-filters
        if row.get("vol_ma20", 0) < vol_thresh:
            return None
        if not (row["close"] > row["sma50"]):
            return None
        if not (row["sma50"] > df.iloc[i - 5]["sma50"]):
            return None
        if row["rsi14"] >= rsi_lim:
            return None

        # Entry triggers
        touch_reclaim = (row["low"] <= row["ema20"]) and (row["close"] >= row["ema20"])
        reclaim_from_below = (prev["close"] < prev["ema20"]) and (row["close"] >= row["ema20"])
        if not (touch_reclaim or reclaim_from_below):
            return None

        if row["crsi"] > crsi_lim:
            return None

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            return None

        lowest5_yesterday = df.iloc[i - 1]["lowest5"]
        if pd.isna(lowest5_yesterday):
            return None

        stop_price = min(lowest5_yesterday, row["ema20"] - (atr * atr_mult))
        if stop_price <= 0:
            return None

        # Strategy spec: entry at EMA20, not the close (limit order assumption)
        # Note: In the original backtester, it seemed to use EMA20 as entry price.
        return {"entry_price": row["ema20"], "stop_price": stop_price, "entry_type": "limit"}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        # Stop-loss: intraday low below stop
        if row["low"] < stop_price:
            return True

        # Close below EMA20
        if row["close"] <= row["ema20"]:
            return True

        # Time stop (bars since entry)
        if (i - entry_i) >= time_stop:
            return True

        return False
