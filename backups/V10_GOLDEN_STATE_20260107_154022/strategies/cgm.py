from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyCGM10(BaseStrategy):
    def __init__(self, params: Dict = None):
        defaults = {
            "vol_mult_bx": 2.0,
            "vol_mult_ptp": 1.5,
            "rsi_ptp": 5,
            "time_stop": 25
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 60: return None
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        vol_mult_bx = self.params["vol_mult_bx"]
        vol_mult_ptp = self.params["vol_mult_ptp"]
        rsi_ptp = self.params["rsi_ptp"]

        atr = row["atr14"]
        if atr <= 1e-4 or pd.isna(atr):
            atr = row["close"] * 0.01

        # Strategy 1: Breakaway Expansion (BX)
        bx_cond_1 = row["high"] > df.iloc[i - 1]["highest55"]
        bx_cond_2 = row["atr14"] > row["atr14_ma20"]
        bx_cond_3 = row["volume"] >= vol_mult_bx * df.iloc[i - 1]["vol_ma20"]
        is_bx_setup = bx_cond_1 and bx_cond_2 and bx_cond_3

        # Strategy 2: Power Trend Pullback (PTP)
        ptp_cond_1 = (row["ema20"] > row["sma50"]) and (row["sma50"] > df.iloc[i - 5]["sma50"])
        ptp_cond_2 = prev["low"] <= prev["ema20"]
        ptp_cond_3 = row["volume"] >= vol_mult_ptp * df.iloc[i - 1]["vol_ma20"]
        ptp_cond_4 = row["rsi2"] <= rsi_ptp
        is_ptp_setup = ptp_cond_1 and ptp_cond_2 and ptp_cond_3 and ptp_cond_4

        if not (is_bx_setup or is_ptp_setup):
            return None

        stop_price = row["close"] - 2.5 * atr
        if stop_price <= 0:
            return None

        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        time_stop = self.params["time_stop"]

        if row["low"] < stop_price:
            return True

        if row["close"] < row["ema20"]:
            return True

        if (i - entry_i) >= time_stop:
            return True

        return False
