from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyBreakout(BaseStrategy):
    """
    Volatility Breakout Strategy (High CAGR)
    
    Philosophy:
    Catch the start of major trends by buying 20-day highs on strong volume.
    Ride the trend until it breaks the 20-day EMA.

    Setup:
    1. Breakout: Close > Highest High of last 20 days.
    2. Volume: Volume > 1.5 * AvgVol20.
    3. Trend: Close > SMA200 (Optional, usually good for stocks).

    Exits:
    - Trailing Stop: Close < EMA20.
    - Hard Stop: 2.0 * ATR below entry.
    """
    def __init__(self, params: Dict = None):
        default_params = {
            "breakout_period": 20,
            "vol_mult": 1.5,
            "trail_ema": "ema20",
            "stop_atr_mult": 2.0,
            "avg_win_duration": 30 # Longer hold times
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        # Warmup
        if i < 50: return None
        
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # 1. Breakout Condition
        # We need the highest high of the *previous* 20 days to avoid look-ahead bias?
        # Actually, "Close > Highest20" usually means "Close > Highest of PREVIOUS 20 days".
        # engine.py calculates 'highest20' as rolling max including current row? 
        # Let's check engine.py logic or just use shift.
        # engine.py: df["highest20"] = df["high"].rolling(20).max()
        # So highest20 includes today's high.
        # If Close == Highest20, it's a breakout.
        
        # Safer: Close > Highest(High, 20) shifted by 1.
        # But we don't have shifted columns in row.
        # We can check if row["close"] > df.iloc[i-1]["highest20"]? 
        # Wait, df.iloc[i-1]["highest20"] is the max high of i-20 to i-1.
        # Yes, that's the breakout level.
        
        breakout_level = prev_row["highest20"]
        
        if row["close"] <= breakout_level:
            return None
            
        # 2. Volume Confirmation
        if row["volume"] < (row["vol_ma20"] * self.params["vol_mult"]):
            return None
            
        # 3. Trend Filter (Optional but recommended)
        if row["close"] < row["sma200"]:
            return None

        # Entry
        entry_price = row["close"]
        atr = row.get("atr14", entry_price * 0.02)
        stop_price = entry_price - (atr * self.params["stop_atr_mult"])
        
        return {
            "entry_price": entry_price,
            "stop_price": stop_price
        }

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # 1. Hard Stop
        if row["low"] < stop_price:
            return True
            
        # 2. Trailing Stop (Trend Following)
        # Close < EMA20
        trail_col = self.params["trail_ema"]
        if row["close"] < row.get(trail_col, row["sma20"]):
            return True
            
        return False
