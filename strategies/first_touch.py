from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyFirstTouch(BaseStrategy):
    """
    'First Touch' Trend Pullback Strategy (Revised)
    
    Philosophy:
    Buying deep pullbacks in established uptrends. We wait for the "rubber band" 
    to stretch (RSI2 < 15) and then snap back.

    Setup:
    1. Trend: Close > SMA200 (Long-term Bull Market).
    2. Pullback: RSI(2) < 15 (Deeply oversold).
    3. Trigger: Close > Open (Reversal Candle).

    Exits:
    - Trailing Stop: Close < EMA10.
    - Hard Stop: 3.0 * ATR below entry.
    """
    def __init__(self, params: Dict = None):
        default_params = {
            "power_move_pct": 0.20,      # 20% move
            "power_move_days": 20,       # in 20 days
            "pullback_ema": "ema20",     # Target EMA for pullback (ema20 or ema50)
            "target1_pct": 0.05,         # Take 50% profit at +5%
            "trail_ema": "ema10",        # Trail remaining 50% with EMA10
            "stop_atr_mult": 2.0,        # Hard stop distance
            "avg_win_duration": 10       # Estimated
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        # Warmup
        if i < 200: return None
        
        row = df.iloc[i]
        
        # 1. Trend Filter (Long Term)
        # Only trade if we are in a bull market for this stock
        if not (row["close"] > row["sma200"]):
            return None

        # 2. The Pullback (Deep Oversold)
        # RSI(2) < 15 is a strong pullback in an uptrend.
        if not (row["rsi2"] < 15):
            return None
            
        # 3. The Trigger (Reversal Candle)
        # Close > Open indicates buyers stepped in today.
        if row["close"] > row["open"]:
            entry_price = row["close"]
            atr = row.get("atr14", entry_price * 0.02)
            stop_price = entry_price - (atr * 3.0) # Wide stop to allow noise
            
            return {
                "entry_price": entry_price,
                "stop_price": stop_price
            }

        return None

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        # NOTE: The 'meta' dict is not currently persisted in the simple engine loop.
        # For this implementation, we will simplify the "Scale Out" logic:
        # We will use the Trailing Stop (EMA10) as the primary exit, 
        # but we will "simulate" the bank-half logic by just raising the stop to Breakeven
        # once we hit +5%.
        
        row = df.iloc[i]
        
        # 1. Hard Stop
        if row["low"] < stop_price:
            return True
            
        # 2. Trailing Stop (Close below EMA10)
        # Only active if we are in profit or after some time? 
        # The strategy says "Hold remaining until close < EMA10".
        if row["close"] < row[self.params["trail_ema"]]:
            return True
            
        return False
