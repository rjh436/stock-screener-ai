from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategySwingTrader(BaseStrategy):
    """
    NEW Clean Strategy: 10% Minimum Profit Target
    
    ZERO compromises - designed from scratch for swing trading
    
    Entry (Simple & Effective):
      - Close > EMA200 (uptrend only)
      - RSI14 < 30 (oversold bounce)
      - Volume > 1.5x average (confirmation)
    
    Exit (Profit-First):
      - MINIMUM 10% profit target
      - Above 10%: Trail with EMA20
      - Hard stop: 2.5% below entry
    """
    def __init__(self, params: Dict = None):
        default_params = {
            "stop_loss_pct": 2.5,  # Fixed % stop
            "profit_target": 10.0,   # Minimum profit target
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        
        row = df.iloc[i]
        
        # 1. Uptrend Filter
        if not (row["close"] > row.get("ema200", row["sma200"])):
            return None
        
        # 2. Oversold Signal
        if not (row["rsi14"] < 30):
            return None
        
        # 3. Volume Confirmation
        vol_avg = row.get("avgvol50", 1)
        if vol_avg > 0 and not (row["volume"] > vol_avg * 1.5):
            return None
        
        # Calculate Stop (Fixed 2.5%)
        entry_price = row["close"]
        stop_price = entry_price * 0.975  # 2.5% below
        
        if stop_price <= 0: return None
        
        return {"entry_price": entry_price, "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Current profit %
        profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        
        # 1. Hard Stop
        if row["low"] < stop_price:
            return True
        
        # 2. Time Stop (60 days max)
        if (i - entry_i) >= 60:
            return True
        
        # 3. MINIMUM PROFIT TARGET GATE
        if profit_pct < self.params["profit_target"]:
            # Do NOT exit - hold for target
            # ONLY exception: catastrophic trend break
            if row["close"] < row.get("ema200", row["sma200"]):
                # Fell back below major trend
                return True
            return False  # Otherwise HOLD
        
        # 4. ABOVE TARGET: Trailing Stop
        # Use EMA20 to trail profits
        if row["close"] < row.get("ema20", row["sma20"]):
            return True
        
        return False
