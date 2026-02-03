from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyAIV2(BaseStrategy):
    """
    AI V2: Target or Bust Test Strategy
    Strictly holds for 10% profit or -10% loss.
    """
    def __init__(self, params: Dict = None):
        default_params = {
            "stop_loss_atr": 3.0,
            "max_time_stop": 60,
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        row = df.iloc[i]
        
        # 1. Trend Filter (EMA50 > EMA200)
        if not (row["ema50"] > row["ema200"]):
            return None
            
        # 2. Price relative to Trend (Must be above EMA200)
        if not (row["close"] > row["ema200"]):
            return None

        # 3. Pullback to BB Lower * 1.15 (Moderate Pullback)
        if not (row["close"] <= row["bb_lower"] * 1.15):
            return None
            
        # 4. RSI Confluence (RSI < 30)
        if not (row["rsi2"] < 30):
            return None
            
        atr = row.get("atr14", row["close"]*0.02)
        stop_price = row["close"] - (atr * 3.0)
        
        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        days_held = i - entry_i
        
        # 1. Hard Stop Loss
        if row["low"] < stop_price:
            return True
            
        # 2. Time Stop (60 days)
        if days_held >= 60:
            return True
            
        # 3. PROFIT TARGET GATE (< 10%)
        if current_profit_pct < 10.0:
            # HOLD AGGRESSIVELY
            # Only exit on extreme RSI spike
            if row["rsi2"] > 95:
                return True
            return False
            
        # 4. TRAILING STOP (> 10%)
        if current_profit_pct < 20.0:
            # EMA20 Trailing
            if row["close"] < row.get("ema20", row["sma20"]):
                return True
        else:
            # EMA50 Trailing (Wider)
            if row["close"] < row.get("ema50", row["sma50"]):
                return True
                
        return False
