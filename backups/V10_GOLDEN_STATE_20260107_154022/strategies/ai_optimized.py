from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyAIOptimized(BaseStrategy):
    """
    AI-Optimized Strategy: Swing Trading for 10%+ Profits
    
    Philosophy:
    - Enter on deep pullbacks in strong trends
    - HOLD for minimum 10% profit target
    - Use trailing stops above 10% to let winners run unlimited
    
    Entry:
      - Strong trend: EMA50 > EMA200
      - Price above EMA200 (uptrend participation)
      - Pullback to BB lower band (oversold)
      - RSI2 < 10 (extreme short-term oversold)
    
    Exit:
      - Hard stop: 3 ATR below entry
      - Below 10% profit: HOLD (ignore noise, only exit on major trend break)
      - 10-20% profit: EMA20 trailing stop
      - >20% profit: EMA50 trailing stop (wider)
      - Parabolic spike: RSI2 > 98
    """
    def __init__(self, params: Dict = None):
        # Default to the evolved parameters if none provided
        default_params = {
            "stop_loss_atr": 3.0,
            "max_time_stop": 60,  # Extended to 60 days for swing trades
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        # Warmup
        if i < 200: return None
        
        row = df.iloc[i]
        
        # 1. Simple Trend Filter (Close > SMA200)
        if not (row["close"] > row["sma200"]):
            return None

        # 2. Volatility Filter (ATR > 0.5% of Price)
        # Avoid dead stocks that don't move enough to hit targets
        atr = row.get("atr14", 0)
        if atr < (row["close"] * 0.005):
            return None
            
        # 3. Entry Trigger: Double Confluence
        # Require BOTH RSI2 < 10 AND Price below Lower BB
        is_oversold_rsi = (row["rsi2"] < 10)
        is_oversold_bb = (row["close"] < row["bb_lower"])
        
        if not (is_oversold_rsi and is_oversold_bb):
            return None
            
        # Calculate Stop
        stop_price = row["close"] - (atr * 3.0)
        
        if stop_price <= 0: return None
        
        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Calculate current profit %
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        days_held = i - entry_i
        
        # 1. Hard Stop (Intraday)
        if row["low"] < stop_price:
            return True
            
        # 2. Profit Target (+5%)
        if current_profit_pct >= 5.0:
            return True
            
        # 3. Time Stop (Market didn't move)
        if days_held >= 40 and current_profit_pct < 1.0:
            return True
            
        return False
        
        # 4. Time Stop (Market didn't move)
        if days_held >= 40 and current_profit_pct < 1.0:
            return True
            
        return False
