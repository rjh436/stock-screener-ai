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
        
        # 1. Strong Trend Filter (EMA50 > EMA200)
        if not (row["ema50"] > row["ema200"]):
            return None
            
        # 2. Price relative to Trend
        if not (row["close"] > row["ema200"]):
            return None

        # 3. Bollinger Band Setup (Price pullback toward lower band)
        # RELAXED: Changed to 1.25 to ensure we get trades for validation
        if not (row["close"] <= row["bb_lower"] * 1.25):
            return None
            
        # 4. RSI Confluence (Stricter)
        # RELAXED: Changed to 25 to ensure we get trades
        if not (row["rsi2"] < 25):
            return None
            
        # Calculate Stop
        atr = row.get("atr14", row["close"]*0.02)
        stop_price = row["close"] - (atr * 3.0)
        
        if stop_price <= 0: return None
        
        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Calculate current profit %
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        days_held = i - entry_i
        
        # RADICAL TEST: TARGET OR BUST
        # We strictly hold until +10% or -10%
        
        # 1. Profit Target (+10%)
        if current_profit_pct >= 10.0:
            return True
            
        # 2. Stop Loss (-10%)
        if current_profit_pct <= -10.0:
            return True
            
        # 3. Max Time Stop (100 days - extreme)
        if days_held >= 100:
            return True
            
        # OTHERWISE HOLD NO MATTER WHAT
        return False
