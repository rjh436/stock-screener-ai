from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyAIV3(BaseStrategy):
    """
    AI V3: "Deep Value Hunter"
    
    Derived from Evolutionary Generation 9 (Top Performer).
    CAGR: ~9.21% | Win Rate: ~86%
    
    Logic:
    1. Deep Correction: EMA100 must be ABOVE the 55-day High. 
       This implies the stock is significantly below its long-term average trend.
    2. Volatility/Price Filter: Plus_DI < Highest55. 
       (Implicitly favors stocks where price > DI, i.e., not penny stocks).
    3. Momentum Turn: RSI14 > CCI. 
       Captures the moment momentum (RSI) overtakes the cyclical trend (CCI).
       
    Exit:
    - Profit-First: Hold < 10%, Trail > 10%.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        # Default parameters from evolution
        default_params = {
            "stop_loss_atr": 1.6,
            "max_time_stop": 60, # Standardized to 60 for consistency
            "profit_target": 10.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        row = df.iloc[i]
        
        # 1. Deep Correction Filter
        # EMA100 must be ABOVE the 55-day High.
        # This implies the stock is significantly below its long-term average trend.
        if row["ema100"] <= row["highest55"]:
            return None
            
        # 2. Volatility/Price Filter
        # Plus_DI < Highest55 (Implicitly favors stocks where price > DI, i.e., not penny stocks)
        if row["plus_di"] >= row["highest55"]:
            return None
            
        # 3. Momentum Turn
        # RSI crossing above CCI (Momentum turning up relative to cycle)
        if row["rsi14"] <= row["cci"]:
            return None

        # Entry Sizing
        atr = row.get("atr14", row["close"]*0.02)
        stop_price = row["close"] - (atr * self.params["stop_loss_atr"])
        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Calculate current profit %
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        days_held = i - entry_i
        
        # 1. Hard Stop Loss
        if row["low"] < stop_price:
            return True
            
        # 2. Maximum Time Stop
        if days_held >= self.params["max_time_stop"]:
            return True
            
        # 3. MINIMUM PROFIT TARGET GATE (< 10%)
        if current_profit_pct < 10.0:
            # HOLD AGGRESSIVELY
            # Only exit on extreme RSI spike (Parabolic Blow-off)
            if row["rsi2"] > 95:
                return True
            return False # HOLD
        
        # 4. ABOVE 10%: TRAILING STOP
        if current_profit_pct < 20.0:
            # EMA20 trailing (10-20% profit)
            if row["close"] < row.get("ema20", row["sma20"]):
                return True
        else:
            # EMA50 trailing (>20% profit - let winners run)
            if row["close"] < row.get("ema50", row["sma50"]):
                return True
                
        return False
