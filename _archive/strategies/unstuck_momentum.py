from typing import Dict, Optional
import pandas as pd
from .base import BaseStrategy

class StrategyUnstuckMomentum(BaseStrategy):
    """
    Regime: Bullish (Close > SMA200)
    Setup: RSI Pullback
    Trigger: Breakout or Oversold
    """
    def __init__(self, params: Dict = None):
        self.params = {
            "stop_atr": 2.0,
            "target_atr": 4.0, 
            "time_stop": 15
        }
        if params: self.params.update(params)
        super().__init__(self.params)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        row = df.iloc[i]

        if not (row['close'] > row['sma200']): return None
        if not (row['rsi3'] < 30): return None
            
        atr = row.get('atr14', row['close']*0.02)
        return {
            "entry_price": row['close'], 
            "stop_price": row['close'] - (atr * self.params['stop_atr'])
        }

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        if row['low'] < stop_price: return True
        
        atr = row.get('atr14', entry_price*0.02)
        target = entry_price + (atr * self.params['target_atr'])
        
        if row['high'] > target: return True
        if (i - entry_i) > self.params['time_stop']: return True
        return False