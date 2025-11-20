from strategies.base import BaseStrategy
import pandas as pd

class StrategyApexSniper(BaseStrategy):
    """
    Apex Sniper: A refined version of the 'Strategy_Apex_Gen9' evolved strategy.
    
    Logic:
    - Trend: EMA200 < Highest55 (Price is consolidating/pulling back in a long-term uptrend)
    - Pullback: RSI(2) < 15 (Strict pullback requirement to reduce trade frequency)
    - Exit: Stochastic K > Stochastic D (Momentum shift)
    
    Target: High CAGR with reduced trade frequency (5-10 trades/week).
    """
    def __init__(self, params=None):
        self.params = {
            "rsi_entry": 5,
            "stop_loss_atr": 4.0,
            "time_stop": 30
        }
        if params:
            self.params.update(params)
            
    def entry(self, df: pd.DataFrame, i: int) -> dict:
        # Need enough data
        if i < 200:
            return None
            
        row = df.iloc[i]
        
        # 1. Trend / Structure Filter (From Apex Gen9)
        # EMA200 < Highest55 implies we are not at an all-time high blowout, 
        # but the trend is likely up if EMA200 is below the recent high.
        # Actually, let's verify the logic:
        # If EMA200 is 100 and Highest55 is 120, this is TRUE.
        # If EMA200 is 100 and Highest55 is 90 (Downtrend), this is FALSE.
        # So this acts as a Trend Filter!
        if not (row["ema200"] < row["highest55"]):
            return None
            
        # 2. Sniper Filter (Reduce Frequency)
        # Wait for RSI2 to get oversold
        if row["rsi2"] > self.params["rsi_entry"]:
            return None
            
        # 3. Global Market Filters (AI Enhanced)
        # VIX Check: Avoid buying during extreme panic (VIX > 40)
        if row.get("vix", 20.0) > 40.0:
            return None
            
        # Relative Strength Check: Only buy if stock is outperforming SPY recently
        # This is CRITICAL for finding winners.
        if row.get("rs_trend", 1) == 0:
            return None
            
        # Entry Signal
        return {
            "entry_price": row["open"], # Enter at Open of next bar (simulated by current bar open if using i)
            # Note: Engine passes i-1 as signal_i, so we check yesterday's close/indicators
            # and enter at today's Open.
            "stop_price": row["open"] - (row["atr14"] * self.params["stop_loss_atr"]),
            "take_profit": None,
            "time_stop": self.params["time_stop"]
        }
        
    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Calculate current profit %
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        
        # 1. Hard Stop Loss
        if row["low"] < stop_price:
            return True
        
        # 2. Maximum Time Stop (60 days)
        if (i - entry_i) >= 60:
            return True
        
        # 3. MINIMUM PROFIT TARGET GATE (10%)
        if current_profit_pct < 10.0:
            # FIXED: Use EMA200 not EMA50 (we enter on pullbacks!)
            if row["close"] < row.get("ema200", row["sma200"]) and row.get("adx", 30) < 20:
                return True
            if row["rsi2"] > 95:  # Parabolic spike
                return True
            return False  # Otherwise HOLD
        
        # 4. TRAILING STOP ABOVE 10%
        if current_profit_pct < 20.0:
            # 10-20%: EMA20 trailing
            if row["close"] < row.get("ema20", row["sma20"]):
                return True
        else:
            # >20%: EMA50 trailing (wider)
            if row["close"] < row.get("ema50", row["sma50"]):
                return True
        
        # 5. Extreme spike exit
        if row["rsi2"] > 98:
            return True
        
        return False
