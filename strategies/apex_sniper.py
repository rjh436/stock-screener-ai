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
            "time_stop": 30,
            "breakeven_pct": 0.03
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
            
        # Entry Signal (anchor on signal bar close; actual fill happens next bar at open)
        anchor_price = row["close"]
        atr = row.get("atr14", anchor_price * 0.02)
        return {
            "entry_price": anchor_price,
            "stop_price": anchor_price - (atr * self.params["stop_loss_atr"]),
            "take_profit": None,
            "time_stop": self.params["time_stop"]
        }
        
    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> tuple:
        row = df.iloc[i]

        # 1. Configurable Parameters (The "Winning DNA")
        # These now override hardcoded defaults
        profit_target_mult = float(self.params.get("profit_target", 1.27))
        # Support generic "exit_rules" parsing if needed, but prioritize direct param
        if not profit_target_mult or profit_target_mult < 1.0:
            # Fallback to exit_rules parsing if explicit param missing
            for rule in self.params.get("exit_rules", []) or []:
                if rule.get("type") == "profit_target":
                    profit_target_mult = float(rule.get("val", 1.27))
                    break

        time_stop = int(self.params.get("time_stop", 32))
        breakeven_pct = float(self.params.get("breakeven_pct", 0.083))

        # 2. Profit Target Execution (CRITICAL FIX)
        if profit_target_mult > 1.0 and entry_price > 0:
            target_px = entry_price * profit_target_mult
            # If the high hit the target, we claim the target price
            if row.get("high", row["close"]) >= target_px:
                return True, stop_price, target_px

        # 3. Breakeven / Stop Logic
        current_profit_pct = ((row["close"] - entry_price) / entry_price) if entry_price else 0.0
        effective_stop = stop_price

        if breakeven_pct > 0 and current_profit_pct > breakeven_pct:
            effective_stop = max(effective_stop, entry_price)

        if row["low"] < effective_stop:
            return True, effective_stop, None

        # 4. Time Stop
        if (i - entry_i) >= time_stop:
            return True, effective_stop, None

        # 5. Panic Exit (Only extreme conditions)
        if row.get("rsi2", 50) > 98:
            return True, effective_stop, None

        # Remove legacy EMA choke holds
        return False, effective_stop, None
