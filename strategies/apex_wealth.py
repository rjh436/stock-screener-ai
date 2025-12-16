import numpy as np
import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Apex Wealth (Gen 15 - The Limit Runner):
    A Deep Value strategy that solves the "Profit Ceiling" by:
    1. Buying cheaper (Limit Orders @ 2% discount).
    2. Selling later (ATR Trailing Stops to capture trends).
    """

    def entry(self, df: pd.DataFrame, i: int) -> dict:
        signal = super().entry(df, i)
        if not signal:
            return None

        row = df.iloc[i]
        
        # --- PARAMETERS ---
        # Gen 15: We define the Limit Discount here (e.g., 0.98 = 2% discount)
        limit_ratio = float(self.genome.get("limit_ratio", 0.98))
        
        adx_threshold = float(self.genome.get("adx_threshold", 25.0))
        rsi_strong = float(self.genome.get("rsi_strong", 40.0))
        rsi_weak = float(self.genome.get("rsi_weak", 15.0))
        
        # --- FILTERS ---
        close_px = row.get("close", 0)
        ema200 = row.get("ema200", 0)
        highest55 = row.get("highest55", 0)
        adx = row.get("adx", 0)
        rsi2 = row.get("rsi2", 50)

        # 1. Safety Guardrails
        if close_px < ema200: return None
        if highest55 > 0 and close_px < (0.80 * highest55): return None

        # 2. Regime Filter
        spy_close = row.get("spy_close", np.nan)
        spy_sma = row.get("spy_sma200", np.nan)
        if pd.notna(spy_close) and pd.notna(spy_sma):
            if spy_close < spy_sma: return None

        # 3. Dual Lane Logic
        valid_setup = False
        if adx > adx_threshold:
            if rsi2 <= rsi_strong: valid_setup = True
        else:
            if rsi2 <= rsi_weak: valid_setup = True

        if valid_setup:
            # RETURN LIMIT ORDER INSTRUCTION
            # The engine will only fill if Low < (Close * limit_ratio)
            return {
                "limit_ratio": limit_ratio,
                "stop_loss_atr": float(self.genome.get("stop_loss_atr", 3.0))
            }

        return None

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        high_px = row.get("high", close_px)
        
        # --- GEN 15 EXIT LOGIC (The Runner) ---
        trail_mult = float(self.genome.get("trail_atr", 3.0))
        time_limit = int(self.genome.get("time_stop", 60))
        
        # 1. Calculate Dynamic Trailing Stop
        # Stop is calculated from the Highest High since entry
        # We simulate this by checking if today's Low hit the theoretical trail
        # Note: In a real engine, we'd track 'highest_high' statefully. 
        # Here we approximate using the current bar's ATR.
        atr = row.get("atr14", close_px * 0.02)
        
        # Base hard stop (Initial Risk)
        if row.get("low") < stop_price:
            return True

        # 2. Profit Trailing (The "Let it Run" Logic)
        # If we are profitable, we switch to a trailing stop
        if close_px > entry_price:
            # Theoretical Trail: High - (ATR * Mult)
            dynamic_stop = high_px - (atr * trail_mult)
            
            # If price drops below this dynamic stop, we exit
            if row.get("low") < dynamic_stop:
                # But ensure we don't exit below our entry if we are just starting
                # (Allow some breathing room unless we are deep in profit)
                if dynamic_stop > entry_price: 
                    return True

        # 3. Time Stop (Fail-safe)
        if days_held >= time_limit:
            return True

        return False
