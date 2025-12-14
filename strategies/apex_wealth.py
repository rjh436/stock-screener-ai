import numpy as np
import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Wealth variant (Gen 12):
    Restores 'Sniper' precision by enforcing hard structural filters
    on top of the evolved configuration.
    
    AUDIT FIXES (2025-12-14):
    1. Replaced flawed 'EMA200 < Highest55' with strict Trend + Crash Protection.
    2. Enforced ADX > 25 to avoid 'dead money' chops, but added 20% max-drawdown gate to prevent knife-catching.
    """

    def entry(self, df: pd.DataFrame, i: int) -> dict:
        """
        Overridden entry to enforce Sniper mandates.
        """
        # 1. Get Base Signal from Genome (Gen 12 Config)
        signal = super().entry(df, i)
        if not signal:
            return None

        row = df.iloc[i]
        
        # --- PRE-COMPUTE VALUES ---
        close_px = row.get("close", 0)
        ema200 = row.get("ema200", 0)
        highest55 = row.get("highest55", 0)

        # 2. SNIPER STRUCTURE FILTER (Audit Fix)
        # OLD FLAWED LOGIC: if not (ema200 < highest55): return None
        
        # NEW LOGIC A: Structural Uptrend
        # Price must be above the long-term baseline.
        if close_px < ema200:
            return None
            
        # NEW LOGIC B: Crash Protection (The "Falling Knife" Gate)
        # Even if > EMA200, do not buy if we are > 20% off the recent high.
        # This prevents buying the "first leg down" of a major correction.
        if highest55 > 0 and close_px < (0.80 * highest55):
            return None

        # 3. REGIME FILTER (Market Health)
        # Only deploy Wealth capital when the broad market is constructive.
        # Note: Engine passes i-1 (Yesterday), so this is safe for MOO.
        spy_close = row.get("spy_close", np.nan)
        spy_sma = row.get("spy_sma200", np.nan)
        if pd.notna(spy_close) and pd.notna(spy_sma):
            if spy_close < spy_sma:
                return None

        # 4. PRECISION FILTERS (Reduce Noise)
        # RSI2 < 25: Ensure deep pullback (User Mandate: "Deepest 10%")
        if row.get("rsi2", 50) > 25:
            return None
        
        # ADX > 25: Trend Strength
        # We only want to buy dips in stocks that are actually MOVING.
        # The Crash Protection (Logic B) makes this safe.
        if row.get("adx", 0) < 25:
            return None

        return signal

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        """
        Wealth variant exit with trailing stops and hard loss floor.
        """
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        pnl_pct = ((close_px - entry_price) / entry_price) * 100

        # 1. Configuration (Source of Truth)
        time_limit = int(self.genome.get("time_stop", 60))
        use_bb_exit = bool(self.genome.get("use_bb_exit", False))

        # Resolve profit target from exit_rules
        profit_mult = 1.15
        exit_rules = self.genome.get("exit_rules", [])
        for rule in exit_rules:
            if rule.get("type") == "profit_target":
                profit_mult = float(rule.get("val", 1.20))
                break

        # Calculate current ATR-based trailing stop
        atr = row.get("atr14", close_px * 0.02)
        stop_mult = float(self.genome.get("stop_loss_atr", 3.0))
        trailing_stop = close_px - (atr * stop_mult)

        # Hard floor: Never lose more than 12% (tightened to prevent gap-down slippage)
        hard_floor = entry_price * 0.88

        # Effective stop is the HIGHEST (tightest protection)
        effective_stop = max(trailing_stop, hard_floor, stop_price)

        # 2. STOP LOSS CHECK (now uses dynamic stop)
        if row.get("low", np.inf) < effective_stop:
            return True

        # 2b. Bollinger-based profit release
        bb_upper = row.get("bb_upper", np.nan)
        if use_bb_exit and pd.notna(bb_upper):
            if row.get("high", close_px) >= bb_upper:
                return True

        # 3. PROFIT TARGET (Dynamic)
        if (not use_bb_exit) and row.get("high", 0) > entry_price * profit_mult:
            return True

        # 4. TIME STOP (Dynamic)
        # Progressive tightening in final 20% of hold period
        if days_held >= time_limit * 0.8:
            if pnl_pct < -5.0:
                return True
            if pnl_pct < 3.0:
                sma20 = row.get("sma20", close_px)
                if close_px < sma20:
                    return True

        # Final time limit
        if days_held >= time_limit:
            return True

        # 5. TREND BREAK PROTECTION
        sma50 = row.get("sma50")
        if sma50 and close_px < sma50:
            if pnl_pct > 0:
                return True  # Protect gains
            if days_held > 10:
                return True  # Exit laggards

        return False
