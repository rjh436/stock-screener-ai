import numpy as np
import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Wealth variant (Gen 14 - Optimizer Ready): 
    Fully parameterized Dual Lane logic to allow Genetic Optimization 
    of entry thresholds and exit behaviors.
    """

    def entry(self, df: pd.DataFrame, i: int) -> dict:
        # We still call super() to respect basic hygiene (if any), 
        # but the JSON 'entry_rules' will be empty for Wealth, passing this immediately.
        signal = super().entry(df, i)
        if not signal:
            return None

        row = df.iloc[i]
        
        # --- PARAMETERS (Optimizer Tunable) ---
        # Defaults set to current "Dual Lane" values, but now mutable.
        adx_threshold = float(self.genome.get("adx_threshold", 25.0))
        rsi_strong = float(self.genome.get("rsi_strong", 40.0))
        rsi_weak = float(self.genome.get("rsi_weak", 15.0))
        
        # --- PRE-COMPUTE VALUES ---
        close_px = row.get("close", 0)
        ema200 = row.get("ema200", 0)
        highest55 = row.get("highest55", 0)
        adx = row.get("adx", 0)
        rsi2 = row.get("rsi2", 50)

        # 1. STRUCTURAL SAFETY (Hardcoded Guardrails)
        # These are non-negotiable safety floors.
        if close_px < ema200: return None
        if highest55 > 0 and close_px < (0.80 * highest55): return None

        # 2. REGIME FILTER
        spy_close = row.get("spy_close", np.nan)
        spy_sma = row.get("spy_sma200", np.nan)
        if pd.notna(spy_close) and pd.notna(spy_sma):
            if spy_close < spy_sma: return None

        # 3. DUAL LANE ADAPTIVE ENTRY
        if adx > adx_threshold:
            # Lane A: Strong Trend
            if rsi2 > rsi_strong: return None
        else:
            # Lane B: Weak Trend
            if rsi2 > rsi_weak: return None

        return signal

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        pnl_pct = ((close_px - entry_price) / entry_price) * 100
        
        # --- PARAMETERS ---
        # Optimizer can now toggle the "Scalp Exit" on/off
        use_scalp_exit = bool(self.genome.get("use_scalp_exit", True)) 
        time_limit = int(self.genome.get("time_stop", 60))
        use_bb_exit = bool(self.genome.get("use_bb_exit", False))

        # Standard Exits
        profit_mult = 1.15
        for rule in self.genome.get("exit_rules", []):
            if rule.get("type") == "profit_target":
                profit_mult = float(rule.get("val", 1.20))
                break

        # Dynamic Stop Loss
        atr = row.get("atr14", close_px * 0.02)
        stop_mult = float(self.genome.get("stop_loss_atr", 3.0))
        trailing_stop = close_px - (atr * stop_mult)
        hard_floor = entry_price * 0.88
        effective_stop = max(trailing_stop, hard_floor, stop_price)

        # 1. STOP LOSS
        if row.get("low", np.inf) < effective_stop: return True

        # 2. PROFIT TAKING
        if use_bb_exit and pd.notna(row.get("bb_upper")) and row.get("high") >= row["bb_upper"]: return True
        if (not use_bb_exit) and row.get("high") > entry_price * profit_mult: return True

        # 3. TIME STOP
        if days_held >= time_limit: return True

        # 4. CONDITIONAL SCALP EXIT (Tunable)
        # Only active if enabled by Optimizer. 
        # Exits if Green AND below SMA50 (Structure Broken).
        if use_scalp_exit:
            sma50 = row.get("sma50")
            if sma50 and close_px < sma50:
                if pnl_pct > 0: return True
                if days_held > 10: return True

        return False
