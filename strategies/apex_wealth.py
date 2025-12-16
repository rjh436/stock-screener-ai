import numpy as np
import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Apex Wealth (Gen 16.1 - Staircase Hunter):
    Uses a 3-stage exit to safely transition from 'Volatility Acceptance'
    to 'Breakeven Protection' to 'Aggressive Trailing'.
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
            return {
                "limit_ratio": limit_ratio,
                "stop_loss_atr": float(self.genome.get("stop_loss_atr", 3.0))
            }

        return None

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        row = df.iloc[i]

        entry_idx = int(entry_i) if entry_i is not None else 0
        entry_idx = max(0, min(entry_idx, i))
        days_held = i - entry_idx

        close_px = float(row.get("close", entry_price) or entry_price)
        low_px = float(row.get("low", close_px) or close_px)

        # --- GEN 16.1 STAIRCASE EXIT ---
        trail_mult = float(self.genome.get("trail_atr", 3.0))
        time_limit = int(self.genome.get("time_stop", 60))

        # Entry volatility reference (ATR14 at entry decision time; avoids lookahead)
        atr_ref_idx = max(0, entry_idx - 1)
        entry_atr = df.iloc[atr_ref_idx].get("atr14", np.nan)
        try:
            entry_atr = float(entry_atr)
        except (TypeError, ValueError):
            entry_atr = np.nan
        if not np.isfinite(entry_atr) or entry_atr <= 0:
            entry_atr = float(entry_price) * 0.02

        # Use peak high since entry for monotonic stage activation and true trailing behavior
        peak_high = float(row.get("high", close_px) or close_px)
        if "high" in df.columns and entry_idx <= i:
            peak_high_val = df.iloc[entry_idx : i + 1]["high"].max()
            try:
                peak_high_val = float(peak_high_val)
            except (TypeError, ValueError):
                peak_high_val = np.nan
            if np.isfinite(peak_high_val):
                peak_high = peak_high_val

        profit_per_share = max(0.0, peak_high - float(entry_price))
        profit_atr_units = profit_per_share / entry_atr if entry_atr > 0 else 0.0

        hard_stop = float(stop_price) if stop_price is not None else float(entry_price) * 0.90
        effective_stop = hard_stop

        # STAGE 1: Early (< 1.0 ATR profit) -> Hard Stop only
        # STAGE 2: Bridge (1.0 - 2.0 ATR profit) -> Move stop to breakeven + buffer
        breakeven_buffer_stop = float(entry_price) + (0.1 * entry_atr)
        if 1.0 <= profit_atr_units < 2.0:
            effective_stop = max(effective_stop, breakeven_buffer_stop)

        # STAGE 3: Runner (>= 2.0 ATR profit) -> Activate trailing stop: peak_high - 3.0 * ATR
        if profit_atr_units >= 2.0:
            trailing_stop = peak_high - (entry_atr * trail_mult)
            effective_stop = max(effective_stop, breakeven_buffer_stop, trailing_stop)

        if low_px < effective_stop:
            return True

        if days_held >= time_limit:
            return True

        return False
