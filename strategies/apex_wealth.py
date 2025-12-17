import numpy as np
import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Apex Wealth (Gen 16.2 - Stricter Hunter):
    Uses a 3-stage exit but requires 1.5 ATR profit before offering
    Breakeven protection. This prevents 'camping' on small gains.
    """

    def entry(self, df: pd.DataFrame, i: int) -> dict:
        signal = super().entry(df, i)
        if not signal:
            return None

        row = df.iloc[i]

        limit_ratio = float(self.genome.get("limit_ratio", 0.98))
        adx_threshold = float(self.genome.get("adx_threshold", 25.0))
        rsi_strong = float(self.genome.get("rsi_strong", 40.0))
        rsi_weak = float(self.genome.get("rsi_weak", 15.0))

        close_px = row.get("close", 0)
        ema200 = row.get("ema200", 0)
        highest55 = row.get("highest55", 0)
        adx = row.get("adx", 0)
        rsi2 = row.get("rsi2", 50)

        # Safety & Regime
        if close_px < ema200:
            return None
        if highest55 > 0 and close_px < (0.80 * highest55):
            return None

        spy_close = row.get("spy_close", np.nan)
        spy_sma = row.get("spy_sma200", np.nan)
        if pd.notna(spy_close) and pd.notna(spy_sma):
            if spy_close < spy_sma:
                return None

        # Dual Lane Logic
        valid_setup = False
        if adx > adx_threshold:
            if rsi2 <= rsi_strong:
                valid_setup = True
        else:
            if rsi2 <= rsi_weak:
                valid_setup = True

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

        trail_mult = float(self.genome.get("trail_atr", 3.0))
        time_limit = int(self.genome.get("time_stop", 60))

        atr_ref_idx = max(0, entry_idx - 1)
        entry_atr = df.iloc[atr_ref_idx].get("atr14", np.nan)
        try:
            atr = float(entry_atr)
        except (TypeError, ValueError):
            atr = np.nan
        if not np.isfinite(atr) or atr <= 0:
            atr = close_px * 0.02

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
        profit_atr_units = profit_per_share / atr if atr > 0 else 0.0

        # STAGE 1: Volatility Acceptance (< 1.5 ATR Profit)
        # Increased from 1.0 to 1.5 to prevent premature breakeven.
        effective_stop = stop_price

        # STAGE 2: The Breakeven Bridge (1.5 - 2.5 ATR Profit)
        if 1.5 <= profit_atr_units < 2.5:
            breakeven_buffer = 0.1 * atr
            effective_stop = max(stop_price, entry_price + breakeven_buffer)

        # STAGE 3: The Runner ( > 2.5 ATR Profit)
        if profit_atr_units >= 2.5:
            dynamic_trail = peak_high - (atr * trail_mult)
            effective_stop = max(stop_price, entry_price + (0.1 * atr), dynamic_trail)

        if low_px < effective_stop:
            return True

        if days_held >= time_limit:
            return True

        return False
