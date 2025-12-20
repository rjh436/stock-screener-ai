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
