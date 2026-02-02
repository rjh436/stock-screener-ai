from typing import Dict, Optional
import pandas as pd
import numpy as np
from .generic import GenericStrategy

class SuperperformanceStrategy(GenericStrategy):
    """
    Project Apex: Superperformance Strategy
    Implements strict Minervini Stage 2 filters and Qullamaggie/VCP entry triggers.
    """
    def __init__(self, genome: dict):
        super().__init__(genome)
        self._name = genome.get("name", "Superperformance")

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = 200 # Need 200 MA
        if i < warmup:
            return None
            
        row = df.iloc[i]
        
        # --- 1. STRICT STAGE 2 FILTER (The Gatekeeper) ---
        close = row.get("close")
        sma50 = row.get("sma50")
        sma150 = row.get("sma150")
        sma200 = row.get("sma200")
        rs_rating = row.get("rs_rating", 0)
        
        # Filter 1: Moving Average Alignment (Price > 50 > 200)
        # REMOVED SMA150 check to allow early Stage 2 entries
        if not (close > sma50 > sma200):
            return None
            
        # Filter 2: 200-Day Trend (Slope Positive)
        # Look back 20 days to check slope
        try:
            prev_sma200 = df.iloc[i-20]["sma200"]
            if sma200 <= prev_sma200:
                return None
        except:
            return None
            
        # Filter 3: 52-Week High/Low
        high_52 = row.get("high_52", close) # Assuming engine calculates this, or use fallback
        low_52 = row.get("low_52", close)
        
        if close < (0.75 * high_52): # Must be within 25% of highs
            return None
        if close < (1.25 * low_52): # Must be 25% above lows
            return None
            
        # Filter 4: Relative Strength
        if rs_rating < 65: # Relaxed to 65 to widen the funnel
            return None
            
        # --- 2. ENTRY TRIGGERS (The Spark) ---
        # Trigger A: VCP Breakout (Tightness + Volume Dry Up)
        natr = row.get("natr", 100)
        vol = row.get("volume", 0)
        vol_sma50 = row.get("vol_sma50", vol)
        
        # Trigger A: VCP Breakout (Tightness + Volume Dry Up BEFORE today + Breakout TODAY)
        # Logic: 
        # 1. Yesterday (or recent days) had low volume (Dry Up).
        # 2. Today price is breaking out (Close > 20-Day High).
        # 3. Volatility (NATR) is low (Tightness).
        
        prev_vol = df.iloc[i-1]["volume"]
        high_20_prev = row.get("high_20_prev", 0)
        
        is_vcp = False
        if natr < 3.0: # Tightness
            if prev_vol < (vol_sma50 * 1.0): # Dry Up (Below Average is enough)
                if close > high_20_prev: # Breakout TODAY
                   is_vcp = True
               
        # Trigger B: Episodic Pivot (Power Breakout)
        is_ep = False
        open_price = row.get("open")
        prev_close = df.iloc[i-1]["close"]
        gap_pct = (open_price - prev_close) / prev_close
        
        if gap_pct > 0.02: # Relaxed Gap (2%)
            if vol > (vol_sma50 * 1.5): # Relaxed Vol (1.5x)
                is_ep = True
        
        # Genome Override: Allow Genetic Algorithm to tune specific detailed triggers if needed,
        # but for now, we fire if EITHER trigger is met.
        if is_vcp or is_ep:
            return {
                "limit_ratio": 0.0, # Market Order (or Limit at Close)
                "stop_loss_atr": self.genome.get("stop_loss_atr", 2.0),
                "stop_loss_type": "atr"
            }
            
        return None
