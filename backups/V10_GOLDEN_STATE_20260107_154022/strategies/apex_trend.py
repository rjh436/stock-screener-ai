import numpy as np
import pandas as pd

from .base import BaseStrategy


class ApexTrendStrategy(BaseStrategy):
    def __init__(self):
        params = {
            "name": "Apex Trend Elite",
            "time_stop": 45,
            "stop_loss_atr": 4.5
        }
        super().__init__(params)
        self._name = params["name"]

    @property
    def name(self) -> str:
        return self._name

    def entry(self, df: pd.DataFrame, i: int):
        if i < 200:
            return None

        row = df.iloc[i]

        # Mandatory gates
        sma200 = row.get("sma200")
        close_px = row.get("close")
        rsi14 = row.get("rsi14")
        bb_width = row.get("bb_width", 0)
        if sma200 is None or pd.isna(sma200) or close_px is None or pd.isna(close_px):
            return None
        if close_px <= sma200:
            return None  # Trend floor
        if rsi14 is None or pd.isna(rsi14) or rsi14 >= 40:
            return None  # Momentum dip requirement
        if bb_width <= 0.17:
            return None  # Require volatility

        # Quality scoring
        quality_score = 100.0
        rsi2 = row.get("rsi2", 50)
        volume = row.get("volume", 0)
        vol_ma20 = row.get("vol_ma20", 0)

        quality_score += (100 - rsi2) * 3.0  # RSI depth bonus
        if vol_ma20 and volume > 2 * vol_ma20:
            quality_score += 40.0
        elif vol_ma20 and volume > 1.5 * vol_ma20:
            quality_score += 20.0
        if close_px > 1.1 * sma200:
            quality_score += 40.0  # Trend strength bonus

        if quality_score < 180.0:
            return None

        atr = row.get("atr14", 0)
        atr_pct = (atr / close_px) * 100 if close_px else 0
        if atr_pct > 5.0:
            stop_mult = 5.5
        elif atr_pct > 3.0:
            stop_mult = 5.0
        else:
            stop_mult = 4.5

        stop_price = close_px - (atr * stop_mult)
        return {"entry_price": close_px, "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        pnl_pct = ((close_px - entry_price) / entry_price) * 100

        # 1. Hard Stop (Volatility Adjusted) - The "Disaster" Line
        if row.get("low", np.inf) < stop_price:
            return True

        # 2. Stale Trade Cleanup (Time Based)
        time_limit = int(self.params.get("time_stop", 45))
        if days_held >= time_limit:
            sma50 = row.get("sma50")
            if pnl_pct < 0:
                return True  # Cut losers at time limit
            if pnl_pct < 8.0:
                return True  # Cut weak winners
            if sma50 and close_px < sma50:
                return True  # Cut broken trends

        # 3. Trailing Stop (Trend Based)
        sma50 = row.get("sma50")
        if sma50 and close_px < sma50:
            if pnl_pct > 0:
                return True  # Protect gains
            if days_held > 10:
                return True  # Trend broken on a laggard

        # 4. Profit Targets (Optional)
        # Placeholder: no explicit targets for Apex Trend Elite
        return False
