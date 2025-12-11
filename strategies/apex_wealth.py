from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Wealth variant: uses patient, fixed exits derived from CONFIG (no longer hardcoded).
    """

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

        # Resolve profit target from exit_rules
        profit_mult = 1.15
        exit_rules = self.genome.get("exit_rules", [])
        for rule in exit_rules:
            if rule.get("type") == "profit_target":
                profit_mult = float(rule.get("val", 1.20))
                break

        # === PHASE 3.6.1: DYNAMIC TRAILING STOP ===
        import numpy as np

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

        # 3. PROFIT TARGET (Dynamic)
        if row.get("high", 0) > entry_price * profit_mult:
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
