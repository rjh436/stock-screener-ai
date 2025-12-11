from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Wealth variant: uses patient, fixed exits derived from CONFIG (no longer hardcoded).
    """

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        row = df.iloc[i]
        days_held = i - entry_i

        # 1. Configuration (Source of Truth)
        # Default to old hardcoded values (60, 1.20) to preserve legacy behavior
        time_limit = int(self.genome.get("time_stop", 60))
        
        # Resolve profit target from exit_rules or direct param
        profit_mult = 1.20
        exit_rules = self.genome.get("exit_rules", [])
        for rule in exit_rules:
            if rule.get("type") == "profit_target":
                profit_mult = float(rule.get("val", 1.20))
                break

        # 2. Hard Stop (Volatility Adjusted from Engine)
        if row["low"] < stop_price:
            return True

        # 3. Time Stop (Dynamic)
        if days_held >= time_limit:
            # Strictly obey the time limit to free up capital.
            return True

        # 4. Profit Target (Dynamic)
        if row["high"] > entry_price * profit_mult:
            return True

        # NO TRAILING STOP for Wealth strategies
        return False
