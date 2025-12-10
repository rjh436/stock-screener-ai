from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Wealth variant: keeps generic entry logic from JSON rules but uses a patient, fixed exit.
    """

    def exit(self, df, i, entry_i, entry_price, stop_price) -> bool:
        row = df.iloc[i]
        days_held = i - entry_i

        # 1. Hard Stop
        if row["low"] < stop_price:
            return True

        # 2. Time Stop (60 Days)
        if days_held >= 60:
            return True

        # 3. Profit Target (+20%)
        if row["high"] > entry_price * 1.20:
            return True

        # NO TRAILING STOP.
        return False
