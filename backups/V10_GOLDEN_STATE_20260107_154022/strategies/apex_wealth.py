import pandas as pd
from strategies.generic import GenericStrategy


class ApexWealthStrategy(GenericStrategy):
    """
    Apex Wealth: wrapper strategy that defers to evolved genome rules.
    """

    def entry(self, df: pd.DataFrame, i: int) -> dict:
        return super().entry(df, i)
