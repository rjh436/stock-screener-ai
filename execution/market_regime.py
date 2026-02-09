from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _normalize_spy_df(spy_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if spy_df is None or spy_df.empty:
        return pd.DataFrame(columns=["close", "volume"])
    df = spy_df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    if "close" not in df.columns:
        df["close"] = np.nan
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            return pd.DataFrame(columns=["close", "volume"])
    df = df[~df.index.isna()].sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    return df[["close", "volume"]]


def compute_regime_series(spy_df: Optional[pd.DataFrame]) -> pd.Series:
    """
    Compute daily market regime labels from SPY.

    Green  : close > SMA200 and distribution_day_count_25 < 4
    Yellow : close > SMA200 and distribution_day_count_25 >= 4
    Red    : close <= SMA200 (or SMA unavailable)
    """
    df = _normalize_spy_df(spy_df)
    if df.empty:
        return pd.Series(dtype="object")

    close = df["close"]
    volume = df["volume"]
    sma200 = close.rolling(200, min_periods=200).mean()

    prev_close = close.shift(1)
    prev_volume = volume.shift(1)
    distribution_day = (close < prev_close) & (volume > prev_volume)
    distribution_count = distribution_day.astype(np.int8).rolling(25, min_periods=1).sum()

    above_200 = close > sma200
    green = above_200 & (distribution_count < 4)
    yellow = above_200 & (distribution_count >= 4)
    red = ~above_200

    regime = pd.Series("RED", index=df.index, dtype="object")
    regime.loc[yellow] = "YELLOW"
    regime.loc[green] = "GREEN"
    regime.loc[red] = "RED"
    return regime


def analyze_market_health(spy_df: Optional[pd.DataFrame]) -> str:
    """
    Analyze latest market health state.
    Returns one of: GREEN, YELLOW, RED.
    """
    regime = compute_regime_series(spy_df)
    if regime.empty:
        return "RED"
    last = str(regime.iloc[-1]).upper()
    if last not in {"GREEN", "YELLOW", "RED"}:
        return "RED"
    return last
