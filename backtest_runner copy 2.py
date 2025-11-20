import math
from datetime import datetime, date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import os
import concurrent.futures


# ============================================================
# INDICATOR COMPUTATION (PANDAS-BASED)
# ============================================================

MIN_BARS_FOR_WARMUP = 200  # similar to prior Backtrader setup


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all indicators needed for the strategies, using only pandas.
    Expects columns: ['open', 'high', 'low', 'close', 'volume'] and a
    DateTimeIndex sorted in ascending order.
    """
    df = df.sort_index().copy()

    # Simple and exponential moving averages
    for p in (20, 50, 100, 200):
        df[f"sma{p}"] = df["close"].rolling(p, min_periods=p).mean()
        df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False, min_periods=p).mean()

    # ATR family (Wilder-style TR but with simple rolling mean)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["tr"] = tr

    for p in (10, 14, 20):
        df[f"atr{p}"] = df["tr"].rolling(p, min_periods=p).mean()

    df["atr14_ma20"] = df["atr14"].rolling(20, min_periods=20).mean()

    # Bollinger Bands (20, 2)
    bb_mid = df["close"].rolling(20, min_periods=20).mean()
    bb_std = df["close"].rolling(20, min_periods=20).std(ddof=0)
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2.0 * bb_std
    df["bb_lower"] = bb_mid - 2.0 * bb_std

    # Highest / Lowest for breakout logic
    df["highest20"] = df["high"].rolling(20, min_periods=20).max()
    df["highest55"] = df["high"].rolling(55, min_periods=55).max()
    df["lowest5"] = df["low"].rolling(5, min_periods=5).min()
    df["lowest20"] = df["low"].rolling(20, min_periods=20).min()

    # Keltner channels (for Mindful Trader)
    df["kc_mid"] = df["sma20"]
    df["kc_upper"] = df["kc_mid"] + df["atr14"] * 2.25
    df["kc_lower"] = df["kc_mid"] - df["atr14"] * 2.25

    # Volume average
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=20).mean()

    # RSI family and ConnorsRSI-style composite
    def rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
        rsi_val[avg_loss == 0] = 100.0  # no losses -> fully overbought
        return rsi_val

    df["rsi2"] = rsi(df["close"], 2)
    df["rsi3"] = rsi(df["close"], 3)
    df["rsi5"] = rsi(df["close"], 5)
    df["rsi14"] = rsi(df["close"], 14)

    df["crsi"] = 0.4 * df["rsi2"] + 0.4 * df["rsi3"] + 0.2 * df["rsi14"]

    return df


# ============================================================
# STRATEGY-SPECIFIC ENTRY / EXIT HELPERS
# ============================================================

def _rhcts_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """
    RHCTS entry conditions and initial stop, based on the indicators.
    Returns a dict with {'entry_price', 'stop_price'} or None.
    """
    if i < 6:  # need lookback for -5 etc.
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    # Pre-filters
    if row["vol_ma20"] < 1_000_000:
        return None
    if not (row["close"] > row["sma50"]):
        return None
    if not (row["sma50"] > df.iloc[i - 5]["sma50"]):
        return None
    if row["rsi14"] >= 70:
        return None

    # Entry triggers
    touch_reclaim = (row["low"] <= row["ema20"]) and (row["close"] >= row["ema20"])
    reclaim_from_below = (prev["close"] < prev["ema20"]) and (row["close"] >= row["ema20"])
    if not (touch_reclaim or reclaim_from_below):
        return None

    if row["crsi"] > 20:
        return None

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        return None

    lowest5_yesterday = df.iloc[i - 1]["lowest5"]
    if pd.isna(lowest5_yesterday):
        return None

    stop_price = min(lowest5_yesterday, row["ema20"] - atr)
    if stop_price <= 0:
        return None

    # Strategy spec: entry at EMA20, not the close
    return {"entry_price": row["ema20"], "stop_price": stop_price}


def _rhcts_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    """
    RHCTS exit: stop-loss, close below EMA20, or time stop.
    """
    row = df.iloc[i]

    # Stop-loss: intraday low below stop
    if row["low"] < stop_price:
        return True

    # Close below EMA20
    if row["close"] <= row["ema20"]:
        return True

    # Time stop (bars since entry)
    if (i - entry_i) >= time_stop:
        return True

    return False


def _crsi_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    if i < 20:
        return None

    row = df.iloc[i]
    if not (row["close"] > row["sma200"]):
        return None
    if row["crsi"] > 15:
        return None

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        return None

    stop_price = row["close"] - 2 * atr
    if stop_price <= 0:
        return None

    return {"entry_price": row["close"], "stop_price": stop_price}


def _crsi_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    # Mean reversion exit: close back above SMA20
    if row["close"] > row["sma20"]:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _cbc_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    # Need at least 21 bars for yesterday + lookback
    if i < 21:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    prev_idx = i - 1

    # Volatility contraction yesterday
    if df.iloc[prev_idx]["atr14"] >= df.iloc[prev_idx]["atr14_ma20"]:
        return None

    high20_prev = df.iloc[prev_idx]["highest20"]
    if pd.isna(high20_prev):
        return None

    # Yesterday near resistance (within 2% below 20d high)
    if not (prev["close"] >= 0.98 * high20_prev and prev["close"] <= high20_prev):
        return None

    # RSI(2) reset yesterday
    if df.iloc[prev_idx]["rsi2"] > 20:
        return None

    # Today breaks out above yesterday's 20d high
    if not (row["close"] > high20_prev):
        return None

    # Volume spike vs yesterday's vol_ma20
    if not (row["volume"] >= 1.3 * df.iloc[prev_idx]["vol_ma20"]):
        return None

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        return None

    lowest20_prev = df.iloc[prev_idx]["lowest20"]
    if pd.isna(lowest20_prev):
        return None

    stop_price = max(lowest20_prev, high20_prev - atr)
    if stop_price <= 0:
        return None

    return {"entry_price": row["close"], "stop_price": stop_price}


def _cbc_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    # Exit on close below SMA20
    if row["close"] < row["sma20"]:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _raptor_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    if i < 25:
        return None

    row = df.iloc[i]

    # Regime uptrend
    if not (row["close"] > row["ema100"]):
        return None
    if not (row["ema100"] > df.iloc[i - 5]["ema100"]):
        return None

    # ATR squeeze
    if not (row["atr14"] < row["atr14_ma20"]):
        return None

    # Price at or below lower Bollinger band
    if not (row["close"] <= row["bb_lower"]):
        return None

    # Short-term RSI(2) pullback
    if row["rsi2"] > 10:
        return None

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        return None

    lowest5_prev = df.iloc[i - 1]["lowest5"]
    if pd.isna(lowest5_prev):
        return None

    stop_price = lowest5_prev
    if stop_price <= 0:
        return None

    return {"entry_price": row["close"], "stop_price": stop_price}


def _raptor_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    # Profit target: close above middle Bollinger band
    if row["close"] > row["bb_mid"]:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _hmp_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    if i < 50:
        return None

    row = df.iloc[i]

    # Strong uptrend
    if not (row["ema50"] > row["ema200"]):
        return None
    if not (row["close"] > row["ema50"]):
        return None

    # Short-term oversold filter
    if row["rsi5"] > 35:
        return None

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        return None

    stop_price = row["ema50"] - atr
    if stop_price <= 0:
        return None

    return {"entry_price": row["close"], "stop_price": stop_price}


def _hmp_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    # Trail on EMA50
    if row["close"] < row["ema50"]:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _cgm10_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    if i < 60:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]

    atr = row["atr14"]
    if atr <= 1e-4 or pd.isna(atr):
        atr = row["close"] * 0.01  # small fallback

    # Strategy 1: Breakaway Expansion (BX)
    bx_cond_1 = row["high"] > df.iloc[i - 1]["highest55"]
    bx_cond_2 = row["atr14"] > row["atr14_ma20"]
    bx_cond_3 = row["volume"] >= 2.0 * df.iloc[i - 1]["vol_ma20"]

    is_bx_setup = bx_cond_1 and bx_cond_2 and bx_cond_3

    # Strategy 2: Power Trend Pullback (PTP)
    ptp_cond_1 = (row["ema20"] > row["sma50"]) and (row["sma50"] > df.iloc[i - 5]["sma50"])
    ptp_cond_2 = prev["low"] <= prev["ema20"]
    ptp_cond_3 = row["volume"] >= 1.5 * df.iloc[i - 1]["vol_ma20"]
    ptp_cond_4 = row["rsi2"] <= 5

    is_ptp_setup = ptp_cond_1 and ptp_cond_2 and ptp_cond_3 and ptp_cond_4

    if not (is_bx_setup or is_ptp_setup):
        return None

    stop_price = row["close"] - 2.5 * atr
    if stop_price <= 0:
        return None

    return {"entry_price": row["close"], "stop_price": stop_price}


def _cgm10_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    # Simple exit: close below EMA20
    if row["close"] < row["ema20"]:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _r2x_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    if i < 200:
        return None

    row = df.iloc[i]

    # Regime
    if not (row["close"] > row["sma200"]):
        return None

    atr10 = row["atr10"]
    close0 = row["close"]
    if atr10 <= 1e-4 or close0 <= 0 or pd.isna(atr10):
        return None

    atr_pct = (atr10 / max(close0, 1e-6)) * 100.0
    threshold = 5.0 if atr_pct > 2.5 else 10.0
    if row["rsi2"] >= threshold:
        return None

    stop_price = close0 - 2 * atr10
    if stop_price <= 0:
        return None

    return {"entry_price": close0, "stop_price": stop_price}


def _r2x_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float) -> bool:
    row = df.iloc[i]

    if row["low"] < stop_price:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


def _mindful_entry(df: pd.DataFrame, i: int) -> Optional[Dict]:
    """
    Mindful Trader entry approximation:
    - Last 10 closes above SMA20
    - At least one Keltner upper pierce in last 10 bars
    - Entry price = current SMA20 (approximation of the limit at SMA20).
    """
    if i < 10:
        return None

    window = df.iloc[i - 9 : i + 1]  # last 10 bars including today
    row = df.iloc[i]

    # 10 consecutive closes above SMA20
    if (window["close"] < window["sma20"]).any():
        return None

    # At least one Keltner upper pierce in last 10 days
    if not (window["high"] > window["kc_upper"]).any():
        return None

    atr14 = row["atr14"]
    if atr14 <= 1e-4 or pd.isna(atr14):
        return None

    sma20 = row["sma20"]
    if pd.isna(sma20):
        return None

    # Use current SMA20 as the limit entry price
    entry_price = float(sma20)
    stop_price = entry_price - 2 * atr14
    if entry_price <= 0 or stop_price <= 0:
        return None

    return {"entry_price": entry_price, "stop_price": stop_price}


def _mindful_exit(df: pd.DataFrame, i: int, entry_i: int, time_stop: int, stop_price: float, entry_price: float) -> bool:
    row = df.iloc[i]

    # Profit target: 2R
    target_price = entry_price + 2 * (entry_price - stop_price)

    if row["high"] >= target_price:
        return True

    if row["low"] <= stop_price:
        return True

    if (i - entry_i) >= time_stop:
        return True

    return False


# ============================================================
# GENERIC PANDAS BACKTEST ENGINE
# ============================================================

STRATEGY_TIME_STOPS = {
    "RHCTS": 10,
    "CRSI": 10,
    "CBC": 15,
    "RAPTOR": 20,
    "HMP": 10,
    "CGM10": 25,
    "R2X": 10,
    "MINDFUL": 9,
}


def _backtest_strategy(
    strategy_name: str,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
) -> Dict:
    """
    Run a long-only daily backtest for a single strategy over the given
    symbol universe, using a $100,000 starting portfolio and allocating
    ~10% of equity per new position (max 10 concurrent positions).
    """

    # Filter universe
    symbols = list(data_dict.keys())
    if symbol_universe:
        symbols = [s for s in symbols if s in symbol_universe]

    if not symbols:
        return {
            "strategy": strategy_name,
            "final_value": start_cash,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "total_trades": 0,
            "win_total": 0,
            "loss_total": 0,
            "hit_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "cagr": 0.0,
            "risk_adjusted_return": 0.0,
            "Score": 0.0,
            "sharpe": 0.0,
        }

    # Prepare data with indicators
    enriched: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = data_dict[sym]
        if df is None or df.empty:
            continue

        df_local = df.copy()
        if not isinstance(df_local.index, pd.DatetimeIndex):
            # assume there is a 'date' column
            if "date" in df_local.columns:
                df_local = df_local.set_index(pd.to_datetime(df_local["date"]))
            else:
                raise ValueError(f"Data for {sym} must have a DatetimeIndex or 'date' column.")

        # Normalize timezone: convert any tz-aware index to naive so comparisons
        # against naive datetime (e.g., start_date) are always valid
        if isinstance(df_local.index, pd.DatetimeIndex) and df_local.index.tz is not None:
            df_local.index = df_local.index.tz_convert(None)

        df_local = _compute_indicators(df_local)

        # Enforce warmup and start_date
        if start_date:
            from_dt = datetime.combine(start_date, datetime.min.time())
            df_local = df_local[df_local.index >= from_dt]

        if len(df_local) < MIN_BARS_FOR_WARMUP:
            continue

        enriched[sym] = df_local

    if not enriched:
        return {
            "strategy": strategy_name,
            "final_value": start_cash,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "total_trades": 0,
            "win_total": 0,
            "loss_total": 0,
            "hit_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "cagr": 0.0,
            "risk_adjusted_return": 0.0,
            "Score": 0.0,
            "sharpe": 0.0,
        }

    # Build global date index (union of all symbol dates)
    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))

    # Portfolio state
    cash = start_cash
    positions: Dict[str, Dict] = {}  # symbol -> {shares, entry_i, entry_price, stop_price}
    equity_history: List[float] = []
    date_history: List[pd.Timestamp] = []
    trade_pnls: List[float] = []

    max_positions = 10
    pos_fraction = 0.10  # 10% of equity per position
    time_stop = STRATEGY_TIME_STOPS.get(strategy_name, 10)

    # Helper to get row index for a given date (or None)
    def get_row_index(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[int]:
        try:
            return df.index.get_loc(dt)
        except KeyError:
            return None

    # Helper to get the close on or before a given date
    def get_price_on_or_before(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
        idx = df.index.searchsorted(dt, side="right") - 1
        if idx < 0:
            return None
        return float(df.iloc[idx]["close"])

    for current_dt in all_dates:
        # 1) Process exits first
        symbols_with_data = [s for s, df in enriched.items() if current_dt in df.index]

        # Process exits
        for sym in list(positions.keys()):
            if sym not in symbols_with_data:
                continue

            df_sym = enriched[sym]
            i = get_row_index(df_sym, current_dt)
            if i is None:
                continue
            row = df_sym.iloc[i]
            pos = positions[sym]
            entry_i = pos["entry_i"]
            stop_price = pos["stop_price"]
            entry_price = pos["entry_price"]
            shares = pos["shares"]

            # Strategy-specific exit logic
            if strategy_name == "RHCTS":
                exit_now = _rhcts_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "CRSI":
                exit_now = _crsi_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "CBC":
                exit_now = _cbc_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "RAPTOR":
                exit_now = _raptor_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "HMP":
                exit_now = _hmp_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "CGM10":
                exit_now = _cgm10_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "R2X":
                exit_now = _r2x_exit(df_sym, i, entry_i, time_stop, stop_price)
            elif strategy_name == "MINDFUL":
                exit_now = _mindful_exit(df_sym, i, entry_i, time_stop, stop_price, entry_price)
            else:
                exit_now = False

            if exit_now:
                exit_price = row["close"]
                pnl = (exit_price - entry_price) * shares
                cash += shares * exit_price
                trade_pnls.append(pnl)
                del positions[sym]

        # 2) Recalculate equity after exits
        equity = cash
        for sym, pos in positions.items():
            df_sym = enriched.get(sym)
            if df_sym is None:
                continue
            price = get_price_on_or_before(df_sym, current_dt)
            if price is None:
                continue
            equity += pos["shares"] * price

        date_history.append(current_dt)
        equity_history.append(equity)

        # 3) Process entries
        # Skip new entries if no cash or already fully invested
        open_positions = len(positions)
        if open_positions >= max_positions or equity <= 0:
            continue

        for sym in symbols_with_data:
            if sym in positions:
                continue  # already holding

            df_sym = enriched[sym]
            i = get_row_index(df_sym, current_dt)
            if i is None:
                continue

            # We use a realistic model: signal is evaluated on the *prior* bar,
            # and any entry is executed on the NEXT day's open.
            signal_i = i - 1
            if signal_i < MIN_BARS_FOR_WARMUP:
                continue

            # Strategy-specific entry logic evaluated on the prior bar
            if strategy_name == "RHCTS":
                entry_info = _rhcts_entry(df_sym, signal_i)
            elif strategy_name == "CRSI":
                entry_info = _crsi_entry(df_sym, signal_i)
            elif strategy_name == "CBC":
                entry_info = _cbc_entry(df_sym, signal_i)
            elif strategy_name == "RAPTOR":
                entry_info = _raptor_entry(df_sym, signal_i)
            elif strategy_name == "HMP":
                entry_info = _hmp_entry(df_sym, signal_i)
            elif strategy_name == "CGM10":
                entry_info = _cgm10_entry(df_sym, signal_i)
            elif strategy_name == "R2X":
                entry_info = _r2x_entry(df_sym, signal_i)
            elif strategy_name == "MINDFUL":
                entry_info = _mindful_entry(df_sym, signal_i)
            else:
                entry_info = None

            if not entry_info:
                continue

            stop_price = float(entry_info["stop_price"])

            # Execute the trade on today's open
            row_today = df_sym.iloc[i]
            entry_price = float(row_today["open"])
            if entry_price <= 0 or math.isnan(entry_price):
                continue

            # Position sizing: 10% of current equity
            position_value = equity * pos_fraction
            if position_value <= 0:
                continue

            shares = math.floor(position_value / entry_price)
            if shares <= 0:
                continue

            cost = shares * entry_price
            if cost > cash:
                continue  # not enough cash

            # Open position
            cash -= cost
            positions[sym] = {
                "shares": shares,
                "entry_i": i,
                "entry_price": entry_price,
                "stop_price": stop_price,
            }

            open_positions += 1
            if open_positions >= max_positions:
                break  # no more entries this day

    # Final equity
    final_equity = cash
    if equity_history:
        # last known prices (on or before the last backtest date)
        last_dt = date_history[-1]
        for sym, pos in positions.items():
            df_sym = enriched.get(sym)
            if df_sym is None:
                continue
            price = get_price_on_or_before(df_sym, last_dt)
            if price is None:
                continue
            final_equity += pos["shares"] * price

    # Compute stats
    pnl = final_equity - start_cash
    pnl_pct = (pnl / start_cash) * 100.0 if start_cash != 0 else 0.0

    total_trades = len(trade_pnls)
    win_total = sum(1 for x in trade_pnls if x > 0)
    loss_total = sum(1 for x in trade_pnls if x < 0)
    hit_rate = (win_total / total_trades) * 100.0 if total_trades > 0 else 0.0

    # Max drawdown and Sharpe from equity curve
    if equity_history:
        eq_series = pd.Series(equity_history, index=pd.to_datetime(date_history))
        running_max = eq_series.cummax()
        drawdowns = (eq_series - running_max) / running_max
        max_drawdown_pct = drawdowns.min() * 100.0

        # Daily returns
        rets = eq_series.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * math.sqrt(252)
        else:
            sharpe = 0.0

        # CAGR
        num_days = (eq_series.index[-1] - eq_series.index[0]).days
        if num_days > 0 and start_cash > 0:
            years = num_days / 365.25
            cagr = (final_equity / start_cash) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0
        else:
            cagr = 0.0
    else:
        max_drawdown_pct = 0.0
        sharpe = 0.0
        cagr = 0.0

    # Score = CAGR * Sharpe * Hit_Rate (scaled)
    cagr_val = cagr
    sharpe_val = sharpe if math.isfinite(sharpe) else 0.0
    hit_rate_val = hit_rate / 100.0
    score = cagr_val * sharpe_val * hit_rate_val

    return {
        "strategy": strategy_name,
        "final_value": final_equity,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "total_trades": total_trades,
        "win_total": win_total,
        "loss_total": loss_total,
        "hit_rate": hit_rate,
        "max_drawdown_pct": max_drawdown_pct,
        "cagr": cagr_val,
        "risk_adjusted_return": None,
        "Score": score,
        "sharpe": sharpe_val,
    }



# ============================================================
# Parallel-safe helper for per-strategy backtest with error handling
# ============================================================
def _run_backtest_for_strategy(
    name: str,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]],
    start_cash: float,
    start_date: Optional[date],
) -> Dict:
    """
    Helper to run a single strategy backtest with robust error handling.
    This is separated so it can be used from both serial and parallel
    execution paths in run_compare.
    """
    try:
        return _backtest_strategy(
            strategy_name=name,
            data_dict=data_dict,
            symbol_universe=symbol_universe,
            start_cash=start_cash,
            start_date=start_date,
        )
    except Exception as e:
        # Fail-safe: never crash the whole app
        print(f"!!! Strategy {name} failed in pandas backtest: {e}")
        return {
            "strategy": name,
            "final_value": start_cash,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "total_trades": 0,
            "win_total": 0,
            "loss_total": 0,
            "hit_rate": 0.0,
            "max_drawdown_pct": 0.0,
            "cagr": 0.0,
            "risk_adjusted_return": 0.0,
            "Score": 0.0,
            "sharpe": 0.0,
            "error": str(e),
        }


def run_compare(
    strategy_names,
    data_dict,
    symbol_universe=None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    **kwargs,
):
    """
    Public API used by app.py.

    For each strategy in strategy_names, runs a pure-pandas portfolio
    backtest over the provided data_dict and returns a DataFrame of
    summary statistics, similar to the previous Backtrader-based
    implementation but without any of the ZeroDivisionError issues.

    Performance tuning for M3 Max:
    - By default, this will run strategies in parallel using a
      ThreadPoolExecutor, which lets NumPy/Pandas' C-level code take
      advantage of multiple performance cores.
    - You can override this by passing use_parallel=False or by
      specifying max_workers explicitly via **kwargs.
    """

    results_rows = []

    # Parallel execution settings
    use_parallel = kwargs.get("use_parallel", True)
    max_workers = kwargs.get("max_workers")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        # Leave at least one core free and cap at 8 to avoid oversubscription
        max_workers = min(max(1, cpu_count - 1), 8)

    strategy_names = list(strategy_names) if strategy_names is not None else []

    if use_parallel and len(strategy_names) > 1:
        # Run each strategy's backtest in parallel threads.
        # This is safe because each strategy is independent and the heavy
        # lifting is done in NumPy/Pandas (which releases the GIL).
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_backtest_for_strategy,
                    name,
                    data_dict,
                    symbol_universe,
                    start_cash,
                    start_date,
                )
                for name in strategy_names
            ]
            for future in concurrent.futures.as_completed(futures):
                results_rows.append(future.result())
    else:
        # Serial fallback (single-threaded, deterministic ordering)
        for name in strategy_names:
            result = _run_backtest_for_strategy(
                name=name,
                data_dict=data_dict,
                symbol_universe=symbol_universe,
                start_cash=start_cash,
                start_date=start_date,
            )
            results_rows.append(result)

    if not results_rows:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_rows)

    # Z-score the Score column (if available and non-constant)
    if "Score" in results_df.columns:
        score_std = results_df["Score"].std()
        if score_std and not math.isnan(score_std) and score_std != 0:
            results_df["Score_z"] = (results_df["Score"] - results_df["Score"].mean()) / score_std
        else:
            results_df["Score_z"] = 0.0
    else:
        results_df["Score_z"] = 0.0

    results_df.sort_values(by="Score", ascending=False, inplace=True)

    return results_df