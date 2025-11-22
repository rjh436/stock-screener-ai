import math
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands
from datetime import datetime, date
from typing import Dict, List, Optional, Type
import numpy as np
import pandas as pd
import concurrent.futures
from strategies.base import BaseStrategy

# ============================================================
# INDICATOR COMPUTATION (PANDAS-BASED)
# ============================================================

MIN_BARS_FOR_WARMUP = 200

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
    df["kc_upper"] = df["kc_mid"] + df["atr14"]    # R2-X
    df["atr10"] = df["tr"].rolling(10, min_periods=10).mean()
    df["atr_pct10"] = (df["atr10"] / (df["close"] + 1e-9)) * 100.0
    df["atr_pct20"] = df["atr_pct10"].rolling(20, min_periods=20).mean()

    # --- Extended Indicators for Evolution ---
    
    # MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    
    # ADX
    adx = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()
    
    # Stochastic
    stoch = StochasticOscillator(df["high"], df["low"], df["close"])
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    
    # CCI
    cci = CCIIndicator(df["high"], df["low"], df["close"])
    df["cci"] = cci.cci()
    
    # ROC / Momentum
    roc = ROCIndicator(df["close"], window=12)
    df["roc"] = roc.roc()
    df["mom"] = df["close"].diff(10)
    
    # Extra MAs
    df["sma5"] = SMAIndicator(df["close"], 5).sma_indicator()
    df["sma10"] = SMAIndicator(df["close"], 10).sma_indicator()
    df["sma100"] = SMAIndicator(df["close"], 100).sma_indicator()
    df["ema10"] = EMAIndicator(df["close"], 10).ema_indicator()
    df["ema100"] = EMAIndicator(df["close"], 100).ema_indicator()
    df["avgvol50"] = df["volume"].rolling(50).mean()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

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
# BACKTEST ENGINE
# ============================================================

def run_backtest(
    strategy: BaseStrategy,
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: Optional[List[str]] = None,
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict:
    """
    Run a long-only daily backtest for a single strategy object.
    """

    # Filter universe
    symbols = list(data_dict.keys())
    if symbol_universe:
        symbols = [s for s in symbols if s in symbol_universe]

    if not symbols:
        return _empty_result(strategy.name, start_cash, strategy.params)

    # Prepare data with indicators
    enriched: Dict[str, pd.DataFrame] = {}
    
    # Pre-process Global Data (SPY) for RS calculation
    spy_df = None
    if global_data and "SPY" in global_data and global_data["SPY"] is not None:
        spy_df_raw = global_data["SPY"].copy()
        if not isinstance(spy_df_raw.index, pd.DatetimeIndex):
            if "date" in spy_df_raw.columns:
                spy_df_raw = spy_df_raw.set_index(pd.to_datetime(spy_df_raw["date"]))
            else:
                spy_df_raw = pd.DataFrame() # Mark as empty if date column is missing
        if not spy_df_raw.empty:
            # CRITICAL FIX: Normalize timezone to avoid comparison errors
            if isinstance(spy_df_raw.index, pd.DatetimeIndex) and spy_df_raw.index.tz is not None:
                spy_df_raw.index = spy_df_raw.index.tz_localize(None)
            try:
                spy_df = _compute_indicators(spy_df_raw) # Ensure SPY has indicators too
            except Exception:
                spy_df = None
        
    # Pre-process Global Data (VIX) for regime filtering
    vix_df_global = None
    if global_data and "VIX" in global_data and global_data["VIX"] is not None:
        vix_df_raw = global_data["VIX"].copy()
        if not isinstance(vix_df_raw.index, pd.DatetimeIndex):
            if "date" in vix_df_raw.columns:
                vix_df_raw = vix_df_raw.set_index(pd.to_datetime(vix_df_raw["date"]))
            else:
                vix_df_raw = pd.DataFrame()
        if not vix_df_raw.empty:
            # CRITICAL FIX: Normalize timezone to avoid comparison errors
            if isinstance(vix_df_raw.index, pd.DatetimeIndex) and vix_df_raw.index.tz is not None:
                vix_df_raw.index = vix_df_raw.index.tz_localize(None)
            vix_df_global = vix_df_raw
        
    for sym in symbols:
        df = data_dict[sym]
        if df is None or df.empty:
            continue

        df_local = df.copy()
        if not isinstance(df_local.index, pd.DatetimeIndex):
            if "date" in df_local.columns:
                df_local = df_local.set_index(pd.to_datetime(df_local["date"]))
            else:
                continue # Skip bad data

        if isinstance(df_local.index, pd.DatetimeIndex) and df_local.index.tz is not None:
            df_local.index = df_local.index.tz_convert(None)

        try:
            df_local = _compute_indicators(df_local)
        except Exception:
            continue
        
        # Compute Relative Strength vs SPY
        if spy_df is not None:
            # Reindex SPY to match Stock's index (ffill to handle missing days)
            spy_reindexed = spy_df["close"].reindex(df_local.index, method="ffill")
            
            # RS Ratio = Stock / SPY
            df_local["rs_ratio"] = df_local["close"] / spy_reindexed
            
            # RS SMA (Trend of RS)
            df_local["rs_sma20"] = df_local["rs_ratio"].rolling(20).mean()
            
            # RS Momentum (Is RS rising?)
            df_local["rs_trend"] = (df_local["rs_ratio"] > df_local["rs_sma20"]).astype(int)
        else:
            # Default if no SPY data
            df_local["rs_ratio"] = 1.0
            df_local["rs_sma20"] = 1.0
            df_local["rs_trend"] = 0
            
        # Merge VIX if available
        if vix_df_global is not None:
            # Reindex VIX to match Stock's index (both are now timezone-naive)
            vix_reindexed = vix_df_global["close"].reindex(df_local.index, method="ffill")
            df_local["vix"] = vix_reindexed
        else:
            df_local["vix"] = 20.0 # Default neutral VIX

        if start_date:
            from_dt = datetime.combine(start_date, datetime.min.time())
            df_local = df_local[df_local.index >= from_dt]

        if len(df_local) < MIN_BARS_FOR_WARMUP:
            continue

        enriched[sym] = df_local

    if not enriched:
        return _empty_result(strategy.name, start_cash, strategy.params)

    # Build global date index
    all_dates = sorted(set().union(*[df.index for df in enriched.values()]))

    # Portfolio state
    cash = start_cash
    positions: Dict[str, Dict] = {}
    equity_history: List[float] = []
    date_history: List[pd.Timestamp] = []
    trade_pnls: List[float] = []
    trade_returns: List[float] = [] # New list for % returns
    winning_trade_durations: List[int] = [] # New: Track duration of winners

    max_positions = 5
    pos_fraction = 0.20

    def get_row_index(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[int]:
        try:
            return df.index.get_loc(dt)
        except KeyError:
            return None

    def get_price_on_or_before(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[float]:
        idx = df.index.searchsorted(dt, side="right") - 1
        if idx < 0:
            return None
        return float(df.iloc[idx]["close"])

    for current_dt in all_dates:
        symbols_with_data = [s for s, df in enriched.items() if current_dt in df.index]

        # 1) Process exits
        for sym in list(positions.keys()):
            if sym not in symbols_with_data:
                continue

            df_sym = enriched[sym]
            i = get_row_index(df_sym, current_dt)
            if i is None:
                continue
            row = df_sym.iloc[i]
            pos = positions[sym]
            
            exit_now = strategy.exit(df_sym, i, pos["entry_i"], pos["entry_price"], pos["stop_price"])

            if exit_now:
                # Use stop fill if breached intraday; otherwise exit at close
                # Use stop fill if breached intraday
                if row["low"] < pos["stop_price"]:
                    # GAP DOWN CHECK: If we opened below the stop, we get filled at Open
                    if row["open"] < pos["stop_price"]:
                        exit_price = row["open"]
                    else:
                        exit_price = pos["stop_price"]
                else:
                    exit_price = row["close"]
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                pnl_pct_trade = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100.0
                cash += pos["shares"] * exit_price
                trade_pnls.append(pnl)
                trade_returns.append(pnl_pct_trade)
                
                # Track duration if winner
                if pnl > 0:
                    duration = i - pos["entry_i"]
                    winning_trade_durations.append(duration)
                    
                del positions[sym]

        # 2) Recalculate equity
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
        open_positions = len(positions)
        if open_positions >= max_positions or equity <= 0:
            continue

        for sym in symbols_with_data:
            if sym in positions:
                continue

            df_sym = enriched[sym]
            i = get_row_index(df_sym, current_dt)
            if i is None:
                continue

            signal_i = i - 1
            if signal_i < MIN_BARS_FOR_WARMUP:
                continue

            # Use the Strategy object
            entry_info = strategy.entry(df_sym, signal_i)

            if not entry_info:
                continue

            stop_price = float(entry_info["stop_price"])
            row_today = df_sym.iloc[i]

            # Determine entry fill rules
            entry_type = entry_info.get("entry_type") or entry_info.get("order_type") or "market"
            desired_price = float(entry_info.get("entry_price", row_today["open"]))
            open_px = float(row_today["open"])

            if entry_type.lower() == "limit":
                day_low = float(row_today.get("low", open_px))
                day_high = float(row_today.get("high", open_px))
                # Fill if the day's range touches the limit; price = better of open or limit
                if day_low <= desired_price <= day_high:
                    if open_px <= desired_price:
                        entry_price = open_px  # gapped down through limit
                    else:
                        entry_price = desired_price
                else:
                    continue  # limit not reached intraday
            else:
                entry_price = open_px
                desired_price = entry_price

            if entry_price <= 0 or math.isnan(entry_price):
                continue

            if stop_price >= entry_price:
                continue
                
            # Filter: Avoid penny stocks / bad data
            if entry_price < 5.0:
                continue

            position_value = equity * pos_fraction
            if position_value <= 0:
                continue

            shares = math.floor(position_value / entry_price)
            if shares <= 0:
                continue

            cost = shares * entry_price
            if cost > cash:
                continue

            cash -= cost
            positions[sym] = {
                "shares": shares,
                "entry_i": i,
                "entry_price": entry_price,
                "stop_price": stop_price,
            }

            open_positions += 1
            if open_positions >= max_positions:
                break

    # Final stats
    final_equity = cash
    if equity_history:
        last_dt = date_history[-1]
        for sym, pos in positions.items():
            df_sym = enriched.get(sym)
            if df_sym is None:
                continue
            price = get_price_on_or_before(df_sym, last_dt)
            if price is None:
                continue
            final_equity += pos["shares"] * price

    pnl = final_equity - start_cash
    pnl_pct = (pnl / start_cash) * 100.0 if start_cash != 0 else 0.0

    total_trades = len(trade_pnls)
    win_total = sum(1 for x in trade_pnls if x > 0)
    hit_rate = (win_total / total_trades) * 100.0 if total_trades > 0 else 0.0
    
    avg_win_duration = np.mean(winning_trade_durations) if winning_trade_durations else 0.0
    duration_std = np.std(winning_trade_durations) if winning_trade_durations else 0.0

    # Max drawdown and Sharpe from equity curve
    max_drawdown_pct = 0.0
    sharpe = 0.0
    cagr = 0.0
    
    if equity_history:
        eq_series = pd.Series(equity_history, index=pd.to_datetime(date_history))
        running_max = eq_series.cummax()
        drawdowns = (eq_series - running_max) / running_max
        max_drawdown_pct = drawdowns.min() * 100.0
        
        # Sharpe
        rets = eq_series.pct_change().dropna()
        if len(rets) > 1 and rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * math.sqrt(252)
            
        # CAGR
        num_days = (eq_series.index[-1] - eq_series.index[0]).days
        if num_days > 0 and start_cash > 0:
            years = num_days / 365.25
            cagr = (final_equity / start_cash) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0

    # Score
    score = cagr * sharpe * (hit_rate / 100.0)

    return {
        "strategy": strategy.name,
        "final_value": final_equity,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "total_trades": total_trades,
        "win_total": win_total,
        "loss_total": total_trades - win_total,
        "hit_rate": hit_rate,
        "max_drawdown_pct": max_drawdown_pct,
        "avg_win_duration": avg_win_duration,
        "duration_std": duration_std,
        "avg_profit_pct": np.mean(trade_returns) if trade_returns else 0.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "Score": score,
        "params": strategy.params
    }

def _empty_result(name, start_cash, params=None):
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
        "avg_win_duration": 0.0,
        "avg_profit_pct": 0.0,
        "cagr": 0.0,
        "sharpe": 0.0,
        "Score": 0.0,
        "params": params if params else {}
    }

def run_compare(
    strategy_names: List[str],
    data_dict: Dict[str, pd.DataFrame],
    symbol_universe: List[str],
    start_cash: float = 100000.0,
    start_date: Optional[date] = None,
    use_parallel: bool = True,
    max_workers: int = 8,
    global_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    Run multiple strategies and return a comparison DataFrame.
    """
    from strategies.rhcts import StrategyRHCTS
    from strategies.connors_rsi import StrategyConnorsRSI
    from strategies.cbc import StrategyCBC
    from strategies.raptor import StrategyRaptor
    from strategies.hmp import StrategyHMP
    from strategies.cgm import StrategyCGM10
    from strategies.r2x import StrategyR2X
    from strategies.mindful import StrategyMindful
    from strategies.ai_optimized import StrategyAIOptimized
    from strategies.first_touch import StrategyFirstTouch
    from strategies.breakout import StrategyBreakout
    from strategies.apex_sniper import StrategyApexSniper
    from strategies.ai_v2 import StrategyAIV2
    from strategies.ai_v3 import StrategyAIV3
    import json
    import os

    # Load config
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/strategies.json'))
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except:
        config = {}

    # Load generated strategies
    gen_strategies = {}
    gen_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/generated_strategies.json'))
    try:
        with open(gen_config_path, "r") as f:
            gen_list = json.load(f)
            for g in gen_list:
                gen_strategies[g["name"]] = g
    except:
        pass

    strategies = []
    
    if "RHCTS" in strategy_names: strategies.append(StrategyRHCTS(config.get("RHCTS", {})))
    if "CRSI" in strategy_names: strategies.append(StrategyConnorsRSI(config.get("ConnorsRSI", {})))
    if "CBC" in strategy_names: strategies.append(StrategyCBC(config.get("CBC", {})))
    if "RAPTOR" in strategy_names: strategies.append(StrategyRaptor(config.get("Raptor", {})))
    if "HMP" in strategy_names: strategies.append(StrategyHMP(config.get("HMP", {})))
    if "CGM10" in strategy_names: strategies.append(StrategyCGM10(config.get("CGM10", {})))
    if "R2X" in strategy_names: strategies.append(StrategyR2X(config.get("R2X", {})))
    if "Mindful" in strategy_names: strategies.append(StrategyMindful(config.get("Mindful", {})))
    if "AI_OPT" in strategy_names: strategies.append(StrategyAIOptimized(config.get("AIOptimized", {})))
    if "FIRST_TOUCH" in strategy_names: strategies.append(StrategyFirstTouch(config.get("FirstTouch", {})))
    if "BREAKOUT" in strategy_names: strategies.append(StrategyBreakout(config.get("Breakout", {})))
    if "APEX_SNIPER" in strategy_names: strategies.append(StrategyApexSniper(config.get("ApexSniper", {})))
    if "AI_V3" in strategy_names: strategies.append(StrategyAIV3(config.get("AIV3", {})))
    
    # Handle generated strategies
    for name in strategy_names:
        if name in gen_strategies:
            from strategies.generic import GenericStrategy
            strategies.append(GenericStrategy(gen_strategies[name]))

    results = []
    
    if use_parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat for strat in strategies}
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    print(f"Comparison failed for {futures[future].name}: {e}")
    else:
        for strat in strategies:
            try:
                res = run_backtest(strat, data_dict, symbol_universe, start_cash, start_date, global_data)
                results.append(res)
            except Exception as e:
                print(f"Comparison failed for {strat.name}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df.sort_values("pnl_pct", ascending=False)
