from __future__ import annotations

import concurrent.futures
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ta.momentum import StochasticOscillator
from ta.trend import ADXIndicator, CCIIndicator

from strategies.strategy_loader import load_strategies

DEFAULT_SCORING_WEIGHTS: Dict[str, float] = {
    "rsi_factor": 3.0,
    "atr_high_bonus": 12.0,
    "atr_med_bonus": 5.0,
    "vol_bonus": 11.0,
    "sniper_bonus": 50.0,
    "trend_bonus": 40.0,
    "trend_penalty": -15.0,
}

MIN_BARS = 200

_ATR_HIGH_THRESH_PCT = 3.0
_ATR_MED_THRESH_PCT = 2.0
_VOL_REL_THRESH = 1.5
_SNIPER_BB_WIDTH_THRESH = 0.17


def get_sector(symbol: str) -> str:
    tech = {"AAPL", "MSFT", "NVDA", "GOOG", "GOOGL", "META", "AMZN", "TSLA", "AVGO", "AMD"}
    symbol_upper = (symbol or "").upper()
    if symbol_upper in tech:
        return "Technology"
    return "Unknown"


def _compute_indicators(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    try:
        df = df.sort_index().copy()
        df.columns = df.columns.str.lower()

        for p in (10, 20, 50, 200):
            df[f"sma{p}"] = df["close"].rolling(p).mean()
            df[f"ema{p}"] = df["close"].ewm(span=p, adjust=False).mean()

        df["bb_upper"] = df["close"].rolling(20).mean() + (df["close"].rolling(20).std() * 2)
        df["bb_lower"] = df["close"].rolling(20).mean() - (df["close"].rolling(20).std() * 2)
        df["bb_mid"] = df["close"].rolling(20).mean()

        mask = df["bb_mid"] != 0
        df["bb_width"] = 0.0
        df.loc[mask, "bb_width"] = (df.loc[mask, "bb_upper"] - df.loc[mask, "bb_lower"]) / df.loc[mask, "bb_mid"]

        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr14"] = tr.rolling(14).mean()
        df["atr14_ma20"] = df["atr14"].rolling(20).mean()

        df["highest20"] = df["high"].rolling(20).max()
        df["highest20_1"] = df["highest20"].shift(1)
        df["highest55"] = df["high"].rolling(55).max()
        df["highest55_1"] = df["highest55"].shift(1)
        df["lowest5"] = df["low"].rolling(5).min()
        df["lowest5_1"] = df["lowest5"].shift(1)

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        g2 = (delta.where(delta > 0, 0)).rolling(2).mean()
        l2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
        rs2 = g2 / l2.replace(0, np.nan)
        df["rsi2"] = 100 - (100 / (1 + rs2))

        adx = ADXIndicator(df["high"], df["low"], df["close"])
        df["adx"] = adx.adx()
        df["plus_di"] = adx.adx_pos()
        df["minus_di"] = adx.adx_neg()

        df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()
        df["stoch_k"] = StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        if spy_df is not None and not spy_df.empty:
            spy_aligned = spy_df["close"].reindex(df.index).ffill().bfill()
            df["rs_ratio"] = df["close"] / spy_aligned
            df["rs_sma20"] = df["rs_ratio"].rolling(20).mean()
            df["rs_trend"] = df["rs_ratio"] - df["rs_sma20"]
            df["spy_close"] = spy_aligned
            df["spy_sma200"] = spy_aligned.rolling(200).mean()
        else:
            df["rs_ratio"] = 1.0
            df["rs_trend"] = 0.0
            df["spy_close"] = np.nan
            df["spy_sma200"] = np.nan

        return df
    except Exception:
        return df


def _empty_result(name: str, start_cash: float, params: Optional[Dict] = None) -> Dict[str, Any]:
    return {
        "strategy": name,
        "final_value": start_cash,
        "total_trades": 0,
        "hit_rate": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "cagr": 0.0,
        "calmar": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_profit_pct": 0.0,
        "avg_days_held": 0.0,
        "exposure_pct": 0.0,
        "profit_factor": 0.0,
        "payoff_ratio": 0.0,
        "max_consecutive_losses": 0,
        "beta": 0.0,
        "avg_signals_per_day": 0.0,
        "Score": 0.0,
        "params": params or {},
        "trades_list": [],
        "equity_curve": [],
    }


@dataclass(frozen=True, slots=True)
class _ScoreWeights:
    rsi_factor: float
    atr_high_bonus: float
    atr_med_bonus: float
    vol_bonus: float
    trend_bonus: float
    trend_penalty: float
    sniper_bonus: float


def _compile_scoring_weights(
    base_override: Optional[Dict[str, float]],
    strategy_override: Optional[Dict[str, float]],
) -> _ScoreWeights:
    merged = dict(DEFAULT_SCORING_WEIGHTS)
    if base_override:
        merged.update(base_override)
    if strategy_override:
        merged.update(strategy_override)

    return _ScoreWeights(
        rsi_factor=float(merged.get("rsi_factor", 0.0) or 0.0),
        atr_high_bonus=float(merged.get("atr_high_bonus", 0.0) or 0.0),
        atr_med_bonus=float(merged.get("atr_med_bonus", 0.0) or 0.0),
        vol_bonus=float(merged.get("vol_bonus", 0.0) or 0.0),
        trend_bonus=float(merged.get("trend_bonus", 0.0) or 0.0),
        trend_penalty=float(merged.get("trend_penalty", 0.0) or 0.0),
        sniper_bonus=float(merged.get("sniper_bonus", 0.0) or 0.0),
    )


def _score_candidate(
    rsi2: float,
    close_px: float,
    atr14: float,
    volume: float,
    vol_ma20: float,
    sma200: float,
    cci: float,
    bb_width: float,
    w: _ScoreWeights,
) -> float:
    np_isfinite = np.isfinite

    if not np_isfinite(rsi2):
        rsi2 = 50.0

    base_score = (100.0 - float(rsi2)) * w.rsi_factor
    base_score = min(100.0, max(0.0, base_score))

    score = base_score

    if np_isfinite(close_px) and close_px > 0 and np_isfinite(atr14) and atr14 > 0:
        atr_pct = (atr14 / close_px) * 100.0
        if atr_pct >= _ATR_HIGH_THRESH_PCT:
            score += w.atr_high_bonus
        elif atr_pct >= _ATR_MED_THRESH_PCT:
            score += w.atr_med_bonus

    if np_isfinite(volume) and np_isfinite(vol_ma20) and vol_ma20 >= 0:
        vol_rel = volume / (vol_ma20 + 1.0)
        if vol_rel >= _VOL_REL_THRESH:
            score += w.vol_bonus

    if np_isfinite(close_px) and close_px > 0 and np_isfinite(sma200) and sma200 > 0:
        score += w.trend_bonus if close_px > sma200 else w.trend_penalty

    if np_isfinite(cci) and np_isfinite(bb_width):
        if cci < 0 and bb_width > _SNIPER_BB_WIDTH_THRESH:
            score += w.sniper_bonus

    return max(0.0, float(score))


def calculate_backtest_quality_score(
    row_or_rsi2: Any,
    strategy_name: str = "",
    weights: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> float:
    """
    Compatibility wrapper.

    - Legacy call sites pass a pandas Series/dict row as the first arg.
    - Performance call sites can pass scalars: `calculate_backtest_quality_score(rsi2, close=..., atr14=..., ...)`.
    """
    _ = strategy_name
    w = _compile_scoring_weights(weights, None)

    if isinstance(row_or_rsi2, (pd.Series, dict)):
        row = row_or_rsi2
        rsi2 = float(row.get("rsi2", 50) or 50)
        close_px = float(row.get("close", 0) or 0)
        atr14 = float(row.get("atr14", 0) or 0)
        volume = float(row.get("volume", 0) or 0)
        vol_ma20 = float(row.get("vol_ma20", 1) or 1)
        sma200 = float(row.get("sma200", np.nan))
        cci = float(row.get("cci", 0) or 0)
        bb_width = float(row.get("bb_width", 0) or 0)
        return _score_candidate(rsi2, close_px, atr14, volume, vol_ma20, sma200, cci, bb_width, w)

    rsi2 = float(row_or_rsi2)
    close_px = float(kwargs.get("close", 0.0) or 0.0)
    atr14 = float(kwargs.get("atr14", 0.0) or 0.0)
    volume = float(kwargs.get("volume", 0.0) or 0.0)
    vol_ma20 = float(kwargs.get("vol_ma20", 1.0) or 1.0)
    sma200 = float(kwargs.get("sma200", np.nan))
    cci = float(kwargs.get("cci", 0.0) or 0.0)
    bb_width = float(kwargs.get("bb_width", 0.0) or 0.0)
    return _score_candidate(rsi2, close_px, atr14, volume, vol_ma20, sma200, cci, bb_width, w)


@dataclass(slots=True)
class _SymbolArrays:
    df: pd.DataFrame
    index: np.ndarray
    open: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    rsi2: np.ndarray
    adx: np.ndarray
    atr14: np.ndarray
    vol_ma20: np.ndarray
    sma50: np.ndarray
    sma200: np.ndarray
    cci: np.ndarray
    bb_width: np.ndarray
    spy_close: np.ndarray
    spy_sma200: np.ndarray


@dataclass(slots=True)
class _Candidate:
    sym: str
    entry_px: float
    stop_px: float
    score: float
    strategy_name: str
    strategy_obj: Any
    entry_i: int
    signal_i: int


def _load_sector_map() -> Dict[str, str]:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "sectors.json"))
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k).upper(): str(v) for k, v in data.items()}
        except Exception:
            return {}
    return {}


def _resolve_gap_protection_ratio(params: Dict[str, Any]) -> float:
    raw_ratio = params.get("gap_protection")
    if raw_ratio is None:
        raw_pct = params.get("gap_protection_pct")
        if raw_pct is None:
            return 0.0
        try:
            pct = float(raw_pct)
        except (TypeError, ValueError):
            return 0.0
        if pct <= 0:
            return 0.0
        return max(0.0, 1.0 - pct)
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        return 0.0
    return ratio if ratio > 0 else 0.0


def _get_np_col(df: pd.DataFrame, col: str, fallback: float, *, length: int) -> np.ndarray:
    if col in df.columns:
        return df[col].to_numpy(dtype=np.float64, copy=False)
    return np.full(length, fallback, dtype=np.float64)


def run_backtest(
    strategy,
    data,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    global_data=None,
    scoring_weights=None,
    export_ml_data=False,
    ai_model=None,
    ai_threshold=0.60,
):
    strategies = strategy if isinstance(strategy, (list, tuple)) else [strategy]
    strategies = [s for s in strategies if s is not None]
    if not strategies:
        return _empty_result("NoStrategy", float(start_cash), {})

    data_dict: Dict[str, pd.DataFrame] = data or {}
    strategy_label = strategies[0].name if len(strategies) == 1 else "MultiStrategy"

    sector_map = _load_sector_map()

    def resolve_sector(sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in sector_map:
            return sector_map[sym_up]
        return get_sector(sym_up)

    ml_feature_cols = ["rsi2", "adx", "atr_pct", "dist_sma50", "dist_sma200", "vol_rel"]
    if ai_model is not None and hasattr(ai_model, "feature_names_in_"):
        try:
            ml_feature_cols = [str(c) for c in ai_model.feature_names_in_]
        except Exception:
            pass

    symbols = list(data_dict.keys())
    if symbol_universe:
        universe_set = set(symbol_universe)
        symbols = [s for s in symbols if s in universe_set]

    vix_df = global_data.get("VIX") if global_data else None
    spy_df = global_data.get("SPY") if global_data else None

    enriched: Dict[str, _SymbolArrays] = {}
    for sym in symbols:
        df_raw = data_dict.get(sym)
        if df_raw is None or df_raw.empty:
            continue

        try:
            df = _compute_indicators(df_raw.copy(), spy_df=spy_df)

            if "ticker" not in df.columns:
                df["ticker"] = sym

            if vix_df is not None and not vix_df.empty and "close" in vix_df.columns:
                df["vix"] = vix_df["close"].reindex(df.index).ffill().fillna(20.0)
            else:
                df["vix"] = 20.0

            if start_date:
                start_dt = pd.to_datetime(start_date).replace(tzinfo=None)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df = df[df.index >= start_dt]

            if len(df) <= MIN_BARS:
                continue

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            n = len(df)
            open_arr = _get_np_col(df, "open", np.nan, length=n)
            low_arr = _get_np_col(df, "low", np.nan, length=n)
            close_arr = _get_np_col(df, "close", np.nan, length=n)
            if not np.isfinite(low_arr).any():
                low_arr = open_arr

            enriched[sym] = _SymbolArrays(
                df=df,
                index=df.index.values.astype("datetime64[ns]"),
                open=open_arr,
                low=low_arr if low_arr is not None else open_arr,
                close=close_arr,
                volume=_get_np_col(df, "volume", 0.0, length=n),
                rsi2=_get_np_col(df, "rsi2", 50.0, length=n),
                adx=_get_np_col(df, "adx", 0.0, length=n),
                atr14=_get_np_col(df, "atr14", 0.0, length=n),
                vol_ma20=_get_np_col(df, "vol_ma20", 1.0, length=n),
                sma50=_get_np_col(df, "sma50", np.nan, length=n),
                sma200=_get_np_col(df, "sma200", np.nan, length=n),
                cci=_get_np_col(df, "cci", 0.0, length=n),
                bb_width=_get_np_col(df, "bb_width", 0.0, length=n),
                spy_close=_get_np_col(df, "spy_close", np.nan, length=n),
                spy_sma200=_get_np_col(df, "spy_sma200", np.nan, length=n),
            )
        except Exception:
            continue

    if not enriched:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    all_index = None
    for sym_data in enriched.values():
        all_index = sym_data.df.index if all_index is None else all_index.union(sym_data.df.index)
    all_index = all_index.sort_values()

    min_date = pd.Timestamp.now() - pd.Timedelta(days=365 * 20)
    all_index = all_index[all_index >= min_date]
    if all_index.empty:
        return _empty_result(strategy_label, float(start_cash), strategies[0].params if strategies else {})

    all_dates = all_index.values.astype("datetime64[ns]")
    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    candidates_by_day: List[List[_Candidate]] = [[] for _ in range(len(all_dates))]

    compiled_strategies: List[Tuple[Any, _ScoreWeights, Dict[str, Any], float, bool, float]] = []
    for strat in strategies:
        params = getattr(strat, "params", {}) if hasattr(strat, "params") else {}
        params = params or {}
        strat_weights = params.get("scoring_weights") if isinstance(params.get("scoring_weights"), dict) else None
        w = _compile_scoring_weights(scoring_weights, strat_weights)
        gap_ratio = _resolve_gap_protection_ratio(params)
        regime_filter = bool(params.get("regime_filter", False))
        base_stop_mult = float(params.get("stop_loss_atr", 3.0) or 3.0)
        compiled_strategies.append((strat, w, params, gap_ratio, regime_filter, base_stop_mult))

    np_isfinite = np.isfinite

    for sym, sd in enriched.items():
        df = sd.df
        idx = sd.index
        open_arr = sd.open
        low_arr = sd.low
        close_arr = sd.close
        volume_arr = sd.volume
        rsi2_arr = sd.rsi2
        adx_arr = sd.adx
        atr14_arr = sd.atr14
        vol_ma20_arr = sd.vol_ma20
        sma50_arr = sd.sma50
        sma200_arr = sd.sma200
        cci_arr = sd.cci
        bb_width_arr = sd.bb_width
        spy_close_arr = sd.spy_close
        spy_sma200_arr = sd.spy_sma200

        n = len(idx)
        if n <= MIN_BARS + 1:
            continue

        for i in range(MIN_BARS + 1, n):
            day_idx = date_to_idx.get(idx[i])
            if day_idx is None:
                continue

            prev_i = i - 1

            prev_close = float(close_arr[prev_i])
            if not np_isfinite(prev_close) or prev_close <= 0:
                continue

            open_px = float(open_arr[i])
            if not np_isfinite(open_px) or open_px <= 0:
                continue

            low_px = float(low_arr[i])
            if not np_isfinite(low_px):
                low_px = open_px

            gap_pct = (open_px - prev_close) / prev_close if prev_close > 0 else 0.0
            if gap_pct < -0.08:
                continue

            for strat, w, params, gap_ratio, regime_filter, base_stop_mult in compiled_strategies:
                entry_signal = strat.entry(df, prev_i)
                if not entry_signal:
                    continue

                if regime_filter:
                    spy_close_prev = float(spy_close_arr[prev_i])
                    spy_sma200_prev = float(spy_sma200_arr[prev_i])
                    if np_isfinite(spy_close_prev) and np_isfinite(spy_sma200_prev):
                        if spy_close_prev < spy_sma200_prev:
                            continue

                if gap_ratio > 0 and prev_close > 0:
                    if open_px < prev_close * gap_ratio:
                        continue

                if ai_model is not None:
                    try:
                        rsi2 = float(rsi2_arr[prev_i])
                        adx = float(adx_arr[prev_i])
                        atr14 = float(atr14_arr[prev_i])
                        sma50 = float(sma50_arr[prev_i])
                        sma200 = float(sma200_arr[prev_i])
                        vol = float(volume_arr[prev_i])
                        vol_ma20 = float(vol_ma20_arr[prev_i])

                        atr_pct = (atr14 / prev_close) if prev_close > 0 else 0.0
                        dist_sma50 = (prev_close - sma50) / prev_close if prev_close > 0 else 0.0
                        dist_sma200 = (prev_close - sma200) / prev_close if prev_close > 0 else 0.0
                        vol_rel = vol / (vol_ma20 + 1.0)

                        features = np.array(
                            [[rsi2, adx, atr_pct, dist_sma50, dist_sma200, vol_rel]],
                            dtype=float,
                        )
                        if np.all(np_isfinite(features)):
                            features_df = pd.DataFrame(features, columns=ml_feature_cols)
                            prob = ai_model.predict_proba(features_df)[0][1]
                            if prob < ai_threshold:
                                continue
                    except Exception:
                        pass

                signal_atr = float(atr14_arr[prev_i])
                if not np_isfinite(signal_atr) or signal_atr <= 0:
                    continue

                entry_px = open_px
                limit_ratio = None
                if isinstance(entry_signal, dict):
                    limit_ratio = entry_signal.get("limit_ratio")
                if limit_ratio is None:
                    limit_ratio = params.get("limit_ratio")

                if limit_ratio is not None and prev_close > 0:
                    try:
                        limit_ratio_val = float(limit_ratio)
                    except (TypeError, ValueError):
                        limit_ratio_val = None
                    if limit_ratio_val is not None:
                        target_px = prev_close * limit_ratio_val
                        if open_px < target_px:
                            entry_px = open_px
                        elif low_px < target_px:
                            entry_px = target_px
                        else:
                            continue

                stop_mult = base_stop_mult
                if isinstance(entry_signal, dict) and "stop_loss_atr" in entry_signal:
                    try:
                        stop_mult = float(entry_signal.get("stop_loss_atr") or base_stop_mult)
                    except (TypeError, ValueError):
                        stop_mult = base_stop_mult

                atr_pct_for_stop = (signal_atr / entry_px) * 100.0 if entry_px > 0 else 0.0
                adjusted_mult = stop_mult
                if gap_pct > -0.03 and atr_pct_for_stop > 5.0:
                    adjusted_mult *= 1.2

                stop_price = entry_px - (signal_atr * adjusted_mult)
                if not np_isfinite(stop_price):
                    continue

                score = _score_candidate(
                    float(rsi2_arr[prev_i]),
                    prev_close,
                    signal_atr,
                    float(volume_arr[prev_i]),
                    float(vol_ma20_arr[prev_i]),
                    float(sma200_arr[prev_i]),
                    float(cci_arr[prev_i]),
                    float(bb_width_arr[prev_i]),
                    w,
                )

                candidates_by_day[day_idx].append(
                    _Candidate(
                        sym=sym,
                        entry_px=float(entry_px),
                        stop_px=float(stop_price),
                        score=float(score),
                        strategy_name=strat.name,
                        strategy_obj=strat,
                        entry_i=i,
                        signal_i=prev_i,
                    )
                )

    for day_list in candidates_by_day:
        if len(day_list) > 1:
            day_list.sort(key=lambda c: c.score, reverse=True)

    cash = float(start_cash)
    positions: Dict[str, Dict[str, Any]] = {}
    equity_curve: List[Dict[str, Any]] = []
    trade_pnls: List[float] = []
    trades_list: List[Dict[str, Any]] = []
    ml_data: Optional[List[Dict[str, Any]]] = [] if export_ml_data else None

    pos_fraction = 0.20
    max_positions = 5
    max_risk_per_trade = 0.02

    for day_idx, current_dt in enumerate(all_dates):
        sector_exposure: Dict[str, float] = {}
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                val = pos["shares"] * pos["entry_price"]
            else:
                loc = int(np.searchsorted(sym_data.index, current_dt))
                if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                    px = float(sym_data.close[loc])
                    if not np_isfinite(px):
                        px = float(pos["entry_price"])
                else:
                    px = float(pos["entry_price"])
                val = float(pos["shares"]) * px
            sec = resolve_sector(sym)
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + float(val)

        daily_candidates = candidates_by_day[day_idx]

        current_sector_equity = float(sum(sector_exposure.values()))
        for cand in daily_candidates:
            if cand.sym in positions:
                continue
            if not export_ml_data and len(positions) >= max_positions:
                break

            current_equity = cash + current_sector_equity
            if current_equity <= 0:
                continue

            risk_amt = current_equity * max_risk_per_trade
            risk_per_share = cand.entry_px - cand.stop_px
            if risk_per_share <= cand.entry_px * 0.01:
                risk_per_share = cand.entry_px * 0.01

            shares_risk = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            val_cap = current_equity * pos_fraction
            shares_val = int(val_cap / cand.entry_px) if cand.entry_px > 0 else 0
            shares = min(shares_risk, shares_val)

            cost = shares * cand.entry_px
            if export_ml_data and (shares <= 0 or cash < cost):
                shares = 100
                cost = 0.0
            elif shares <= 0 or cash < cost:
                continue

            cand_sec = resolve_sector(cand.sym)
            projected_exp = (sector_exposure.get(cand_sec, 0.0) + cost) / current_equity if current_equity > 0 else 1.0
            if not export_ml_data and projected_exp > 0.60:
                continue

            cash -= cost
            positions[cand.sym] = {
                "shares": shares,
                "entry_price": cand.entry_px,
                "stop_price": cand.stop_px,
                "entry_i": cand.entry_i,
                "strategy_name": cand.strategy_name,
                "strategy_obj": cand.strategy_obj,
                "signal_i": cand.signal_i,
            }
            sector_exposure[cand_sec] = sector_exposure.get(cand_sec, 0.0) + cost
            current_sector_equity += cost

        for sym in list(positions.keys()):
            sym_data = enriched.get(sym)
            if sym_data is None:
                continue

            loc = int(np.searchsorted(sym_data.index, current_dt))
            if not (0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt):
                continue

            pos = positions[sym]
            if loc <= int(pos.get("entry_i", 0)):
                continue

            active_strat = pos.get("strategy_obj")
            if not active_strat:
                continue

            try:
                should_exit = bool(active_strat.exit(sym_data.df, loc, pos["entry_i"], pos["entry_price"], pos["stop_price"]))
            except Exception:
                should_exit = False

            if not should_exit:
                continue

            open_px = float(sym_data.open[loc])
            low_px = float(sym_data.low[loc])
            close_px = float(sym_data.close[loc])
            stop_px = float(pos["stop_price"])

            exit_px = stop_px if low_px < stop_px else close_px
            if low_px < stop_px and open_px < stop_px:
                exit_px = open_px
            if exit_px < pos["entry_price"] * 0.5:
                exit_px = pos["entry_price"] * 0.5

            pnl = (exit_px - pos["entry_price"]) * pos["shares"]
            pct = ((exit_px - pos["entry_price"]) / pos["entry_price"]) * 100.0 if pos["entry_price"] else 0.0

            cash += pos["shares"] * exit_px
            trade_pnls.append(float(pnl))

            if export_ml_data and ml_data is not None:
                try:
                    entry_feat_i = int(pos.get("signal_i") or pos.get("entry_i") or 0)
                    entry_feat_i = max(0, min(entry_feat_i, len(sym_data.index) - 1))

                    close_entry = float(sym_data.close[entry_feat_i])
                    atr14_entry = float(sym_data.atr14[entry_feat_i])
                    sma50_entry = float(sym_data.sma50[entry_feat_i])
                    sma200_entry = float(sym_data.sma200[entry_feat_i])
                    vol_entry = float(sym_data.volume[entry_feat_i])
                    vol_ma20_entry = float(sym_data.vol_ma20[entry_feat_i])

                    dist50 = (close_entry - sma50_entry) / close_entry if close_entry else 0.0
                    dist200 = (close_entry - sma200_entry) / close_entry if close_entry else 0.0
                    atr_pct = atr14_entry / close_entry if close_entry else 0.0
                    vol_rel = vol_entry / (vol_ma20_entry + 1.0)

                    ml_data.append(
                        {
                            "rsi2": float(sym_data.rsi2[entry_feat_i]),
                            "adx": float(sym_data.adx[entry_feat_i]),
                            "atr14": atr14_entry,
                            "atr_pct": float(atr_pct),
                            "vol_rel": float(vol_rel),
                            "dist_to_sma50": float(dist50),
                            "dist_to_sma200": float(dist200),
                            "dist_sma50": float(dist50),
                            "dist_sma200": float(dist200),
                            "profit_pct": float(pct),
                            "outcome": 1 if pct > 0 else 0,
                        }
                    )
                except Exception:
                    pass

            trades_list.append(
                {
                    "Symbol": sym,
                    "Entry Date": str(sym_data.df.index[pos["entry_i"]].date()),
                    "Exit Date": str(pd.Timestamp(current_dt).date()),
                    "Entry": pos["entry_price"],
                    "Exit": exit_px,
                    "PnL": pnl,
                    "Return%": pct,
                    "Strategy": pos.get("strategy_name", strategy_label),
                }
            )
            del positions[sym]

        equity = cash
        for sym, pos in positions.items():
            sym_data = enriched.get(sym)
            if sym_data is None:
                equity += pos["shares"] * pos["entry_price"]
                continue
            loc = int(np.searchsorted(sym_data.index, current_dt))
            if 0 <= loc < len(sym_data.index) and sym_data.index[loc] == current_dt:
                px = float(sym_data.close[loc])
                if not np_isfinite(px):
                    px = float(pos["entry_price"])
            else:
                px = float(pos["entry_price"])
            equity += float(pos["shares"]) * px

        equity_curve.append({"Date": pd.Timestamp(current_dt), "Equity": float(equity)})

    trades = len(trade_pnls)
    wins = sum(1 for t in trade_pnls if t > 0)
    hit_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    avg_profit = (sum(t.get("Return%", 0.0) for t in trades_list) / len(trades_list)) if trades_list else 0.0

    eq_df = pd.DataFrame(equity_curve).set_index("Date")
    if eq_df.empty:
        eq_df = pd.DataFrame({"Equity": [float(start_cash)]}, index=[pd.Timestamp.now()])

    days = int((eq_df.index[-1] - eq_df.index[0]).days)
    years = days / 365.25 if days > 0 else 1.0
    cagr = ((eq_df["Equity"].iloc[-1] / float(start_cash)) ** (1.0 / years)) - 1.0 if start_cash else 0.0

    rolling_max = eq_df["Equity"].cummax()
    drawdowns = (eq_df["Equity"] - rolling_max) / rolling_max
    max_dd = float(drawdowns.min() * 100.0) if not drawdowns.empty else 0.0

    if export_ml_data and ml_data:
        try:
            out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml_training_data.csv"))
            pd.DataFrame(ml_data).to_csv(out_path, index=False)
            print(f"🧠 ML Data Exported: {len(ml_data)} samples saved to {out_path}")
        except Exception as e:
            print(f"⚠️ ML Data Export failed: {e}")

    return {
        "strategy": strategy_label,
        "final_value": float(eq_df["Equity"].iloc[-1]) if not eq_df.empty else float(start_cash),
        "total_trades": trades,
        "hit_rate": hit_rate,
        "cagr": float(cagr),
        "avg_profit_pct": float(avg_profit),
        "max_drawdown_pct": float(max_dd),
        "equity_curve": eq_df,
        "trades_list": trades_list,
        "params": strategies[0].params if strategies else {},
        "Score": float(cagr) * 1000.0,
    }


def run_compare(
    strategy_names,
    data_dict,
    symbol_universe=None,
    start_cash=100000.0,
    start_date=None,
    use_parallel=True,
    max_workers=8,
    global_data=None,
):
    gen_strategies = {}
    try:
        with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "generated_strategies.json")), "r") as f:
            for g in json.load(f):
                gen_strategies[g["name"]] = g
    except Exception:
        pass

    strategies = load_strategies([gen_strategies[name] for name in strategy_names if name in gen_strategies])
    if not strategies:
        return pd.DataFrame()

    results = []
    if use_parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_backtest, strat, data_dict, symbol_universe, start_cash, start_date, global_data): strat
                for strat in strategies
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
    else:
        for strat in strategies:
            try:
                results.append(run_backtest(strat, data_dict, symbol_universe, start_cash, start_date, global_data))
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["equity_curve", "trades_list"]} for r in results])
    return df.sort_values("Score", ascending=False)

