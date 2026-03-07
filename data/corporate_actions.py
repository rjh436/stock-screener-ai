from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf


CACHE_DIR = Path("data") / "cache_actions_v2"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_META_DATE = pd.Timestamp("1900-01-01")
_META_ROW_TYPE = "meta_ok"
_SPLIT_ROW_TYPE = "split"


def _cache_path(symbol: str) -> Path:
    clean = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(symbol or "").upper())
    return CACHE_DIR / f"{clean}.parquet"


def _normalize_split_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    if not isinstance(series, pd.Series):
        try:
            series = pd.Series(series)
        except Exception:
            return pd.Series(dtype="float64")
    if series.empty:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(series, errors="coerce").dropna()
    if out.empty:
        return pd.Series(dtype="float64")
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce")).tz_localize(None).normalize()
    out.index = idx
    out = out.groupby(level=0).prod()
    out = out[(out > 0) & np.isfinite(out)]
    return out.astype("float64").sort_index()


def _read_cached_series(path: Path) -> pd.Series | None:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.empty or "row_type" not in frame.columns or "split_ratio" not in frame.columns:
        return None
    row_types = frame["row_type"].astype(str)
    if not (row_types == _META_ROW_TYPE).any():
        return None
    split_rows = frame.loc[row_types == _SPLIT_ROW_TYPE, "split_ratio"]
    return _normalize_split_series(split_rows)


def load_split_series(symbol: str, *, allow_fetch: bool = True, force_refresh: bool = False) -> pd.Series | None:
    path = _cache_path(symbol)
    if path.exists() and not force_refresh:
        cached = _read_cached_series(path)
        if cached is not None:
            return cached

    if not allow_fetch:
        return None

    try:
        ticker = yf.Ticker(str(symbol or "").upper())
        series = _normalize_split_series(ticker.splits)
    except Exception:
        return None

    save_split_series(symbol, series, fetched_ok=True)
    return series


def save_split_series(symbol: str, series: pd.Series | None, *, fetched_ok: bool = True) -> None:
    if not fetched_ok:
        return
    path = _cache_path(symbol)
    normalized = _normalize_split_series(series)
    split_frame = pd.DataFrame({"split_ratio": normalized.astype("float64"), "row_type": _SPLIT_ROW_TYPE})
    meta_frame = pd.DataFrame({"split_ratio": [np.nan], "row_type": [_META_ROW_TYPE]}, index=pd.DatetimeIndex([_META_DATE]))
    frame = meta_frame if split_frame.empty else pd.concat([split_frame, meta_frame], axis=0).sort_index()
    frame.index.name = "date"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


def backfill_split_cache(
    symbols: Sequence[str],
    *,
    batch_size: int = 50,
    force_refresh: bool = False,
) -> dict[str, int]:
    requested = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    if not requested:
        return {"requested": 0, "written": 0, "fetched": 0, "failed": 0}

    pending: list[str] = []
    written = 0
    fetched = 0
    failed = 0
    for sym in requested:
        if force_refresh or not _cache_path(sym).exists():
            pending.append(sym)
        else:
            cached = _read_cached_series(_cache_path(sym))
            if cached is not None:
                written += 1
            else:
                pending.append(sym)

    step = max(1, int(batch_size))
    for start in range(0, len(pending), step):
        batch = pending[start : start + step]
        try:
            data = yf.download(
                " ".join(batch),
                period="max",
                auto_adjust=False,
                actions=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception:
            data = None

        for sym in batch:
            series: pd.Series | None = None
            fetched_ok = False
            if isinstance(data, pd.DataFrame) and not data.empty:
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if sym in data.columns.get_level_values(0):
                            candidate = data[sym].get("Stock Splits")
                            if candidate is not None:
                                series = _normalize_split_series(candidate[candidate != 0])
                                fetched_ok = True
                    elif "Stock Splits" in data.columns and len(batch) == 1:
                        series = _normalize_split_series(data["Stock Splits"][data["Stock Splits"] != 0])
                        fetched_ok = True
                except Exception:
                    series = None
                    fetched_ok = False
            if not fetched_ok:
                try:
                    ticker = yf.Ticker(sym)
                    series = _normalize_split_series(ticker.splits)
                    fetched_ok = True
                except Exception:
                    series = None
                    fetched_ok = False

            if fetched_ok:
                save_split_series(sym, series, fetched_ok=True)
                written += 1
                fetched += 1
            else:
                failed += 1

    return {"requested": len(requested), "written": written, "fetched": fetched, "failed": failed}


def build_split_adjustment_factor(index: Iterable[pd.Timestamp], split_series: pd.Series | None) -> pd.Series:
    dt_index = pd.DatetimeIndex(pd.to_datetime(list(index), errors="coerce")).tz_localize(None)
    if len(dt_index) == 0:
        return pd.Series(dtype="float64")
    factors = pd.Series(1.0, index=dt_index, dtype="float64")
    splits = _normalize_split_series(split_series)
    if splits.empty:
        return factors

    normalized_dates = dt_index.normalize()
    for split_dt, ratio in splits.items():
        match = normalized_dates == pd.Timestamp(split_dt).normalize()
        if np.any(match):
            factors.iloc[np.where(match)[0]] *= float(ratio)

    future_factors = factors.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
    return future_factors.astype("float64")


def build_nominal_price_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    split_series: pd.Series | None = None,
    allow_fetch: bool = True,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    cached_or_fetched = split_series
    if cached_or_fetched is None:
        cached_or_fetched = load_split_series(symbol, allow_fetch=allow_fetch)
    if cached_or_fetched is None:
        return pd.DataFrame(index=pd.DatetimeIndex(frame.index))
    splits = _normalize_split_series(cached_or_fetched)
    factor = build_split_adjustment_factor(frame.index, splits)
    out = pd.DataFrame(index=pd.DatetimeIndex(frame.index))
    for col in ("open", "high", "low", "close"):
        if col in frame.columns:
            series = pd.to_numeric(frame[col], errors="coerce").astype("float64")
            out[f"raw_{col}"] = series * factor.reindex(series.index).to_numpy(dtype="float64")
    return out
