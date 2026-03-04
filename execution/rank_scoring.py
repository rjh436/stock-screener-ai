from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def _safe_series(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _slice_cross_section(frame: pd.DataFrame, date_idx: object = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    if isinstance(frame.index, pd.MultiIndex):
        lvl0 = frame.index.get_level_values(0)
        unique_dates = pd.Index(pd.to_datetime(lvl0)).sort_values().unique()
        if len(unique_dates) == 0:
            return pd.DataFrame(index=pd.Index([], dtype=object))

        if date_idx is None:
            chosen = unique_dates[-1]
        else:
            target = pd.Timestamp(date_idx)
            eligible = unique_dates[unique_dates <= target]
            chosen = eligible[-1] if len(eligible) else unique_dates[0]

        xs = frame.xs(chosen, level=0, drop_level=True)
        if isinstance(xs, pd.Series):
            xs = xs.to_frame().T
        if "symbol" in xs.columns:
            xs = xs.set_index("symbol")
        xs.index = xs.index.astype(str)
        return xs

    if {"date", "symbol"}.issubset(set(frame.columns)):
        local = frame.copy()
        local["date"] = pd.to_datetime(local["date"], errors="coerce")
        local = local.dropna(subset=["date"])
        if local.empty:
            return pd.DataFrame(index=pd.Index([], dtype=object))
        if date_idx is None:
            chosen = local["date"].max()
        else:
            target = pd.Timestamp(date_idx)
            eligible = local.loc[local["date"] <= target, "date"]
            chosen = eligible.max() if not eligible.empty else local["date"].min()
        snap = local.loc[local["date"] == chosen].copy()
        snap = snap.set_index("symbol")
        snap.index = snap.index.astype(str)
        return snap

    snap = frame.copy()
    if "symbol" in snap.columns:
        snap = snap.set_index("symbol")
    snap.index = snap.index.astype(str)
    return snap


def _percentile_rank(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    series = pd.to_numeric(values, errors="coerce")
    ranked = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if valid.empty:
        return ranked

    # ascending=True yields larger percentile for larger values.
    pct = valid.rank(method="average", pct=True, ascending=True)
    if not higher_is_better:
        pct = 1.0 - pct
    ranked.loc[pct.index] = pct * 100.0
    return ranked


def _weighted_average_rank(columns: Mapping[str, pd.Series], weights: Mapping[str, float]) -> pd.Series:
    if not columns:
        return pd.Series(dtype=float)

    base_index = next(iter(columns.values())).index
    out = pd.Series(0.0, index=base_index, dtype=float)
    den = pd.Series(0.0, index=base_index, dtype=float)

    for name, series in columns.items():
        weight = float(weights.get(name, 0.0) or 0.0)
        if weight <= 0:
            continue
        valid = pd.to_numeric(series, errors="coerce")
        mask = valid.notna()
        out.loc[mask] += valid.loc[mask] * weight
        den.loc[mask] += weight

    with np.errstate(invalid="ignore", divide="ignore"):
        combined = out / den
    combined[den <= 0] = np.nan
    return combined


def compute_momentum_rank(frame: pd.DataFrame, date_idx: object = None, params: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    cfg = dict(params or {})
    snap = _slice_cross_section(frame, date_idx)
    if snap.empty:
        return pd.DataFrame(columns=["momentum_rank"])

    mom_12_1 = _safe_series(snap, ["mom_12_1", "ret_12_1", "momentum_12_1"])
    mom_6_1 = _safe_series(snap, ["mom_6_1", "ret_6_1", "momentum_6_1"])

    proximity = _safe_series(snap, ["proximity_52w", "near_52w", "high_proximity"])
    if proximity.isna().all() and "pct_off_high_52w" in snap.columns:
        proximity = 1.0 - pd.to_numeric(snap["pct_off_high_52w"], errors="coerce")

    quality_proxy_cols = [
        c
        for c in ("quality_growth", "eps_growth_yoy", "sales_growth_yoy", "roe", "roic", "f_score")
        if c in snap.columns
    ]
    if quality_proxy_cols:
        quality_parts: Dict[str, pd.Series] = {}
        for col in quality_proxy_cols:
            quality_parts[str(col)] = _percentile_rank(pd.to_numeric(snap[col], errors="coerce"), higher_is_better=True)
        quality_growth = _weighted_average_rank(quality_parts, {k: 1.0 for k in quality_parts})
    else:
        quality_growth = pd.Series(np.nan, index=snap.index, dtype=float)

    ranked = {
        "mom_12_1": _percentile_rank(mom_12_1, higher_is_better=True),
        "mom_6_1": _percentile_rank(mom_6_1, higher_is_better=True),
        "proximity_52w": _percentile_rank(proximity, higher_is_better=True),
        "quality_growth": _percentile_rank(quality_growth, higher_is_better=True),
    }

    weights = {
        "mom_12_1": float(cfg.get("mom_12_1_weight", 0.50) or 0.50),
        "mom_6_1": float(cfg.get("mom_6_1_weight", 0.20) or 0.20),
        "proximity_52w": float(cfg.get("proximity_52w_weight", 0.15) or 0.15),
        "quality_growth": float(cfg.get("quality_growth_weight", 0.15) or 0.15),
    }

    out = pd.DataFrame(ranked)
    out["momentum_rank"] = _weighted_average_rank(ranked, weights)
    return out.sort_values("momentum_rank", ascending=False, na_position="last")


def compute_value_rank(frame: pd.DataFrame, date_idx: object = None, params: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    cfg = dict(params or {})
    snap = _slice_cross_section(frame, date_idx)
    if snap.empty:
        return pd.DataFrame(columns=["value_rank"])

    high_cols = cfg.get("value_high_cols", ["ebit_tev", "fcf_yield", "earnings_yield", "book_to_market"])
    low_cols = cfg.get("value_low_cols", ["pe", "pb", "ps", "ev_ebitda"])

    ranks: Dict[str, pd.Series] = {}
    weights: Dict[str, float] = {}

    for col in high_cols:
        if col in snap.columns:
            key = str(col)
            ranks[key] = _percentile_rank(pd.to_numeric(snap[col], errors="coerce"), higher_is_better=True)
            weights[key] = 1.0
    for col in low_cols:
        if col in snap.columns:
            key = str(col)
            ranks[key] = _percentile_rank(pd.to_numeric(snap[col], errors="coerce"), higher_is_better=False)
            weights[key] = 1.0

    out = pd.DataFrame(ranks, index=snap.index)
    out["value_rank"] = _weighted_average_rank(ranks, weights)
    return out.sort_values("value_rank", ascending=False, na_position="last")


def compute_quality_rank(frame: pd.DataFrame, date_idx: object = None, params: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    cfg = dict(params or {})
    snap = _slice_cross_section(frame, date_idx)
    if snap.empty:
        return pd.DataFrame(columns=["quality_rank"])

    high_cols = cfg.get(
        "quality_high_cols",
        ["roe", "roic", "gross_margin", "operating_margin", "f_score", "earnings_stability"],
    )
    low_cols = cfg.get("quality_low_cols", ["debt_to_equity", "accruals"])

    ranks: Dict[str, pd.Series] = {}
    weights: Dict[str, float] = {}

    for col in high_cols:
        if col in snap.columns:
            key = str(col)
            ranks[key] = _percentile_rank(pd.to_numeric(snap[col], errors="coerce"), higher_is_better=True)
            weights[key] = 1.0
    for col in low_cols:
        if col in snap.columns:
            key = str(col)
            ranks[key] = _percentile_rank(pd.to_numeric(snap[col], errors="coerce"), higher_is_better=False)
            weights[key] = 1.0

    out = pd.DataFrame(ranks, index=snap.index)
    out["quality_rank"] = _weighted_average_rank(ranks, weights)
    return out.sort_values("quality_rank", ascending=False, na_position="last")


def combine_rank_columns(frame: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float, name="combined_rank")

    cols = {}
    clean_weights: Dict[str, float] = {}
    for col, weight in dict(weights or {}).items():
        if col not in frame.columns:
            continue
        w = float(weight or 0.0)
        if not np.isfinite(w) or w <= 0:
            continue
        cols[str(col)] = pd.to_numeric(frame[col], errors="coerce")
        clean_weights[str(col)] = w

    if not cols:
        return pd.Series(np.nan, index=frame.index, name="combined_rank", dtype=float)

    combined = _weighted_average_rank(cols, clean_weights)
    combined.name = "combined_rank"
    return combined
