from __future__ import annotations

import os
from datetime import timezone
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
try:
    import polars as pl
except Exception:
    pl = None  # type: ignore[assignment]

from .schwab_client import sd

FUNDAMENTAL_METRIC_COLUMNS: List[str] = [
    "eps_growth_qoq",
    "eps_growth_yoy",
    "sales_growth_qoq",
    "sales_growth_yoy",
    "eps_accel",
    "revenue_accel",
    "institutional_sponsorship",
    "eps_ttm",
    "revenue_ttm",
    "net_income_ttm",
    "net_margin_ttm",
]

_FUND_CACHE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "cache", "fundamentals_history.csv")
)
_EDGAR_BASE_DIR = os.path.abspath(
    os.getenv(
        "FUNDAMENTAL_EDGAR_DIR",
        os.path.join(os.path.dirname(__file__), "fundamentals", "edgar_income"),
    )
)
_DEFAULT_TTL_HOURS = 24
_API_CHUNK_SIZE = 200


def _strict_fundamental_mode() -> bool:
    return str(
        os.getenv(
            "FUNDAMENTAL_STRICT_MODE",
            os.getenv("APEX_STRICT_FUNDAMENTALS", "1"),
        )
        or "1"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _allow_estimated_available_date() -> bool:
    return str(
        os.getenv("FUNDAMENTAL_ALLOW_ESTIMATED_AVAILABLE_DATE", "0") or "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _fundamental_release_lag_days() -> int:
    try:
        lag_days = int(os.getenv("FUNDAMENTAL_RELEASE_LAG_DAYS", "45") or "45")
    except Exception:
        lag_days = 45
    return max(1, lag_days)


def _safe_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _parse_dt(value) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts).normalize()


def _quarter_end(ts: pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None or pd.isna(ts):
        return None
    try:
        return ts.to_period("Q").end_time.normalize()
    except Exception:
        return None


def _chunks(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    n = len(items)
    for i in range(0, n, size):
        yield items[i : i + size]


def _empty_cache_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "report_date",
            "available_date",
            "fetched_at",
            *FUNDAMENTAL_METRIC_COLUMNS,
        ]
    )


def _load_cache() -> pd.DataFrame:
    if not os.path.exists(_FUND_CACHE_PATH):
        return _empty_cache_df()
    try:
        df = pd.read_csv(_FUND_CACHE_PATH)
    except Exception:
        return _empty_cache_df()

    if df.empty:
        return _empty_cache_df()

    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    if "available_date" in df.columns:
        df["available_date"] = pd.to_datetime(df["available_date"], errors="coerce")
    else:
        df["available_date"] = pd.NaT
    if "fetched_at" in df.columns:
        df["fetched_at"] = pd.to_datetime(df["fetched_at"], errors="coerce")
    if (not _strict_fundamental_mode()) or _allow_estimated_available_date():
        lag_days = _fundamental_release_lag_days()
        if "report_date" in df.columns:
            fallback_avail = df["report_date"] + pd.Timedelta(days=lag_days)
            df["available_date"] = df["available_date"].where(df["available_date"].notna(), fallback_avail)

    for col in FUNDAMENTAL_METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "symbol" not in df.columns:
        df["symbol"] = ""
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def _save_cache(df: pd.DataFrame) -> None:
    try:
        os.makedirs(os.path.dirname(_FUND_CACHE_PATH), exist_ok=True)
        out = df.copy()
        out = out.sort_values(["symbol", "available_date", "report_date", "fetched_at"])
        out.to_csv(_FUND_CACHE_PATH, index=False)
    except Exception:
        pass


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for sym in symbols or []:
        s = str(sym or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    return cleaned


def _empty_symbol_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=FUNDAMENTAL_METRIC_COLUMNS)


def _edgar_partition_path(symbol: str) -> str:
    return os.path.join(_EDGAR_BASE_DIR, f"ticker={symbol}", "fundamentals.parquet")


def _load_edgar_symbol_frame(symbol: str) -> pd.DataFrame:
    path = _edgar_partition_path(symbol)
    if not os.path.exists(path):
        return _empty_symbol_frame()

    try:
        if pl is not None:
            edf = pl.read_parquet(path)
            if edf.is_empty():
                return _empty_symbol_frame()
            pdf = edf.to_pandas()
        else:
            pdf = pd.read_parquet(path)
    except Exception:
        return _empty_symbol_frame()

    if pdf is None or pdf.empty:
        return _empty_symbol_frame()

    quarter_col = "quarter_end" if "quarter_end" in pdf.columns else "report_date"
    if quarter_col not in pdf.columns:
        return _empty_symbol_frame()

    report_date = pd.to_datetime(pdf[quarter_col], errors="coerce").dt.normalize()
    filing_date = pd.to_datetime(pdf.get("filing_date"), errors="coerce").dt.normalize()
    if _strict_fundamental_mode() and not _allow_estimated_available_date():
        available_date = filing_date
    else:
        lag_days = _fundamental_release_lag_days()
        available_date = filing_date.where(filing_date.notna(), report_date + pd.Timedelta(days=lag_days))

    revenue = pd.to_numeric(pdf.get("revenue"), errors="coerce")
    net_income = pd.to_numeric(pdf.get("net_income"), errors="coerce")
    eps = pd.to_numeric(pdf.get("eps"), errors="coerce")

    frame = pd.DataFrame(
        {
            "report_date": report_date,
            "available_date": available_date,
            "eps_growth_qoq": pd.to_numeric(pdf.get("eps_qoq_growth_pct"), errors="coerce"),
            "eps_growth_yoy": pd.to_numeric(pdf.get("eps_yoy_growth_pct"), errors="coerce"),
            "sales_growth_qoq": pd.to_numeric(pdf.get("revenue_qoq_growth_pct"), errors="coerce"),
            "sales_growth_yoy": pd.to_numeric(pdf.get("revenue_yoy_growth_pct"), errors="coerce"),
            "institutional_sponsorship": np.nan,
            "revenue": revenue,
            "net_income": net_income,
            "eps": eps,
        }
    )
    frame = frame.dropna(subset=["available_date"]).sort_values("available_date")
    if frame.empty:
        return _empty_symbol_frame()
    frame = frame.drop_duplicates(subset=["available_date"], keep="last")
    frame["eps_ttm"] = frame["eps"].rolling(window=4, min_periods=4).sum()
    frame["revenue_ttm"] = frame["revenue"].rolling(window=4, min_periods=4).sum()
    frame["net_income_ttm"] = frame["net_income"].rolling(window=4, min_periods=4).sum()
    frame["eps_accel"] = (
        frame["eps_growth_qoq"]
        - frame["eps_growth_qoq"].shift(1).rolling(window=3, min_periods=2).mean()
    )
    frame["revenue_accel"] = (
        frame["sales_growth_qoq"]
        - frame["sales_growth_qoq"].shift(1).rolling(window=3, min_periods=2).mean()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["net_margin_ttm"] = (frame["net_income_ttm"] / frame["revenue_ttm"]) * 100.0
    frame["net_margin_ttm"] = pd.to_numeric(frame["net_margin_ttm"], errors="coerce")

    frame = frame.set_index("available_date")[FUNDAMENTAL_METRIC_COLUMNS].sort_index()
    return frame


def _load_edgar_fundamental_data(symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = _load_edgar_symbol_frame(sym)
    return out


def _latest_fetched_map(cache_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
    if cache_df.empty or "fetched_at" not in cache_df.columns:
        return {}
    tmp = cache_df.dropna(subset=["fetched_at"])
    if tmp.empty:
        return {}
    latest = tmp.sort_values("fetched_at").groupby("symbol", as_index=False).tail(1)
    return {
        str(row["symbol"]).upper(): pd.Timestamp(row["fetched_at"])
        for _, row in latest.iterrows()
        if str(row.get("symbol", "")).strip()
    }


def _extract_instrument_rows(payload: Dict) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    if not isinstance(payload, dict):
        return rows

    instruments = payload.get("instruments")
    if isinstance(instruments, list):
        for inst in instruments:
            if not isinstance(inst, dict):
                continue
            fund = inst.get("fundamental")
            if not isinstance(fund, dict):
                fund = {}
            sym = str(inst.get("symbol") or fund.get("symbol") or "").upper()
            if not sym:
                continue
            rows[sym] = fund
        return rows

    # Defensive fallback for alternate payload formats
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        fund = value.get("fundamental")
        if not isinstance(fund, dict):
            continue
        sym = str(fund.get("symbol") or key or "").upper()
        if not sym:
            continue
        rows[sym] = fund
    return rows


def _institutional_sponsorship_proxy(fund: Dict) -> float:
    # Schwab instrument fundamentals do not expose direct institutional ownership.
    # Use a liquidity sponsorship proxy derived from float turnover.
    avg_vol = _safe_float(fund.get("avg3MonthVolume"), np.nan)
    shares_out = _safe_float(fund.get("sharesOutstanding"), np.nan)
    market_cap = _safe_float(fund.get("marketCap"), np.nan)
    market_cap_float = _safe_float(fund.get("marketCapFloat"), np.nan)

    turnover_score = np.nan
    if np.isfinite(avg_vol) and np.isfinite(shares_out) and shares_out > 0:
        # Approximate quarterly turnover, clipped to [0, 100]
        turnover_score = float(np.clip((avg_vol * 63.0 / shares_out) * 100.0, 0.0, 100.0))

    float_score = np.nan
    if np.isfinite(market_cap) and market_cap > 0 and np.isfinite(market_cap_float) and market_cap_float > 0:
        float_score = float(np.clip((market_cap_float / market_cap) * 100.0, 0.0, 100.0))

    if np.isfinite(turnover_score) and np.isfinite(float_score):
        return (0.7 * turnover_score) + (0.3 * float_score)
    if np.isfinite(turnover_score):
        return turnover_score
    if np.isfinite(float_score):
        return float_score
    return np.nan


def _fetch_snapshots(symbols: Sequence[str]) -> pd.DataFrame:
    symbols = _normalize_symbols(symbols)
    if not symbols:
        return _empty_cache_df().iloc[0:0]

    fetched_at = pd.Timestamp.now(tz=timezone.utc).tz_convert(None)
    instrument_rows: Dict[str, Dict] = {}
    quote_rows: Dict[str, Dict] = {}

    try:
        sd._ensure()
    except Exception:
        return _empty_cache_df().iloc[0:0]

    try:
        quote_rows = sd.get_quotes(symbols) or {}
    except Exception:
        quote_rows = {}

    for chunk in _chunks(symbols, _API_CHUNK_SIZE):
        try:
            resp = sd._cli.get_instruments(chunk, sd._cli.Instrument.Projection.FUNDAMENTAL)
            payload = resp.json()
            instrument_rows.update(_extract_instrument_rows(payload))
        except Exception:
            continue

    rows = []
    for sym in symbols:
        fund = instrument_rows.get(sym, {})
        if not isinstance(fund, dict):
            fund = {}

        quote_payload = quote_rows.get(sym) or quote_rows.get(sym.upper()) or {}
        if isinstance(quote_payload, dict):
            quote_fund = quote_payload.get("fundamental", {})
        else:
            quote_fund = {}
        if not isinstance(quote_fund, dict):
            quote_fund = {}

        earnings_dt = _parse_dt(quote_fund.get("lastEarningsDate"))
        if earnings_dt is None:
            earnings_dt = _parse_dt(fund.get("declarationDate"))
        if earnings_dt is None:
            earnings_dt = _parse_dt(fund.get("dividendDate"))
        report_date = _quarter_end(earnings_dt)
        if report_date is None:
            report_date = _quarter_end(fetched_at)
        available_date = earnings_dt
        if available_date is None:
            available_date = report_date + pd.Timedelta(days=_fundamental_release_lag_days())

        eps_qoq = _safe_float(fund.get("epsChange"), np.nan)
        eps_yoy = _safe_float(fund.get("epsChangeYear"), np.nan)
        if not np.isfinite(eps_yoy):
            eps_yoy = _safe_float(fund.get("epsChangePercentTTM"), np.nan)

        sales_qoq = _safe_float(fund.get("revChangeIn"), np.nan)
        if not np.isfinite(sales_qoq):
            sales_qoq = _safe_float(fund.get("revChangeTTM"), np.nan)
        sales_yoy = _safe_float(fund.get("revChangeYear"), np.nan)

        rows.append(
            {
                "symbol": sym,
                "report_date": report_date,
                "available_date": available_date,
                "fetched_at": fetched_at,
                "eps_growth_qoq": eps_qoq,
                "eps_growth_yoy": eps_yoy,
                "sales_growth_qoq": sales_qoq,
                "sales_growth_yoy": sales_yoy,
                "eps_accel": np.nan,
                "revenue_accel": np.nan,
                "institutional_sponsorship": _institutional_sponsorship_proxy(fund),
                "eps_ttm": np.nan,
                "revenue_ttm": np.nan,
                "net_income_ttm": np.nan,
                "net_margin_ttm": np.nan,
            }
        )

    if not rows:
        return _empty_cache_df().iloc[0:0]
    return pd.DataFrame(rows)


def fetch_fundamental_data(symbols: Sequence[str]) -> Dict[str, pd.DataFrame]:
    """
    Fetch point-in-time fundamentals keyed by tradable availability date.

    In strict mode (default), this returns EDGAR-derived data only and does not
    synthesize historical availability from snapshot/cache fields.

    Returns a dict keyed by symbol where each value is:
        index: availability_date (first date fundamentals are tradable)
        cols:  eps_growth_qoq, eps_growth_yoy, sales_growth_qoq, sales_growth_yoy,
               institutional_sponsorship
    """
    symbols = _normalize_symbols(symbols)
    if not symbols:
        return {}

    edgar_data = _load_edgar_fundamental_data(symbols)
    edgar_covered = {sym for sym, frame in edgar_data.items() if frame is not None and not frame.empty}
    strict_mode = _strict_fundamental_mode()

    if strict_mode:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            frame = edgar_data.get(sym)
            if frame is None or frame.empty:
                out[sym] = _empty_symbol_frame()
            else:
                out[sym] = frame
        return out

    cache_df = _load_cache()
    latest_map = _latest_fetched_map(cache_df)

    try:
        ttl_hours = int(os.getenv("FUNDAMENTAL_CACHE_TTL_HOURS", str(_DEFAULT_TTL_HOURS)) or _DEFAULT_TTL_HOURS)
    except Exception:
        ttl_hours = _DEFAULT_TTL_HOURS
    ttl_hours = max(1, ttl_hours)
    cutoff = pd.Timestamp.now(tz=timezone.utc).tz_convert(None) - pd.Timedelta(hours=ttl_hours)

    stale_symbols = [
        sym for sym in symbols
        if sym not in edgar_covered
        and ((sym not in latest_map) or pd.isna(latest_map[sym]) or latest_map[sym] < cutoff)
    ]

    cache_only = str(
        os.getenv(
            "FUNDAMENTAL_CACHE_ONLY",
            os.getenv("APEX_FUNDAMENTAL_CACHE_ONLY", "0"),
        )
        or "0"
    ).strip().lower() in {"1", "true", "yes"}
    disable_network_fetch = str(
        os.getenv(
            "FUNDAMENTAL_DISABLE_NETWORK_FETCH",
            os.getenv("APEX_FUNDAMENTAL_DISABLE_NETWORK_FETCH", "0"),
        )
        or "0"
    ).strip().lower() in {"1", "true", "yes"}

    if stale_symbols and not cache_only and not disable_network_fetch:
        snap_df = _fetch_snapshots(stale_symbols)
        if not snap_df.empty:
            if cache_df.empty:
                cache_df = snap_df.copy()
            else:
                cache_df = pd.concat([cache_df, snap_df], ignore_index=True)
            cache_df = cache_df.dropna(subset=["symbol", "report_date"])
            cache_df = cache_df.sort_values(["symbol", "report_date", "fetched_at"])
            # Keep latest snapshot for each symbol + report period.
            cache_df = cache_df.drop_duplicates(subset=["symbol", "report_date"], keep="last")
            _save_cache(cache_df)

    out: Dict[str, pd.DataFrame] = {}
    grouped = {}
    if not cache_df.empty:
        subset = cache_df[cache_df["symbol"].isin(symbols)].copy()
        if not subset.empty:
            subset = subset.sort_values(["symbol", "report_date", "fetched_at"])
            subset = subset.drop_duplicates(subset=["symbol", "report_date"], keep="last")
            grouped = {sym: grp for sym, grp in subset.groupby("symbol")}

    for sym in symbols:
        edf = edgar_data.get(sym)
        if edf is not None and not edf.empty:
            out[sym] = edf
            continue

        grp = grouped.get(sym)
        if grp is None or grp.empty:
            out[sym] = _empty_symbol_frame()
            continue

        frame = grp.copy()
        if "available_date" in frame.columns:
            frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
        else:
            frame["available_date"] = pd.NaT
        if "report_date" in frame.columns:
            frame["report_date"] = pd.to_datetime(frame["report_date"], errors="coerce")
        else:
            frame["report_date"] = pd.NaT
        if _allow_estimated_available_date():
            lag_days = _fundamental_release_lag_days()
            fallback_avail = frame["report_date"] + pd.Timedelta(days=lag_days)
            frame["available_date"] = frame["available_date"].where(frame["available_date"].notna(), fallback_avail)
        frame = frame.dropna(subset=["available_date"]).sort_values(["available_date", "report_date", "fetched_at"])
        if frame.empty:
            out[sym] = _empty_symbol_frame()
            continue
        frame = frame.drop_duplicates(subset=["available_date"], keep="last")
        frame = frame.set_index("available_date")[FUNDAMENTAL_METRIC_COLUMNS].sort_index()
        for col in FUNDAMENTAL_METRIC_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        out[sym] = frame
    return out
