#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.custom_universe import list_cached_symbols, load_symbol_file
from data.loader import fetch_data_pack
from execution.engine import prepare_backtest_data, run_backtest
from scripts.run_factor_walkforward import _build_test_windows, _pct_dd, _safe_float, _stitch_test_windows_warm
from strategies.strategy_loader import load_strategies


DEFAULT_CONFIG = ROOT / "config" / "superperformance_custom_momo_universe_v1.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Superperformance on a custom quarterly-selected momentum universe.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--universe", default="cache_all")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbol-file", default="")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--days", type=int, default=3000)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config in {cfg_path}")
    return dict(cfg)


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [str(x).strip().upper() for x in str(args.symbols).split(",") if str(x).strip()]
    if args.symbol_file:
        return load_symbol_file(args.symbol_file)
    if str(args.universe).lower() == "cache_all":
        max_symbols = int(args.max_symbols) if int(args.max_symbols or 0) > 0 else None
        return list_cached_symbols(max_symbols=max_symbols)
    raise ValueError(f"Unsupported universe preset: {args.universe}")


def _normalize_ohlcv(frame: pd.DataFrame) -> Optional[pd.DataFrame]:
    if frame is None or frame.empty:
        return None
    out = frame.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    idx = pd.to_datetime(out.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = pd.DatetimeIndex(idx).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    if out.empty:
        return None
    return out


def _precompute_universe_metrics(data: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    metrics: Dict[str, pd.DataFrame] = {}
    for sym, raw in (data or {}).items():
        frame = _normalize_ohlcv(raw)
        if frame is None or len(frame) < 260:
            continue
        df = frame[[c for c in ("open", "high", "low", "close", "volume") if c in frame.columns]].copy()
        if not {"high", "low", "close", "volume"}.issubset(df.columns):
            continue
        vol_ma50 = df["volume"].rolling(50, min_periods=20).mean()
        adv50 = vol_ma50 * df["close"]
        with np.errstate(divide="ignore", invalid="ignore"):
            adr20 = (((df["high"] / df["low"]) - 1.0) * 100.0).rolling(20, min_periods=10).mean()
            ret63 = (df["close"] / df["close"].shift(63)) - 1.0
            ret126 = (df["close"] / df["close"].shift(126)) - 1.0
            prox252 = df["close"] / df["high"].rolling(252, min_periods=126).max()
        out = pd.DataFrame(
            {
                "close": df["close"],
                "adv50": adv50,
                "adr20": adr20,
                "ret63": ret63,
                "ret126": ret126,
                "prox252": prox252,
            },
            index=df.index,
        )
        metrics[str(sym).upper()] = out
    return metrics


def _selection_dates(start_date: str, end_date: str, every_n_months: int) -> List[pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    dates = []
    cur = start_ts
    step = max(1, int(every_n_months))
    while cur <= end_ts:
        dates.append(cur)
        cur = cur + pd.DateOffset(months=step)
    return dates


def _latest_row(frame: pd.DataFrame, asof: pd.Timestamp) -> Optional[pd.Series]:
    hist = frame.loc[frame.index <= asof]
    if hist.empty:
        return None
    return hist.iloc[-1]


def _score_candidates(rows: List[Tuple[str, pd.Series]], cfg: Mapping[str, Any]) -> List[str]:
    if not rows:
        return []
    score_cfg = cfg.get("custom_universe", {}) if isinstance(cfg.get("custom_universe"), Mapping) else {}
    max_names = int(score_cfg.get("max_names", 400) or 400)
    w_ret126 = float(score_cfg.get("ret126_weight", 0.55) or 0.55)
    w_ret63 = float(score_cfg.get("ret63_weight", 0.25) or 0.25)
    w_prox = float(score_cfg.get("proximity_52w_weight", 0.20) or 0.20)

    df = pd.DataFrame(
        {
            "symbol": [sym for sym, _ in rows],
            "ret126": [float(row.get("ret126", np.nan)) for _, row in rows],
            "ret63": [float(row.get("ret63", np.nan)) for _, row in rows],
            "prox252": [float(row.get("prox252", np.nan)) for _, row in rows],
        }
    )
    for col in ("ret126", "ret63", "prox252"):
        vals = pd.to_numeric(df[col], errors="coerce")
        finite_count = int(vals.notna().sum())
        if finite_count == 1:
            df[col + "_rank"] = np.where(vals.notna(), 1.0, np.nan)
        elif finite_count >= 2:
            df[col + "_rank"] = vals.rank(pct=True, method="average")
        else:
            df[col + "_rank"] = np.nan
    df["score"] = (w_ret126 * df["ret126_rank"]) + (w_ret63 * df["ret63_rank"]) + (w_prox * df["prox252_rank"])
    df = df.sort_values(["score", "ret126", "ret63"], ascending=[False, False, False]).dropna(subset=["score"])
    return [str(sym) for sym in df.head(max_names)["symbol"].tolist()]


def _build_selection_schedule(metrics: Mapping[str, pd.DataFrame], cfg: Mapping[str, Any], start_date: str, end_date: str) -> List[Tuple[pd.Timestamp, List[str]]]:
    score_cfg = cfg.get("custom_universe", {}) if isinstance(cfg.get("custom_universe"), Mapping) else {}
    months = int(score_cfg.get("rebalance_months", 3) or 3)
    min_price = float(score_cfg.get("min_price", 8.0) or 8.0)
    max_price = float(score_cfg.get("max_price", 250.0) or 250.0)
    min_adv50 = float(score_cfg.get("min_adv50", 2_000_000.0) or 2_000_000.0)
    max_adv50 = float(score_cfg.get("max_adv50", 150_000_000.0) or 150_000_000.0)
    min_adr20 = float(score_cfg.get("min_adr20_pct", 3.0) or 3.0)
    min_ret63 = float(score_cfg.get("min_ret63", 0.05) or 0.05)
    min_ret126 = float(score_cfg.get("min_ret126", 0.10) or 0.10)
    min_prox = float(score_cfg.get("min_proximity_52w", 0.75) or 0.75)

    selections: List[Tuple[pd.Timestamp, List[str]]] = []
    for asof in _selection_dates(start_date, end_date, months):
        candidates: List[Tuple[str, pd.Series]] = []
        for sym, frame in metrics.items():
            row = _latest_row(frame, asof)
            if row is None:
                continue
            close = _safe_float(row.get("close"), float("nan"))
            adv50 = _safe_float(row.get("adv50"), float("nan"))
            adr20 = _safe_float(row.get("adr20"), float("nan"))
            ret63 = _safe_float(row.get("ret63"), float("nan"))
            ret126 = _safe_float(row.get("ret126"), float("nan"))
            prox252 = _safe_float(row.get("prox252"), float("nan"))
            if not np.isfinite(close) or close < min_price or close > max_price:
                continue
            if not np.isfinite(adv50) or adv50 < min_adv50 or adv50 > max_adv50:
                continue
            if not np.isfinite(adr20) or adr20 < min_adr20:
                continue
            if not np.isfinite(ret63) or ret63 < min_ret63:
                continue
            if not np.isfinite(ret126) or ret126 < min_ret126:
                continue
            if not np.isfinite(prox252) or prox252 < min_prox:
                continue
            candidates.append((sym, row))
        selected = _score_candidates(candidates, cfg)
        selections.append((asof, selected))
    return selections


def _selection_union(selections: Sequence[Tuple[pd.Timestamp, Sequence[str]]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for _, syms in selections:
        for sym in syms:
            s = str(sym).upper()
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
    return sorted(out)


def _membership_by_day(all_dates: Sequence[pd.Timestamp], selections: Sequence[Tuple[pd.Timestamp, Sequence[str]]]) -> List[Optional[set[str]]]:
    if not selections:
        return [None for _ in all_dates]
    ordered = [(pd.Timestamp(dt).normalize(), {str(s).upper() for s in syms}) for dt, syms in selections]
    out: List[Optional[set[str]]] = []
    idx = 0
    current: Optional[set[str]] = None
    for day in pd.to_datetime(list(all_dates)).tz_localize(None).normalize():
        while idx < len(ordered) and ordered[idx][0] <= day:
            current = ordered[idx][1]
            idx += 1
        out.append(set(current) if current is not None else None)
    return out


def _result_payload(
    *,
    cfg: Mapping[str, Any],
    start_date: str,
    end_date: str,
    requested_symbols: Sequence[str],
    loaded_symbols: Sequence[str],
    selected_union: Sequence[str],
    selection_schedule: Sequence[Tuple[pd.Timestamp, Sequence[str]]],
    result: Mapping[str, Any],
    oos24: Mapping[str, Any],
    oos60: Mapping[str, Any],
) -> Dict[str, Any]:
    audit = result.get("audit_report") or {}
    schedule_sizes = [len(syms) for _, syms in selection_schedule]
    return {
        "strategy_name": str(cfg.get("name", "Superperformance") or "Superperformance"),
        "requested_symbols": int(len(requested_symbols)),
        "loaded_symbols": int(len(loaded_symbols)),
        "selected_union_symbols": int(len(selected_union)),
        "selection_rebalances": int(len(selection_schedule)),
        "avg_selected_names": float(np.mean(schedule_sizes)) if schedule_sizes else float("nan"),
        "max_selected_names": int(max(schedule_sizes)) if schedule_sizes else 0,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "full_cagr_pct": float(_safe_float(result.get("cagr"), 0.0) * 100.0),
        "full_max_dd_pct": float(_pct_dd(result.get("max_drawdown_pct", 0.0))),
        "final_value": float(_safe_float(result.get("final_value"), 0.0)),
        "total_trades": int(result.get("total_trades", 0) or 0),
        "avg_annual_turnover_pct": float(_safe_float(result.get("avg_annual_turnover_pct"), float("nan"))),
        "oos_24_12_cagr_pct_warm": float(_safe_float(oos24.get("stitched_cagr_pct"), float("nan"))),
        "oos_24_12_max_dd_pct_warm": float(_safe_float(oos24.get("oos_max_dd_pct"), float("nan"))),
        "oos_60_12_cagr_pct_warm": float(_safe_float(oos60.get("stitched_cagr_pct"), float("nan"))),
        "oos_60_12_max_dd_pct_warm": float(_safe_float(oos60.get("oos_max_dd_pct"), float("nan"))),
        "audit_report": {
            "same_day_open_entries": int(audit.get("same_day_open_entries", 0) or 0),
            "max_gross_exposure_pct": float(_safe_float(audit.get("max_gross_exposure_pct"), 0.0)),
        },
    }


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    requested_symbols = _resolve_symbols(args)
    if not requested_symbols:
        raise RuntimeError("No symbols resolved")

    print(f"superperformance-custom run: requested_symbols={len(requested_symbols)}")
    data = fetch_data_pack(requested_symbols, days=int(args.days), backtest_mode=True) or {}
    loaded_symbols = sorted(data.keys())
    print(f"loaded_symbols={len(loaded_symbols)}")
    global_data = fetch_data_pack(["SPY", "VIX", "$VIX"], days=int(args.days), backtest_mode=True) or {}

    metrics = _precompute_universe_metrics(data)
    print(f"metric_ready_symbols={len(metrics)}")
    schedule = _build_selection_schedule(metrics, cfg, str(args.start_date), str(args.end_date))
    selected_union = _selection_union(schedule)
    print(f"selected_union_symbols={len(selected_union)}")
    if not selected_union:
        raise RuntimeError("Custom universe selection produced zero symbols")

    subset = {sym: data[sym] for sym in selected_union if sym in data}
    prepared = prepare_backtest_data(subset, selected_union, start_date=str(args.start_date), global_data=global_data)
    membership = _membership_by_day(prepared.all_dates, schedule)
    print(f"prepared_symbols={len(prepared.enriched)} prepared_dates={len(prepared.all_dates)}")

    strategies = load_strategies([cfg])
    if not strategies:
        raise RuntimeError("Failed to load strategy")
    result = run_backtest(
        strategies,
        prepared,
        start_cash=100000.0,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        global_data=global_data,
        universe_membership_by_day=membership,
    )
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        result = {}

    windows24 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=24, test_months=12)
    windows60 = _build_test_windows(str(args.start_date), str(args.end_date), train_months=60, test_months=12)
    oos24 = _stitch_test_windows_warm(full_run=result, windows=windows24, test_months=12)
    oos60 = _stitch_test_windows_warm(full_run=result, windows=windows60, test_months=12)

    payload = _result_payload(
        cfg=cfg,
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        requested_symbols=requested_symbols,
        loaded_symbols=loaded_symbols,
        selected_union=selected_union,
        selection_schedule=schedule,
        result=result,
        oos24=oos24,
        oos60=oos60,
    )
    print(json.dumps(payload, indent=2))
    if args.out:
        out_path = Path(str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
