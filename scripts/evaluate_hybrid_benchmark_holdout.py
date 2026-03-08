#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_etf_rotation_holdout import _evaluate_config as _evaluate_etf_config
from scripts.evaluate_etf_rotation_holdout import _load_config as _load_etf_config
from scripts.evaluate_etf_rotation_holdout import _requested_symbols
from scripts.evaluate_smid_pullback_holdout import (
    build_smid_pullback_context,
    evaluate_smid_pullback_configs_on_context,
)
from scripts.run_etf_rotation_walkforward import _download_yfinance_pack


DEFAULT_ETF_CONFIG = ROOT / "config" / "etf_rotation_growth_core5_residual_defensive_calmar_v1.json"
DEFAULT_STOCK_CONFIG = ROOT / "config" / "smid_pullback_r3000_tc4_tb003_v1.json"
DEFAULT_WEIGHTS = [0.25, 0.4, 0.5, 0.6, 0.75]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ETF/stock hybrid benchmark holdout blends.")
    parser.add_argument("--etf-config", default=str(DEFAULT_ETF_CONFIG))
    parser.add_argument("--stock-config", default=str(DEFAULT_STOCK_CONFIG))
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--train-end-date", default="2020-12-31")
    parser.add_argument("--holdout-start-date", default="2021-01-01")
    parser.add_argument("--holdout-end-date", default="2025-12-31")
    parser.add_argument("--etf-days", type=int, default=4000)
    parser.add_argument("--stock-days", type=int, default=3200)
    parser.add_argument(
        "--etf-weights",
        default=",".join(str(x) for x in DEFAULT_WEIGHTS),
        help="Comma-separated ETF sleeve weights. Stock weight is 1 - ETF weight.",
    )
    parser.add_argument("--out", default="")
    return parser.parse_args()


def _resolve_weights(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    out = [x for x in vals if 0.0 <= x <= 1.0]
    if not out:
        raise ValueError("No valid ETF weights provided.")
    return out


def _equity_curve_to_series(run: Mapping[str, Any]) -> pd.Series:
    curve = list(run.get("equity_curve") or [])
    if not curve:
        return pd.Series(dtype="float64")
    idx = pd.to_datetime([row["Date"] for row in curve], errors="coerce")
    vals = pd.to_numeric([row["Equity"] for row in curve], errors="coerce")
    ser = pd.Series(vals, index=idx, dtype="float64").dropna()
    ser = ser[~ser.index.duplicated(keep="last")].sort_index()
    return ser


def _series_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype="float64")
    return equity.pct_change().fillna(0.0).astype("float64")


def _metrics_from_equity(equity: pd.Series) -> Dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {
            "cagr_pct": float("nan"),
            "max_dd_pct": float("nan"),
            "calmar": float("nan"),
            "final_value": float("nan"),
        }
    rets = equity.pct_change().fillna(0.0)
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    max_dd_pct = abs(float(dd.min()) * 100.0)
    total_days = int((equity.index[-1] - equity.index[0]).days)
    if total_days <= 0 or float(equity.iloc[0]) <= 0.0:
        cagr_pct = float("nan")
    else:
        years = float(total_days) / 365.25
        cagr_pct = (float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0
        cagr_pct *= 100.0
    calmar = float(cagr_pct / max_dd_pct) if max_dd_pct > 0 and not math.isnan(cagr_pct) else float("nan")
    return {
        "cagr_pct": float(cagr_pct),
        "max_dd_pct": float(max_dd_pct),
        "calmar": float(calmar),
        "final_value": float(equity.iloc[-1]),
        "days": int(len(equity)),
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
        "avg_daily_return_pct": float(rets.mean() * 100.0),
    }


def _blend_equity_series(etf_eq: pd.Series, stock_eq: pd.Series, *, etf_weight: float) -> pd.Series:
    etf_ret = _series_returns(etf_eq)
    stock_ret = _series_returns(stock_eq)
    idx = etf_ret.index.union(stock_ret.index).sort_values()
    etf_ret = etf_ret.reindex(idx).fillna(0.0)
    stock_ret = stock_ret.reindex(idx).fillna(0.0)
    stock_weight = 1.0 - float(etf_weight)
    blended_ret = (float(etf_weight) * etf_ret) + (stock_weight * stock_ret)
    equity = (1.0 + blended_ret).cumprod() * 100000.0
    return equity


def _download_etf_data_with_retry(symbols: Iterable[str], *, start_date: str, end_date: str, max_attempts: int = 4) -> Dict[str, Any]:
    requested = [str(sym).strip().upper() for sym in symbols if str(sym).strip()]
    merged: Dict[str, Any] = {}
    missing = list(requested)
    for attempt in range(1, max_attempts + 1):
        if not missing:
            break
        chunk = _download_yfinance_pack(missing, start_date=start_date, end_date=end_date)
        merged.update(chunk)
        missing = [sym for sym in requested if sym not in merged]
        if missing and attempt < max_attempts:
            time.sleep(float(attempt))
    if missing:
        raise RuntimeError(f"Missing ETF benchmark price data: {missing}")
    return merged


def main() -> None:
    args = _parse_args()
    etf_weight_grid = _resolve_weights(args.etf_weights)

    etf_cfg = _load_etf_config(Path(args.etf_config).resolve())
    etf_cfg["_config_path"] = str(Path(args.etf_config).resolve())
    etf_requested = _requested_symbols([etf_cfg])
    etf_data = _download_etf_data_with_retry(
        etf_requested,
        start_date=str(args.start_date),
        end_date=str(args.holdout_end_date),
    )
    etf_result = _evaluate_etf_config(
        etf_cfg,
        etf_data,
        start_date=str(args.start_date),
        train_end_date=str(args.train_end_date),
        holdout_start_date=str(args.holdout_start_date),
        holdout_end_date=str(args.holdout_end_date),
        include_runs=True,
    )

    stock_context = build_smid_pullback_context(
        universe="RUSSELL3000",
        start_date=str(args.start_date),
        end_date=str(args.holdout_end_date),
        days=int(args.stock_days),
        config_paths=[str(Path(args.stock_config).resolve())],
    )
    stock_payload = evaluate_smid_pullback_configs_on_context(
        stock_context,
        config_paths=[str(Path(args.stock_config).resolve())],
        universe="RUSSELL3000",
        train_start_date=str(args.start_date),
        train_end_date=str(args.train_end_date),
        holdout_start_date=str(args.holdout_start_date),
        holdout_end_date=str(args.holdout_end_date),
        raise_on_coverage_fail=True,
        include_runs=True,
    )
    stock_result = dict(stock_payload["reports"][0])

    etf_train_eq = _equity_curve_to_series(etf_result["train_run"])
    etf_holdout_eq = _equity_curve_to_series(etf_result["holdout_run"])
    stock_train_eq = _equity_curve_to_series(stock_result["train_run"])
    stock_holdout_eq = _equity_curve_to_series(stock_result["holdout_run"])

    rows: List[Dict[str, Any]] = []
    for etf_weight in etf_weight_grid:
        holdout_eq = _blend_equity_series(etf_holdout_eq, stock_holdout_eq, etf_weight=etf_weight)
        train_eq = _blend_equity_series(etf_train_eq, stock_train_eq, etf_weight=etf_weight)
        rows.append(
            {
                "etf_weight": float(etf_weight),
                "stock_weight": float(1.0 - etf_weight),
                "train": _metrics_from_equity(train_eq),
                "holdout": _metrics_from_equity(holdout_eq),
            }
        )

    payload = {
        "start_date": str(args.start_date),
        "train_end_date": str(args.train_end_date),
        "holdout_start_date": str(args.holdout_start_date),
        "holdout_end_date": str(args.holdout_end_date),
        "etf_config": str(Path(args.etf_config).resolve()),
        "stock_config": str(Path(args.stock_config).resolve()),
        "etf": {
            "train": etf_result["train"],
            "holdout": etf_result["holdout"],
        },
        "stock": {
            "train": stock_result["train"],
            "holdout": stock_result["holdout"],
            "coverage_ratio": float(stock_result.get("coverage_ratio", float("nan"))),
            "nominal_price_proxy_symbols": int(stock_result.get("nominal_price_proxy_symbols", 0) or 0),
            "nominal_price_proxy_missing_symbols": int(stock_result.get("nominal_price_proxy_missing_symbols", 0) or 0),
        },
        "hybrids": rows,
    }
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
