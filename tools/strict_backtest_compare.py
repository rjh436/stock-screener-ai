import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.loader import fetch_data_pack
from data.universe import (
    build_russell3000_membership_by_day,
    get_universe_symbols_pit_window_with_meta,
)
from execution.engine import prepare_backtest_data, run_backtest
from strategies.strategy_loader import load_strategies


def _load_strategy_config(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict config at {path}")
    return payload


def _load_prepared_cache(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "prepared" in payload:
        payload = payload["prepared"]
    return payload


def _save_prepared_cache(path: Path, prepared) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump({"prepared": prepared}, f, protocol=pickle.HIGHEST_PROTOCOL)


def _write_out(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _prepared_bounds(prepared) -> tuple[str | None, str | None]:
    all_dates = getattr(prepared, "all_dates", None)
    if all_dates is None or len(all_dates) == 0:
        return None, None
    return str(all_dates[0]), str(all_dates[-1])


def _prepared_covers_window(prepared, start_date: str, end_date: str) -> bool:
    all_dates = getattr(prepared, "all_dates", None)
    if all_dates is None or len(all_dates) == 0:
        return False
    try:
        loaded_start = pd.Timestamp(all_dates[0]).tz_localize(None)
        loaded_end = pd.Timestamp(all_dates[-1]).tz_localize(None)
        need_start = pd.Timestamp(start_date).tz_localize(None)
        need_end = pd.Timestamp(end_date).tz_localize(None)
    except Exception:
        return False
    return loaded_start <= need_start and loaded_end >= need_end


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare one or more strategy configs on a strict PIT-style cached backtest path."
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--universe", default="RUSSELL3000")
    parser.add_argument("--days", type=int, default=4200)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--prepared-cache",
        default="data/cache_indicators.pkl",
        help="Optional PreparedBacktestData pickle to reuse instead of rebuilding.",
    )
    parser.add_argument(
        "--save-prepared-cache",
        default=None,
        help="Optional path to persist a rebuilt prepared cache for later reuse.",
    )
    parser.add_argument(
        "--rebuild-prepared",
        action="store_true",
        help="Ignore any prepared-cache file and rebuild from raw symbol history.",
    )
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    out_path = Path(args.out)
    symbols, source = get_universe_symbols_pit_window_with_meta(
        args.universe,
        args.start_date,
        args.end_date,
    )
    market = fetch_data_pack(
        ["SPY", "$VIX", "VIX"],
        days=args.days,
        backtest_mode=True,
        max_workers=3,
    ) or {}
    spy = market.get("SPY")
    vix = market.get("$VIX")
    if vix is None or getattr(vix, "empty", False):
        vix = market.get("VIX")
    global_data = {"SPY": spy, "VIX": vix}

    prepared_cache_path = Path(args.prepared_cache)
    used_prepared_cache = False
    if not args.rebuild_prepared and prepared_cache_path.exists():
        prepared = _load_prepared_cache(prepared_cache_path)
        if _prepared_covers_window(prepared, args.start_date, args.end_date):
            used_prepared_cache = True
        else:
            prepared = None
    else:
        prepared = None
    if prepared is None:
        data = fetch_data_pack(
            symbols,
            days=args.days,
            backtest_mode=True,
            max_workers=args.max_workers,
        ) or {}
        prepared = prepare_backtest_data(
            data,
            symbol_universe=symbols,
            start_date=args.start_date,
            global_data=global_data,
        )
        save_cache_path = args.save_prepared_cache
        if save_cache_path:
            _save_prepared_cache(Path(save_cache_path), prepared)

    rows: list[dict[str, Any]] = []
    loaded_start, loaded_end = _prepared_bounds(prepared)
    out = {
        "start_date": args.start_date,
        "end_date": args.end_date,
        "membership_source": source,
        "requested_symbol_count": len(symbols),
        "loaded_symbol_count": len(getattr(prepared, "enriched", {}) or {}),
        "loaded_start": loaded_start,
        "loaded_end": loaded_end,
        "prepared_cache_path": str(prepared_cache_path) if used_prepared_cache else None,
        "used_prepared_cache": bool(used_prepared_cache),
        "rows": rows,
    }
    _write_out(out_path, out)

    membership_by_day = None
    membership_by_day_source = None
    require_pit_membership = False
    if str(args.universe or "").upper() == "RUSSELL3000":
        prepared_dates = getattr(prepared, "all_dates", None)
        if prepared_dates is None:
            prepared_dates = []
        else:
            prepared_dates = list(prepared_dates)
        membership_by_day, membership_by_day_source = build_russell3000_membership_by_day(
            prepared_dates,
            allow_missing_days=False,
        )
        require_pit_membership = bool(membership_by_day)
        out["membership_by_day_source"] = membership_by_day_source
        out["require_pit_membership"] = require_pit_membership
        _write_out(out_path, out)

    for config_path in args.configs:
        cfg = _load_strategy_config(config_path)
        strategy = load_strategies([cfg])[0]
        result = run_backtest(
            [strategy],
            {},
            start_cash=100000.0,
            start_date=args.start_date,
            end_date=args.end_date,
            global_data=global_data,
            pre_calculated_data=prepared,
            universe_membership_by_day=membership_by_day,
            require_pit_membership=require_pit_membership,
        )
        rows.append(
            {
                "config_path": config_path,
                "strategy_name": result.get("strategy_name") or result.get("strategy"),
                "cagr_pct": float(result.get("cagr", 0.0) or 0.0) * 100.0,
                "max_dd_pct": abs(float(result.get("max_drawdown_pct", 0.0) or 0.0)) * 100.0,
                "total_trades": int(result.get("total_trades", 0) or 0),
                "hit_rate_pct": float(result.get("hit_rate", 0.0) or 0.0),
                "final_value": float(result.get("final_value", 0.0) or 0.0),
                "first_trade_date": result.get("first_trade_date"),
            }
        )
        _write_out(out_path, out)
        print(json.dumps(rows[-1], indent=2), flush=True)

    out["completed"] = True
    _write_out(out_path, out)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
