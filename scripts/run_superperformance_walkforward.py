#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization.walkforward import (
    DEFAULT_WALKFORWARD_START,
    prepare_walkforward_context,
    promotion_walkforward_status,
    run_walkforward_report_with_context,
)


DEFAULT_CONFIG = ROOT / "config" / "superperformance_multi_sleeve_allocator_v1.json"


def _load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return cfg


def _parse_friction_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").split(","):
        text = part.strip()
        if not text:
            continue
        try:
            value = float(text)
        except Exception:
            continue
        if value >= 0.0:
            values.append(float(value))
    return values or [10.0]


def _resolve_config_paths(values: list[str]) -> list[Path]:
    raw = [str(v).strip() for v in values if str(v).strip()]
    if not raw:
        raw = [str(DEFAULT_CONFIG)]
    return [Path(item).expanduser().resolve() for item in raw]


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip()).strip("_") or "strategy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cash-only Russell 3000 walk-forward robustness for Superperformance config")
    parser.add_argument("--config", action="append", default=[], help="Path to strategy JSON config. Repeat for batch replay.")
    parser.add_argument("--start", default=DEFAULT_WALKFORWARD_START, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.now().date().isoformat(), help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--output", default="", help="Output JSON report path for a single config")
    parser.add_argument("--output-dir", default="", help="Directory for batch replay reports")
    parser.add_argument("--prepared-cache", default="", help="Optional PreparedBacktestData pickle to reuse instead of rebuilding the PIT universe")
    parser.add_argument("--universe-limit", type=int, default=0, help="Optional limit for quick local smoke tests")
    parser.add_argument(
        "--universe-limit-mode",
        choices=["sample", "first"],
        default="sample",
        help="When universe-limit is used: deterministic random sample (default) or first-N symbols.",
    )
    parser.add_argument(
        "--universe-sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic universe sampling when universe-limit-mode=sample.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=2.0, help="Default transaction cost bps")
    parser.add_argument(
        "--frictions",
        default="5,10,20,35",
        help="Comma-separated slippage bps pairs (entry=exit), e.g. '5,10,20'",
    )
    args = parser.parse_args()

    cfg_paths = _resolve_config_paths(list(args.config))
    for cfg_path in cfg_paths:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
    friction_values = _parse_friction_values(str(args.frictions))

    if len(cfg_paths) == 1:
        cfg_path = cfg_paths[0]
        cfg = _load_config(cfg_path)
        _, _, _, report = promotion_walkforward_status(
            cfg,
            start_date=str(args.start),
            end_date=str(args.end),
            transaction_cost_bps=float(args.transaction_cost_bps),
            friction_values=friction_values,
            universe_limit=int(args.universe_limit),
            universe_limit_mode=str(args.universe_limit_mode),
            universe_sample_seed=int(args.universe_sample_seed),
            prepared_cache_path=str(args.prepared_cache),
            config_path=str(cfg_path),
        )

        output_path = Path(args.output).expanduser().resolve() if args.output else (ROOT / "logs" / f"walkforward_superperformance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written: {output_path}")
        return

    configs = [_load_config(path) for path in cfg_paths]
    context = prepare_walkforward_context(
        start_date=str(args.start),
        end_date=str(args.end),
        universe_limit=int(args.universe_limit),
        universe_limit_mode=str(args.universe_limit_mode),
        universe_sample_seed=int(args.universe_sample_seed),
        prepared_cache_path=str(args.prepared_cache),
    )
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (ROOT / "logs" / f"walkforward_batch_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, (cfg_path, cfg) in enumerate(zip(cfg_paths, configs), start=1):
        print(f"[{idx}/{len(cfg_paths)}] Replaying {cfg_path}")
        report = run_walkforward_report_with_context(
            cfg,
            context,
            transaction_cost_bps=float(args.transaction_cost_bps),
            friction_values=friction_values,
            config_path=str(cfg_path),
        )
        report_path = output_dir / f"{_safe_name(cfg_path.stem)}.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        manifest.append({"config_path": str(cfg_path), "report_path": str(report_path)})
        print(f"Report written: {report_path}")
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"reports": manifest}, indent=2), encoding="utf-8")
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
