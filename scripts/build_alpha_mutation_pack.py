#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from optimization.walkforward import enforce_cash_only


DEFAULT_BASE_CONFIGS = [
    "config/superperformance_alpha_b4.json",
    "config/superperformance_alpha_b9.json",
    "config/superperformance_alpha_b17.json",
]


def _mutation_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "a_regime_defense": {
            "max_total_exposure_pct_bear": 0.10,
            "bear_max_positions": 1,
            "stop_loss_atr_bear": 0.50,
        },
        "b_deconcentrated": {
            "max_total_exposure_pct_bear": 0.10,
            "bear_max_positions": 1,
            "stop_loss_atr_bear": 0.50,
            "risk_per_trade": 0.04,
            "max_pos_size_pct": 0.20,
            "max_positions": 6,
        },
        "c_moderate": {
            "max_total_exposure_pct_bear": 0.15,
            "bear_max_positions": 1,
            "stop_loss_atr_bear": 0.50,
            "risk_per_trade": 0.06,
            "max_pos_size_pct": 0.25,
            "max_positions": 5,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cash-only mutation pack for alpha-family Superperformance configs."
    )
    parser.add_argument("--config", action="append", default=[], help="Base config path. Repeatable.")
    parser.add_argument(
        "--frontier-json",
        default="",
        help="Optional frontier results JSON (for example tmp/alpha_neighborhood_latest.json).",
    )
    parser.add_argument(
        "--frontier-top-n",
        type=int,
        default=3,
        help="When --frontier-json is provided, use the top N rows from its rows array.",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp/alpha_cash_only_mutations",
        help="Directory where mutated configs and manifest will be written.",
    )
    return parser.parse_args()


def _base_configs(values: list[str]) -> list[Path]:
    configs = values or list(DEFAULT_BASE_CONFIGS)
    return [(ROOT / rel_path).resolve() for rel_path in configs]


def _frontier_bases(frontier_json: str, top_n: int) -> list[tuple[str, Dict[str, Any], str]]:
    path = (ROOT / frontier_json).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    out: list[tuple[str, Dict[str, Any], str]] = []
    for idx, row in enumerate(rows[: max(0, int(top_n))], start=1):
        cfg = row.get("config", {}) if isinstance(row, dict) else {}
        if not isinstance(cfg, dict) or not cfg:
            continue
        label = str(row.get("name", cfg.get("name", f"frontier_{idx}")) or f"frontier_{idx}")
        out.append((label, cfg, f"{path.relative_to(ROOT)}#{idx}"))
    return out


def _base_label(path: Path, cfg: Dict[str, Any]) -> str:
    raw = str(cfg.get("name", "") or "").strip()
    if raw:
        return raw.lower().replace(" ", "_")
    return path.stem.lower()


def _apply_mutation(base_cfg: Dict[str, Any], mutation_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.update(copy.deepcopy(overrides))
    if "max_pos_size_pct" in overrides:
        pos_size = float(overrides["max_pos_size_pct"])
        cfg["vcp_max_pos_size_pct"] = pos_size
        cfg["ep_max_pos_size_pct"] = pos_size
    cfg["name"] = f"{cfg.get('name', 'Alpha Variant')} [{mutation_name}]"
    return enforce_cash_only(cfg)


def main() -> None:
    args = _parse_args()
    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    mutation_specs = _mutation_specs()
    generated_from: list[str] = []

    base_entries: list[tuple[str, Dict[str, Any], str]] = []
    if str(args.frontier_json or "").strip():
        base_entries.extend(_frontier_bases(str(args.frontier_json), int(args.frontier_top_n)))
    else:
        for config_path in _base_configs(args.config):
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
            base_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            base_entries.append((str(base_cfg.get("name", config_path.stem)), base_cfg, str(config_path.relative_to(ROOT))))

    for idx, (base_name, base_cfg, source_ref) in enumerate(base_entries, start=1):
        generated_from.append(source_ref)
        base_slug = _base_label(Path(source_ref.split("#", 1)[0]), base_cfg)
        base_slug = f"{idx:02d}_{base_slug}"
        for mutation_name, overrides in mutation_specs.items():
            variant_cfg = _apply_mutation(base_cfg, mutation_name, overrides)
            out_name = f"{base_slug}__{mutation_name}.json"
            out_path = out_dir / out_name
            out_path.write_text(json.dumps(variant_cfg, indent=2) + "\n", encoding="utf-8")
            manifest_rows.append(
                {
                    "base_name": base_name,
                    "base_config": source_ref,
                    "mutation": mutation_name,
                    "config_path": str(out_path.relative_to(ROOT)),
                    "name": variant_cfg.get("name", ""),
                    "overrides": overrides,
                    "cash_only": {
                        "allow_margin": bool(variant_cfg.get("allow_margin", False)),
                        "max_total_exposure_pct_bull": float(
                            variant_cfg.get("max_total_exposure_pct_bull", 1.0) or 1.0
                        ),
                        "max_total_exposure_pct_bear": float(
                            variant_cfg.get("max_total_exposure_pct_bear", 0.0) or 0.0
                        ),
                    },
                }
            )

    manifest = {
        "generated_from": generated_from,
        "count": len(manifest_rows),
        "rows": manifest_rows,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"saved {manifest_path}")


if __name__ == "__main__":
    main()
