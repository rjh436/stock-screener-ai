#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional


ENGINE_REL_PATH = os.path.join("execution", "engine.py")
CONFIG_REL_PATH = os.path.join("config", "generated_strategies.json")
TARGET_STRATEGY_NAME = "Apex Kinetic VCP (Strategy B)"


NEW_CALCULATE_ROW_SCORE = textwrap.dedent(
    """
    def _calculate_row_score(
        row_or_rsi2: Any,
        strategy_name: str = "",
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> float:
        # --- DUAL-CORE RANKING ENGINE ---
        # Breakout/Momentum: high RSI + VCP + NATR fuel.
        # Wealth/Mean Reversion: low RSI dip buying.
        if weights is None or not isinstance(weights, dict):
            weights = DEFAULT_SCORING_WEIGHTS

        merged = dict(DEFAULT_SCORING_WEIGHTS)
        merged.update(weights)

        rsi_factor = float(merged.get("rsi_factor", 0.0) or 0.0)
        vcp_bonus = float(merged.get("vcp_bonus", merged.get("sniper_bonus", 0.0)) or 0.0)
        vol_bonus = float(merged.get("vol_bonus", 0.0) or 0.0)

        row = row_or_rsi2 if isinstance(row_or_rsi2, (pd.Series, dict)) else None

        def _get_val(key: str, default: float) -> float:
            if row is not None:
                try:
                    return row.get(key, default)
                except Exception:
                    pass
            return kwargs.get(key, default)

        def _as_float(value: float, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        scoring_type = str(kwargs.get("scoring_type", "") or "").lower()
        name = str(strategy_name or "").lower()
        if scoring_type:
            is_breakout = scoring_type in ("breakout", "momentum")
            is_wealth = scoring_type in ("wealth", "mean_reversion")
        else:
            is_breakout = any(key in name for key in ("breakout", "momentum", "vcp", "kinetic"))
            is_wealth = "wealth" in name
        if not is_breakout and not is_wealth:
            is_wealth = True

        score = 0.0

        if is_breakout:
            rsi14 = _as_float(_get_val("rsi14", 50.0), 50.0)
            if rsi14 > 50:
                score += rsi14 * rsi_factor
            else:
                score -= (50.0 - rsi14)

            bb_width = _as_float(_get_val("bb_width", 1.0), 1.0)
            if bb_width < _VCP_BB_WIDTH_THRESH:
                score += vcp_bonus

            natr = _as_float(_get_val("natr", 0.0), 0.0)
            if natr > 2.5:
                score += vol_bonus
        else:
            rsi2 = _as_float(_get_val("rsi2", _get_val("rsi14", 50.0)), 50.0)
            score += (100.0 - rsi2) * rsi_factor

        return max(0.0, float(score))
    """
).lstrip("\n")


def _iter_candidate_roots() -> List[str]:
    roots: List[str] = []
    for base in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        cur = os.path.abspath(base)
        for _ in range(6):
            if cur not in roots:
                roots.append(cur)
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    return roots


def _find_file(rel_path: str) -> str:
    filename = os.path.basename(rel_path)
    candidates: List[str] = []

    for root in _iter_candidate_roots():
        direct = os.path.join(root, rel_path)
        if os.path.isfile(direct):
            return direct

    skip_dirs = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
    }
    for root in _iter_candidate_roots():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            if filename in filenames:
                path = os.path.join(dirpath, filename)
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(f"Could not locate {rel_path} from {os.getcwd()}")

    suffix = rel_path.replace(os.sep, "/")
    suffix_matches = [p for p in candidates if p.replace(os.sep, "/").endswith(suffix)]
    matches = suffix_matches or candidates
    matches.sort(key=lambda p: (len(p), p))
    return matches[0]


def _upsert_calculate_row_score(text: str) -> tuple[str, str]:
    pattern = re.compile(r"^def _calculate_row_score\s*\(.*?\):\n", re.M)
    match = pattern.search(text)
    if match:
        start = match.start()
        tail = text[match.end():]
        next_match = re.search(r"^(def|class)\s+", tail, re.M)
        end = match.end() + (next_match.start() if next_match else len(tail))

        before = text[:start]
        after = text[end:]
        after = after.lstrip("\n")
        return before + NEW_CALCULATE_ROW_SCORE + "\n\n" + after, "replaced"

    insert_pattern = re.compile(r"^def calculate_backtest_quality_score\s*\(", re.M)
    insert_match = insert_pattern.search(text)
    if not insert_match:
        return text + "\n\n" + NEW_CALCULATE_ROW_SCORE + "\n", "appended"

    insert_at = insert_match.start()
    before = text[:insert_at]
    after = text[insert_at:]
    after = after.lstrip("\n")
    return before + NEW_CALCULATE_ROW_SCORE + "\n\n" + after, "inserted"


def patch_engine(engine_path: str) -> tuple[bool, str]:
    with open(engine_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    updated, mode = _upsert_calculate_row_score(original)
    if updated == original:
        return False, mode

    with open(engine_path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True, mode


def patch_config(config_path: str) -> bool:
    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("generated_strategies.json must contain a list")

    found = False
    for strat in data:
        if not isinstance(strat, dict):
            continue
        if strat.get("name") != TARGET_STRATEGY_NAME:
            continue
        found = True
        strat["scoring_type"] = "breakout"
        strat["regime_filter"] = True
        strat["scoring_weights"] = {
            "rsi_factor": 2.0,
            "vcp_bonus": 100.0,
            "trend_bonus": 50.0,
        }
        strat["stop_loss_atr"] = 1.0
        break

    if not found:
        raise ValueError(f"Strategy not found: {TARGET_STRATEGY_NAME}")

    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    return True


def main() -> int:
    try:
        engine_path = _find_file(ENGINE_REL_PATH)
        config_path = _find_file(CONFIG_REL_PATH)
    except Exception as exc:
        print(f"Path detection failed: {exc}", file=sys.stderr)
        return 1

    try:
        engine_changed, engine_mode = patch_engine(engine_path)
        config_changed = patch_config(config_path)
    except Exception as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1

    print(f"engine.py updated: {engine_changed} ({engine_path}) mode={engine_mode}")
    print(f"generated_strategies.json updated: {config_changed} ({config_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
