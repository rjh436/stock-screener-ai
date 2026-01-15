#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Tuple


ENGINE_REL_PATH = os.path.join("execution", "engine.py")
SHARED_REL_PATH = os.path.join("execution", "shared_logic.py")
CONFIG_REL_PATH = os.path.join("config", "generated_strategies.json")
TARGET_STRATEGY_NAME = "Apex Kinetic VCP (Strategy B)"


NEW_CALCULATE_SCORE = textwrap.dedent(
    """
    def calculate_backtest_quality_score(
        row_or_rsi2: Any,
        strategy_name: str = "",
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> float:
        \"\"\"
        Bifurcated ranking engine:
        - Breakout/VCP/Kinetic: high RSI, tight BB width, SMA10 surfing.
        - Wealth/Mean Reversion: low RSI dip buying.
        \"\"\"
        if weights is None or not isinstance(weights, dict):
            weights = DEFAULT_SCORING_WEIGHTS

        merged = dict(DEFAULT_SCORING_WEIGHTS)
        merged.update(weights)

        rsi_factor = float(merged.get("rsi_factor", 0.0) or 0.0)
        vcp_bonus = float(merged.get("vcp_bonus", merged.get("sniper_bonus", 0.0)) or 0.0)
        trend_bonus = float(merged.get("trend_bonus", 0.0) or 0.0)

        is_row = isinstance(row_or_rsi2, (pd.Series, dict))
        row = row_or_rsi2 if is_row else None

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

        strat_key = str(strategy_name or "").lower()
        is_breakout = any(token in strat_key for token in ("breakout", "vcp", "kinetic"))

        score = 0.0
        if is_breakout:
            rsi14 = _as_float(_get_val("rsi14", 50.0), 50.0)
            if rsi14 > 50.0:
                score += rsi14 * rsi_factor

            bb_width = _as_float(_get_val("bb_width", 1.0), 1.0)
            if bb_width < 0.15:
                score += vcp_bonus

            close_px = _as_float(_get_val("close", 0.0), 0.0)
            sma10 = _as_float(_get_val("sma10", 0.0), 0.0)
            if close_px > 0.0 and sma10 > 0.0 and close_px > sma10:
                score += trend_bonus
        else:
            if is_row:
                rsi2 = _as_float(_get_val("rsi2", _get_val("rsi14", 50.0)), 50.0)
            else:
                rsi2 = _as_float(row_or_rsi2, 50.0)
            score += (100.0 - rsi2) * rsi_factor

        return max(0.0, float(score))
    """
).lstrip("\n")


SMA_SURFING_BLOCK = textwrap.dedent(
    """
    # SMA surfing override (profit-protect for runners)
    if pnl_pct > 0 and days_held > 3:
        sma10 = state.get("sma10")
        if sma10 is None:
            row_last = state.get("row_last")
            if row_last is not None:
                sma10 = row_last.get("sma10")
        if sma10 is not None and np_isfinite(sma10) and close_px < float(sma10):
            return True, effective_stop, None
    """
).strip("\n")


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


def _replace_function_block(text: str, func_name: str, new_block: str) -> tuple[str, bool]:
    pattern = re.compile(rf"^def {re.escape(func_name)}\s*\(", re.M)
    match = pattern.search(text)
    if not match:
        return text, False

    start = match.start()
    tail = text[match.end():]
    next_match = re.search(r"^(def|class)\s+", tail, re.M)
    end = match.end() + (next_match.start() if next_match else len(tail))

    before = text[:start]
    after = text[end:]
    after = after.lstrip("\n")
    return before + new_block + "\n\n" + after, True


def _maybe_remove_score_candidate(text: str) -> tuple[str, bool]:
    occurrences = len(re.findall(r"_score_candidate\\b", text))
    if occurrences > 1:
        return text, False

    pattern = re.compile(r"^def _score_candidate\s*\(.*?\):\n", re.M)
    match = pattern.search(text)
    if not match:
        return text, False

    start = match.start()
    tail = text[match.end():]
    next_match = re.search(r"^(def|class)\s+", tail, re.M)
    end = match.end() + (next_match.start() if next_match else len(tail))

    before = text[:start]
    after = text[end:]
    after = after.lstrip("\n")
    return before + after, True


def patch_engine(engine_path: str) -> Tuple[bool, bool]:
    with open(engine_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    updated, replaced = _replace_function_block(
        original,
        "calculate_backtest_quality_score",
        NEW_CALCULATE_SCORE,
    )
    if not replaced:
        raise RuntimeError("calculate_backtest_quality_score not found in engine.py")

    updated, removed = _maybe_remove_score_candidate(updated)
    if updated != original:
        with open(engine_path, "w", encoding="utf-8") as handle:
            handle.write(updated)
        return True, removed
    return False, removed


def _insert_sma_surfing(text: str) -> tuple[str, bool]:
    if "SMA surfing override" in text:
        return text, False

    marker = re.compile(r"^(\s*)# 4\. PROFIT TARGETS", re.M)
    match = marker.search(text)
    if not match:
        return text, False

    indent = match.group(1)
    block = textwrap.indent(SMA_SURFING_BLOCK, indent) + "\n\n"
    insert_at = match.start()
    return text[:insert_at] + block + text[insert_at:], True


def patch_shared_logic(shared_path: str) -> bool:
    with open(shared_path, "r", encoding="utf-8") as handle:
        original = handle.read()

    updated, inserted = _insert_sma_surfing(original)
    if not inserted:
        return False

    with open(shared_path, "w", encoding="utf-8") as handle:
        handle.write(updated)
    return True


def _update_entry_rules(entry_rules: List[Dict[str, object]]) -> List[Dict[str, object]]:
    found = False
    for rule in entry_rules:
        if not isinstance(rule, dict):
            continue
        col = str(rule.get("col") or "").lower()
        op = str(rule.get("op") or "")
        if col == "rsi14" and op in (">", ">="):
            rule["val"] = 50
            found = True
    if not found:
        entry_rules.append({"col": "rsi14", "op": ">", "val": 50})
    return entry_rules


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
        entry_rules = strat.get("entry_rules")
        if not isinstance(entry_rules, list):
            entry_rules = []
        strat["entry_rules"] = _update_entry_rules(entry_rules)
        strat["stop_loss_atr"] = 1.0
        strat["partial_profit_target"] = 0.0
        strat["regime_filter"] = True
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
        shared_path = _find_file(SHARED_REL_PATH)
        config_path = _find_file(CONFIG_REL_PATH)
    except Exception as exc:
        print(f"Path detection failed: {exc}", file=sys.stderr)
        return 1

    try:
        engine_updated, score_removed = patch_engine(engine_path)
        shared_updated = patch_shared_logic(shared_path)
        config_updated = patch_config(config_path)
    except Exception as exc:
        print(f"Update failed: {exc}", file=sys.stderr)
        return 1

    print(f"engine.py updated: {engine_updated} ({engine_path})")
    print(f"_score_candidate removed: {score_removed}")
    print(f"shared_logic.py updated: {shared_updated} ({shared_path})")
    print(f"generated_strategies.json updated: {config_updated} ({config_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
