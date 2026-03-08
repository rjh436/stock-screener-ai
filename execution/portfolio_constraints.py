from __future__ import annotations

from typing import Dict, Iterable, Mapping

import numpy as np


def _clean_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw_key, raw_val in dict(weights or {}).items():
        key = str(raw_key)
        try:
            val = float(raw_val)
        except Exception:
            continue
        if not np.isfinite(val) or val <= 0.0:
            continue
        out[key] = val
    total = float(sum(out.values()))
    if total <= 0:
        return {}
    if total > 1.0 + 1e-12:
        return {k: (v / total) for k, v in out.items()}
    return out


def _enforce_group_cap(weights: Mapping[str, float], group_map: Mapping[str, str], cap: float) -> Dict[str, float]:
    out = _clean_weights(weights)
    if not out:
        return {}

    group_cap = float(cap)
    if not np.isfinite(group_cap) or group_cap <= 0.0:
        return {}
    if group_cap >= 1.0:
        return out

    groups: Dict[str, list[str]] = {}
    for sym in out:
        grp = str(group_map.get(sym, f"__{sym}__"))
        groups.setdefault(grp, []).append(sym)

    for _ in range(12):
        group_totals = {g: float(sum(out.get(sym, 0.0) for sym in syms)) for g, syms in groups.items()}
        over = [g for g, total in group_totals.items() if total > (group_cap + 1e-12)]
        if not over:
            break

        released = 0.0
        for g in over:
            total = group_totals[g]
            if total <= 0:
                continue
            scale = group_cap / total
            for sym in groups[g]:
                old = out.get(sym, 0.0)
                new = old * scale
                out[sym] = new
                released += max(0.0, old - new)

        if released <= 0.0:
            continue

        group_totals = {g: float(sum(out.get(sym, 0.0) for sym in syms)) for g, syms in groups.items()}
        under = [g for g, total in group_totals.items() if total < (group_cap - 1e-12)]
        if not under:
            break

        headroom = {g: max(0.0, group_cap - group_totals[g]) for g in under}
        total_headroom = float(sum(headroom.values()))
        if total_headroom <= 0.0:
            break

        for g in under:
            alloc = released * (headroom[g] / total_headroom)
            syms = groups[g]
            base = float(sum(out.get(sym, 0.0) for sym in syms))
            if base > 0:
                for sym in syms:
                    out[sym] = out.get(sym, 0.0) + (alloc * (out.get(sym, 0.0) / base))
            else:
                each = alloc / float(len(syms))
                for sym in syms:
                    out[sym] = out.get(sym, 0.0) + each

    # Final strict clamp to remove tiny numeric violations.
    group_totals = {g: float(sum(out.get(sym, 0.0) for sym in syms)) for g, syms in groups.items()}
    for g, total in group_totals.items():
        if total > group_cap + 1e-10:
            scale = group_cap / total
            for sym in groups[g]:
                out[sym] = out.get(sym, 0.0) * scale

    total = float(sum(out.values()))
    if total <= 0:
        return {}
    if total > 1.0 + 1e-10:
        out = {k: (v / total) for k, v in out.items()}
    return out


def enforce_position_cap(weights: Mapping[str, float], position_cap: float) -> Dict[str, float]:
    out = _clean_weights(weights)
    if not out:
        return {}

    cap = float(position_cap)
    if not np.isfinite(cap) or cap <= 0.0:
        return {}
    cap = min(cap, 1.0)

    for _ in range(12):
        over = {k: v for k, v in out.items() if v > (cap + 1e-12)}
        if not over:
            break

        released = 0.0
        for sym, val in over.items():
            released += val - cap
            out[sym] = cap

        under = {k: v for k, v in out.items() if v < (cap - 1e-12)}
        if not under or released <= 0:
            break

        headroom = {k: max(0.0, cap - v) for k, v in under.items()}
        head_total = float(sum(headroom.values()))
        if head_total <= 0:
            break

        for sym in under:
            out[sym] += released * (headroom[sym] / head_total)

    total = float(sum(out.values()))
    if total <= 0:
        return {}
    if total > 1.0 + 1e-10:
        out = {k: (v / total) for k, v in out.items()}

    # Ensure strict cap after final normalization.
    for sym, val in list(out.items()):
        if val > cap:
            out[sym] = cap

    total = float(sum(out.values()))
    if total > 1.0 + 1e-10:
        out = {k: (v / total) for k, v in out.items()}
    return out


def enforce_sector_cap(weights: Mapping[str, float], sector_map: Mapping[str, str], sector_cap: float) -> Dict[str, float]:
    return _enforce_group_cap(weights, sector_map, sector_cap)


def enforce_industry_cap(weights: Mapping[str, float], industry_map: Mapping[str, str], industry_cap: float) -> Dict[str, float]:
    return _enforce_group_cap(weights, industry_map, industry_cap)


def enforce_turnover_budget(
    prev_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    turnover_budget: float,
    *,
    mode: str = "blend",
    priority_symbols: Iterable[str] | None = None,
    cleanup_weight_floor: float = 0.0,
) -> Dict[str, float]:
    def _clean_abs(weights: Mapping[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for raw_key, raw_val in dict(weights or {}).items():
            key = str(raw_key)
            try:
                val = float(raw_val)
            except Exception:
                continue
            if not np.isfinite(val) or val <= 0.0:
                continue
            out[key] = val
        total = float(sum(out.values()))
        if total <= 0.0:
            return {}
        # Preserve residual cash when total <= 1.0.
        if total > 1.0 + 1e-12:
            out = {k: (v / total) for k, v in out.items()}
        return out

    prev = _clean_abs(prev_weights)
    target = _clean_abs(target_weights)

    budget = float(turnover_budget)
    if not np.isfinite(budget) or budget <= 0.0:
        return prev

    keys = sorted(set(prev.keys()) | set(target.keys()))
    if not keys:
        return {}

    mode_name = str(mode or "blend").strip().lower()
    if mode_name in {"priority", "concentrated"}:
        current = dict(prev)
        priority_list = [str(sym) for sym in (priority_symbols or []) if str(sym)]
        priority_set = set(priority_list)
        cleanup_floor = float(cleanup_weight_floor or 0.0)

        def _purge_tiny(weights: Dict[str, float]) -> Dict[str, float]:
            return {k: float(v) for k, v in weights.items() if np.isfinite(v) and float(v) > 1e-12}

        # Stage 0: sell tiny obsolete names first. This improves concentration at low turnover cost.
        if np.isfinite(cleanup_floor) and cleanup_floor > 0.0:
            small_non_targets = sorted(
                (
                    (sym, float(wt))
                    for sym, wt in current.items()
                    if sym not in priority_set and float(wt) <= cleanup_floor + 1e-12
                ),
                key=lambda item: float(item[1]),
            )
            for sym, wt in small_non_targets:
                cost = float(wt) / 2.0
                if cost > budget + 1e-12:
                    break
                budget -= cost
                current.pop(sym, None)
            current = _purge_tiny(current)

        def _ordered_deficits(weights: Mapping[str, float]) -> list[tuple[str, float]]:
            rows = []
            seen = set()
            # Buy the highest target weights first.
            preferred = sorted(
                ((sym, float(target.get(sym, 0.0))) for sym in priority_list if float(target.get(sym, 0.0)) > 0.0),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            for sym, _ in preferred:
                deficit = float(target.get(sym, 0.0)) - float(weights.get(sym, 0.0))
                if deficit > 1e-12:
                    rows.append((sym, deficit))
                    seen.add(sym)
            for sym, tgt in sorted(target.items(), key=lambda item: float(item[1]), reverse=True):
                if sym in seen:
                    continue
                deficit = float(tgt) - float(weights.get(sym, 0.0))
                if deficit > 1e-12:
                    rows.append((sym, deficit))
            return rows

        def _ordered_excesses(weights: Mapping[str, float]) -> list[tuple[str, float]]:
            non_target = []
            target_over = []
            for sym, wt in weights.items():
                excess = float(wt) - float(target.get(sym, 0.0))
                if excess <= 1e-12:
                    continue
                if sym in priority_set:
                    target_over.append((sym, excess))
                else:
                    non_target.append((sym, excess))
            # Exit obsolete names first; largest weights first aligns the book faster.
            non_target.sort(key=lambda item: float(item[1]), reverse=True)
            target_over.sort(key=lambda item: float(item[1]), reverse=True)
            return non_target + target_over

        while budget > 1e-12:
            deficits = _ordered_deficits(current)
            excesses = _ordered_excesses(current)

            if deficits and excesses:
                sell_sym, sell_amt = excesses[0]
                buy_sym, buy_amt = deficits[0]
                transfer = min(float(sell_amt), float(buy_amt), float(budget))
                if transfer <= 1e-12:
                    break
                current[sell_sym] = float(current.get(sell_sym, 0.0)) - transfer
                current[buy_sym] = float(current.get(buy_sym, 0.0)) + transfer
                budget -= transfer
                current = _purge_tiny(current)
                continue

            if deficits:
                buy_sym, buy_amt = deficits[0]
                transfer = min(float(buy_amt), float(budget) * 2.0)
                if transfer <= 1e-12:
                    break
                current[buy_sym] = float(current.get(buy_sym, 0.0)) + transfer
                budget -= transfer / 2.0
                current = _purge_tiny(current)
                continue

            if excesses:
                sell_sym, sell_amt = excesses[0]
                transfer = min(float(sell_amt), float(budget) * 2.0)
                if transfer <= 1e-12:
                    break
                current[sell_sym] = float(current.get(sell_sym, 0.0)) - transfer
                budget -= transfer / 2.0
                current = _purge_tiny(current)
                continue

            break

        return _clean_abs(current)

    turnover = 0.5 * sum(abs(target.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
    if turnover <= budget + 1e-12:
        return target

    alpha = max(0.0, min(1.0, budget / turnover))
    blended = {k: (prev.get(k, 0.0) + alpha * (target.get(k, 0.0) - prev.get(k, 0.0))) for k in keys}
    return _clean_abs(blended)
