from __future__ import annotations

from typing import Dict, Mapping

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
    return {k: (v / total) for k, v in out.items()}


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

    turnover = 0.5 * sum(abs(target.get(k, 0.0) - prev.get(k, 0.0)) for k in keys)
    if turnover <= budget + 1e-12:
        return target

    alpha = max(0.0, min(1.0, budget / turnover))
    blended = {k: (prev.get(k, 0.0) + alpha * (target.get(k, 0.0) - prev.get(k, 0.0))) for k in keys}
    return _clean_abs(blended)
