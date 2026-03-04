from __future__ import annotations

from typing import Any, Dict, Optional


def select_breakout_candidate(
    vcp_candidate: Optional[Dict[str, Any]],
    ep_candidate: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the strongest breakout-family candidate and normalize sleeve metadata."""
    candidates = []
    if isinstance(vcp_candidate, dict):
        cand = dict(vcp_candidate)
        cand.setdefault("entry_type", "vcp")
        cand.setdefault("sleeve", "breakout")
        candidates.append(cand)
    if isinstance(ep_candidate, dict):
        cand = dict(ep_candidate)
        cand.setdefault("entry_type", "ep")
        cand.setdefault("sleeve", "breakout")
        candidates.append(cand)
    if not candidates:
        return None
    return max(candidates, key=lambda c: float(c.get("signal_strength", 0.0)))
