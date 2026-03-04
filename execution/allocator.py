from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class AllocationDecision:
    gross_target: float
    offense_score: float
    sleeve_weights: Dict[str, float]


class RegimeAllocator:
    """
    Lightweight meta-allocator that maps regime+breadth state into
    portfolio gross target and sleeve budget weights.

    Defaults preserve offensive bias while reducing exposure in weak breadth.
    """

    def __init__(self, config: Optional[Dict[str, object]] = None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.max_gross = float(cfg.get("max_gross", 1.0) or 1.0)
        self.min_gross = float(cfg.get("min_gross", 0.25) or 0.25)
        self.max_gross = float(np.clip(self.max_gross, 0.0, 1.0))
        self.min_gross = float(np.clip(self.min_gross, 0.0, self.max_gross))

        self.state_thresholds = dict(cfg.get("state_thresholds", {}) or {})

        self._bull_weights = self._normalize_weights(
            cfg.get(
                "bull_weights",
                {"breakout": 0.60, "continuation": 0.30, "recovery": 0.10},
            )
        )
        self._neutral_weights = self._normalize_weights(
            cfg.get(
                "neutral_weights",
                {"breakout": 0.40, "continuation": 0.40, "recovery": 0.20},
            )
        )
        self._defensive_weights = self._normalize_weights(
            cfg.get(
                "defensive_weights",
                {"breakout": 0.20, "continuation": 0.30, "recovery": 0.50},
            )
        )

    @staticmethod
    def _normalize_weights(weights: object) -> Dict[str, float]:
        if not isinstance(weights, dict):
            return {"breakout": 1.0}
        parsed: Dict[str, float] = {}
        total = 0.0
        for raw_key, raw_value in weights.items():
            key = str(raw_key).strip().lower()
            if not key:
                continue
            try:
                val = float(raw_value)
            except Exception:
                continue
            if not np.isfinite(val) or val <= 0:
                continue
            parsed[key] = val
            total += val
        if total <= 0:
            return {"breakout": 1.0}
        return {k: v / total for k, v in parsed.items()}

    @staticmethod
    def _coerce_prob(value: object) -> float:
        try:
            x = float(value)
        except Exception:
            return float("nan")
        if not np.isfinite(x):
            return float("nan")
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _base_from_regime(self, regime_state: str) -> float:
        state = str(regime_state or "RED").upper()
        if state == "GREEN":
            return 0.90
        if state == "YELLOW":
            return 0.70
        if state == "ORANGE":
            return 0.45
        return 0.15

    def compute_offense_score(
        self,
        *,
        regime_state: str,
        breadth_50: object,
        breadth_200: object,
        breadth_rs: object,
        breakout_hit_rate: object = float("nan"),
    ) -> float:
        score = self._base_from_regime(regime_state)

        b50 = self._coerce_prob(breadth_50)
        b200 = self._coerce_prob(breadth_200)
        brs = self._coerce_prob(breadth_rs)
        bhr = self._coerce_prob(breakout_hit_rate)

        if np.isfinite(b50):
            score += 0.30 * (b50 - 0.50)
        if np.isfinite(b200):
            score += 0.20 * (b200 - 0.50)
        if np.isfinite(brs):
            score += 0.10 * (brs - 0.50)
        if np.isfinite(bhr):
            score += 0.10 * (bhr - 0.50)

        return float(np.clip(score, 0.0, 1.0))

    def allocate(
        self,
        *,
        regime_state: str,
        breadth_50: object,
        breadth_200: object,
        breadth_rs: object,
        breakout_hit_rate: object = float("nan"),
    ) -> AllocationDecision:
        if not self.enabled:
            return AllocationDecision(
                gross_target=1.0,
                offense_score=1.0,
                sleeve_weights={"breakout": 1.0},
            )

        offense = self.compute_offense_score(
            regime_state=regime_state,
            breadth_50=breadth_50,
            breadth_200=breadth_200,
            breadth_rs=breadth_rs,
            breakout_hit_rate=breakout_hit_rate,
        )

        bull_floor = float(self.state_thresholds.get("bull", 0.72) or 0.72)
        neutral_floor = float(self.state_thresholds.get("neutral", 0.45) or 0.45)

        if offense >= bull_floor:
            weights = self._bull_weights
        elif offense >= neutral_floor:
            weights = self._neutral_weights
        else:
            weights = self._defensive_weights

        gross_target = self.min_gross + ((self.max_gross - self.min_gross) * offense)
        gross_target = float(np.clip(gross_target, self.min_gross, self.max_gross))

        return AllocationDecision(
            gross_target=gross_target,
            offense_score=offense,
            sleeve_weights=weights,
        )
