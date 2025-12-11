import math
import operator
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy

OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}
MIN_BARS = 200


class GenericStrategy(BaseStrategy):
    def __init__(self, genome: Dict):
        self.genome = genome
        self._name = genome.get("name", "Generic")
        super().__init__(genome)

    @property
    def name(self) -> str:
        return self._name

    def _resolve_value(self, row: pd.Series, rule: Dict) -> float:
        if "val" in rule:
            return float(rule["val"])
        if "ref" in rule:
            return float(row.get(rule["ref"], 0))
        return 0.0

    def _check_condition(self, row: pd.Series, rule: Dict) -> bool:
        col = rule.get("col")
        op = OPS.get(rule.get("op"))
        if col not in row or op is None:
            return False
        val_a = row.get(col, np.nan)
        val_b = self._resolve_value(row, rule)
        if val_a is None or np.isnan(val_a) or val_b is None or np.isnan(val_b):
            return False
        return bool(op(float(val_a), float(val_b)))

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        warmup = int(self.genome.get("warmup_bars", MIN_BARS))
        if i < warmup:
            return None
        row = df.iloc[i]
        for rule in self.genome.get("entry_rules", []):
            if not self._check_condition(row, rule):
                return None

        atr = row.get("atr14", row.get("close", 0) * 0.02)
        mult = float(self.genome.get("stop_loss_atr", 2.5))
        stop_price = float(row.get("close", 0)) - (atr * mult)
        return {"entry_price": float(row.get("close", 0)), "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        """
        Enhanced exit logic with trailing stops and hard loss floor.
        """
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        pnl_pct = ((close_px - entry_price) / entry_price) * 100

        # === DYNAMIC TRAILING STOP ===
        atr = row.get("atr14", close_px * 0.02)
        stop_mult = float(self.genome.get("stop_loss_atr", 3.0))
        trailing_stop = close_px - (atr * stop_mult)

        # Hard floor: Never lose more than 12% (tightened to prevent gap-down slippage)
        hard_floor = entry_price * 0.88

        # Use the HIGHEST stop (tightest protection)
        effective_stop = max(trailing_stop, hard_floor, stop_price)

        # 1. STOP LOSS CHECK
        if row.get("low", np.inf) < effective_stop:
            return True

        # 2. TIME-BASED EXITS
        time_limit = int(self.genome.get("time_stop", 45))

        # Progressive tightening in final 20% of hold period
        if days_held >= time_limit * 0.8:
            if pnl_pct < -5.0:
                return True
            if pnl_pct < 3.0:
                sma20 = row.get("sma20", close_px)
                if close_px < sma20:
                    return True

        # Final time limit
        if days_held >= time_limit:
            if pnl_pct < 0:
                return True
            if pnl_pct < 8.0:
                return True
            sma50 = row.get("sma50")
            if sma50 and close_px < sma50:
                return True

        # 3. TREND BREAK PROTECTION
        sma50 = row.get("sma50")
        if sma50 and close_px < sma50:
            if pnl_pct > 0:
                return True
            if days_held > 10:
                return True

        # 4. PROFIT TARGETS
        for rule in self.genome.get("exit_rules", []):
            if rule.get("type") == "profit_target":
                target_multiple = float(rule.get("val", 1.0))
                if row.get("high", 0) >= (entry_price * target_multiple):
                    return True
            elif self._check_condition(row, rule):
                return True

        return False
