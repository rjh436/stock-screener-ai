import operator
import numpy as np
import pandas as pd

from .base import BaseStrategy

OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}
MIN_BARS = 200


class ApexWealthStrategy(BaseStrategy):
    def __init__(self, genome: dict):
        self.genome = genome
        self._name = genome.get("name", "Apex Wealth")
        super().__init__(genome)

    @property
    def name(self) -> str:
        return self._name

    def _resolve_value(self, row: pd.Series, rule: dict) -> float:
        if "val" in rule:
            return float(rule["val"])
        if "ref" in rule:
            return float(row.get(rule["ref"], 0))
        return 0.0

    def _check_condition(self, row: pd.Series, rule: dict) -> bool:
        col = rule.get("col")
        op = OPS.get(rule.get("op"))
        if col not in row or op is None:
            return False
        val_a = row.get(col, np.nan)
        val_b = self._resolve_value(row, rule)
        if val_a is None or np.isnan(val_a) or val_b is None or np.isnan(val_b):
            return False
        return bool(op(float(val_a), float(val_b)))

    def entry(self, df: pd.DataFrame, i: int):
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
        row = df.iloc[i]
        days_held = i - entry_i

        # Hard stop
        if row.get("low", np.inf) <= stop_price:
            return True

        # Time stop (patient)
        if days_held >= int(self.genome.get("time_stop", 60)):
            return True

        # Profit target
        if row.get("high", 0) >= entry_price * 1.20:
            return True

        # Optional extra exit rules
        for rule in self.genome.get("exit_rules", []):
            if rule.get("type") == "profit_target":
                target_multiple = float(rule.get("val", 1.0))
                if row.get("high", 0) >= (entry_price * target_multiple):
                    return True
            elif self._check_condition(row, rule):
                return True

        return False
