import pandas as pd
import operator
from typing import Dict, Optional
from .base import BaseStrategy

OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}

class GenericStrategy(BaseStrategy):
    def __init__(self, genome: Dict):
        self.genome = genome
        self._name = genome.get("name", "Generic")
        super().__init__(genome)

    @property
    def name(self) -> str:
        return self._name

    def _resolve_value(self, row: pd.Series, rule: Dict) -> float:
        if "val" in rule: return float(rule["val"])
        if "ref" in rule: return float(row.get(rule["ref"], 0))
        return 0.0

    def _check_condition(self, row: pd.Series, rule: Dict) -> bool:
        if rule["col"] not in row: return False
        val_a = float(row[rule["col"]])
        val_b = self._resolve_value(row, rule)
        op = OPS.get(rule["op"])
        return op(val_a, val_b) if op else False

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        if i < 200: return None
        row = df.iloc[i]
        
        # Check Entry Rules
        for rule in self.genome.get("entry_rules", []):
            if not self._check_condition(row, rule): return None
            
        # Stop Loss
        atr = row.get("atr14", row["close"] * 0.02)
        mult = self.genome.get("stop_loss_atr", 2.0)
        return {"entry_price": row["close"], "stop_price": row["close"] - (atr * mult)}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        days = i - entry_i
        
        # 1. Hard Stop
        if row["low"] < stop_price: return True
        
        # 2. Time Stop (Corrected parameter name)
        time_stop = self.genome.get("time_stop", 40)
        if days >= time_stop: return True
        
        # 3. Exit Rules (If any exist)
        # If list is empty, we HOLD until Time Stop or Stop Loss
        exit_rules = self.genome.get("exit_rules", [])
        if exit_rules:
            for rule in exit_rules:
                if self._check_condition(row, rule): return True
                
        return False
