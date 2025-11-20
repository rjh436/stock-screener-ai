from typing import Dict, List, Optional, Any
import pandas as pd
import operator
from .base import BaseStrategy

# Map string operators to python functions
OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq
}

class GenericStrategy(BaseStrategy):
    """
    A strategy defined by a configuration dictionary (genome).
    This allows for dynamic generation and evolution of strategies without writing new code.
    """
    def __init__(self, genome: Dict):
        # genome structure example:
        # {
        #   "name": "Gen1_Strat5",
        #   "type": "trend_pullback",
        #   "entry_rules": [
        #       {"col": "close", "op": ">", "ref": "sma200"},
        #       {"col": "rsi2", "op": "<", "val": 10}
        #   ],
        #   "exit_rules": [
        #       {"col": "close", "op": ">", "ref": "ema5"}
        #   ],
        #   "stop_loss_atr": 2.0,
        #   "time_stop": 10
        # }
        self.genome = genome
        self._name = genome.get("name", "Generic")
        super().__init__(genome)

    @property
    def name(self) -> str:
        return self._name

    def _resolve_value(self, row: pd.Series, rule: Dict) -> float:
        """Resolve the comparison value (either a fixed 'val' or a reference column 'ref')."""
        if "val" in rule:
            return float(rule["val"])
        elif "ref" in rule:
            ref_col = rule["ref"]
            return float(row.get(ref_col, 0))
        return 0.0

    def _check_condition(self, row: pd.Series, rule: Dict) -> bool:
        col = rule["col"]
        if col not in row:
            return False
            
        val_a = float(row[col])
        val_b = self._resolve_value(row, rule)
        
        op_func = OPS.get(rule["op"])
        if not op_func:
            return False
            
        return op_func(val_a, val_b)

    def entry(self, df: pd.DataFrame, i: int) -> Optional[Dict]:
        # Warmup check
        if i < 200: return None
        
        row = df.iloc[i]
        
        # Check all entry rules (AND logic)
        for rule in self.genome.get("entry_rules", []):
            if not self._check_condition(row, rule):
                return None
                
        # Calculate Stop Loss
        atr = row.get("atr14", row["close"]*0.02)
        sl_mult = self.genome.get("stop_loss_atr", 2.0)
        
        stop_price = row["close"] - (atr * sl_mult)
        if stop_price <= 0: return None
        
        return {"entry_price": row["close"], "stop_price": stop_price}

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        row = df.iloc[i]
        
        # Calculate current profit %
        current_profit_pct = ((row["close"] - entry_price) / entry_price) * 100.0
        days_held = i - entry_i
        
        # 1. Hard Stop Loss (Always enforced)
        if row["low"] < stop_price:
            return True
            
        # 2. Maximum Time Stop (Default 60 days if not specified)
        max_time_stop = self.genome.get("max_time_stop", 60)
        if days_held >= max_time_stop:
            return True
            
        # 3. MINIMUM PROFIT TARGET GATE (< 10%)
        if current_profit_pct < 10.0:
            # HOLD AGGRESSIVELY
            # Only exit on extreme RSI spike (Parabolic Blow-off)
            if row["rsi2"] > 95:
                return True
            return False # HOLD
        
        # 4. ABOVE 10%: TRAILING STOP
        if current_profit_pct < 20.0:
            # EMA20 trailing (10-20% profit)
            if row["close"] < row.get("ema20", row["sma20"]):
                return True
        else:
            # EMA50 trailing (>20% profit - let winners run)
            if row["close"] < row.get("ema50", row["sma50"]):
                return True
                
        return False
