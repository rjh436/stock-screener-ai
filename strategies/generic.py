import math
import operator
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import time

from .base import BaseStrategy

OPS = {">": operator.gt, "<": operator.lt, ">=": operator.ge, "<=": operator.le, "==": operator.eq}
MIN_BARS = 200

try:
    import yfinance as yf
except Exception:  # Optional dependency
    yf = None

_YF_CRUMB_PATCHED = False


def _patch_yfinance_invalid_crumb() -> bool:
    """
    Workaround for Yahoo 'Invalid Crumb' failures in some environments.
    yfinance bootstraps cookies via fc.yahoo.com, which may be blocked or
    mis-resolved; this swaps the cookie bootstrap to a consent endpoint that
    sets the required A3 cookie without following redirects.
    """
    global _YF_CRUMB_PATCHED
    if _YF_CRUMB_PATCHED:
        return True

    try:
        if yf is None:
            return False
        from yfinance.data import YfData
        from curl_cffi import requests as curl_requests
    except Exception:
        return False

    def _get_cookie_basic(self, timeout=30):
        if getattr(self, "_cookie", None) is not None:
            return True

        try:
            if self._load_cookie_curlCffi():
                return True
        except Exception:
            pass

        try:
            self._session.get(
                url="https://guce.yahoo.com/consent",
                timeout=timeout,
                allow_redirects=False,
            )
        except curl_requests.exceptions.DNSError:
            return False
        except Exception:
            return False

        try:
            cookies = self._session.cookies.jar._cookies
            yahoo_domains = [d for d in cookies.keys() if "yahoo" in d]
            if len(yahoo_domains) > 1:
                yahoo_domains = [d for d in yahoo_domains if "consent" not in d]
            if not yahoo_domains:
                return False
            domain = yahoo_domains[0]
            cookie = cookies[domain]["/"].get("A3")
            if cookie is None:
                return False
            self._cookie = cookie
        except Exception:
            return False

        try:
            self._save_cookie_curlCffi()
        except Exception:
            pass

        return True

    YfData._get_cookie_basic = _get_cookie_basic
    _YF_CRUMB_PATCHED = True
    return True


class GenericStrategy(BaseStrategy):
    def __init__(self, genome: Dict):
        self.genome = genome
        self._name = genome.get("name", "Generic")
        self.fundamental_cache = {}
        super().__init__(genome)

    @property
    def params(self) -> Dict:
        return self.genome

    @params.setter
    def params(self, value: Dict) -> None:
        self.genome = value or {}

    @property
    def name(self) -> str:
        return self._name

    def check_fundamentals(self, ticker: str) -> bool:
        """
        The 'Anti-Rot' Filter.
        Returns True if the stock is fundamentally safe enough to trade.
        Rejects 'Value Traps' (Toxic debt or no earnings visibility).

        Fail-open: If the data fetch fails, allow the trade to avoid starvation.
        """
        if not ticker:
            return True

        cached = self.fundamental_cache.get(ticker)
        if cached is not None:
            return cached

        try:
            if yf is None:
                self.fundamental_cache[ticker] = True
                return True

            _patch_yfinance_invalid_crumb()
            stock = yf.Ticker(ticker)
            info = stock.info or {}

            debt_eq = info.get("debtToEquity")
            forward_pe = info.get("forwardPE")
            trailing_pe = info.get("trailingPE")

            if debt_eq is None:
                debt_eq = self._compute_debt_to_equity_fallback(stock)

            # Rule 1: Excessive Debt (>300%)
            if debt_eq is not None and debt_eq > 300:
                self.fundamental_cache[ticker] = False
                return False

            # Rule 2: Zombie (No earnings visibility)
            if trailing_pe is None and forward_pe is None:
                self.fundamental_cache[ticker] = False
                return False

            self.fundamental_cache[ticker] = True
            return True

        except Exception:
            # Fail-open: If yfinance glitches, don't starve the system.
            self.fundamental_cache[ticker] = True
            return True

    def _compute_debt_to_equity_fallback(self, stock: Any) -> Optional[float]:
        """
        Computes Debt/Equity % from the balance sheet when `debtToEquity`
        is unavailable (common when equity is negative).
        """
        try:
            bs = stock.balance_sheet
        except Exception:
            return None

        if bs is None or getattr(bs, "empty", True):
            return None

        latest_col = bs.columns[0]

        total_debt = None
        if "Total Debt" in bs.index:
            total_debt = bs.loc["Total Debt", latest_col]

        equity = None
        for key in (
            "Stockholders Equity",
            "Total Stockholder Equity",
            "Total Equity Gross Minority Interest",
            "Total Equity",
        ):
            if key in bs.index:
                equity = bs.loc[key, latest_col]
                break

        try:
            total_debt_val = float(total_debt)
            equity_val = float(equity)
        except Exception:
            return None

        if total_debt_val <= 0:
            return 0.0

        if equity_val <= 0:
            return float("inf")

        return (total_debt_val / equity_val) * 100.0

    def _resolve_value(self, row: pd.Series, rule: Dict) -> float:
        if "val" in rule:
            return float(rule["val"])
        if "ref" in rule:
            base = row.get(rule["ref"], 0)
            try:
                base = float(base)
            except Exception:
                return float("nan")
            mult = rule.get("mult")
            if mult is not None:
                try:
                    base *= float(mult)
                except Exception:
                    return float("nan")
            return base
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

        if self.genome.get("use_fundamentals", False):
            ticker = row.get("ticker", None)
            if ticker:
                if not self.check_fundamentals(str(ticker)):
                    return None

        _ = row  # keep signature stable; engine computes fills/stops natively
        return {
            "limit_ratio": self.params.get("limit_ratio"),
            "stop_loss_atr": self.params.get("stop_loss_atr"),
            "stop_loss_type": self.params.get("stop_loss_type", "atr"),
        }

    def exit(self, df: pd.DataFrame, i: int, entry_i: int, entry_price: float, stop_price: float) -> bool:
        """
        Fallback exit logic.

        Note: Primary execution for `GenericStrategy` genomes is handled by the native
        engine (`execution/engine.py`) via `_generic_exit_decision`. This method is
        kept for compatibility and non-native execution paths.
        """
        row = df.iloc[i]
        days_held = i - entry_i
        close_px = row.get("close", entry_price)
        pnl_pct = ((close_px - entry_price) / entry_price) * 100
        use_bb_exit = bool(self.genome.get("use_bb_exit", False))

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

        # 1b. Bollinger profit release (optional)
        if use_bb_exit:
            bb_upper = row.get("bb_upper", np.nan)
            if pd.notna(bb_upper) and row.get("high", close_px) >= bb_upper:
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
            if use_bb_exit and rule.get("type") == "profit_target":
                continue
            if rule.get("type") == "profit_target":
                target_multiple = float(rule.get("val", 1.0))
                if row.get("high", 0) >= (entry_price * target_multiple):
                    return True
            elif self._check_condition(row, rule):
                return True

        return False
