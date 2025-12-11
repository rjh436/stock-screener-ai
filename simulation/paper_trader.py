
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from data.loader import fetch_single_symbol, fetch_data_pack
from data.indices import get_index_symbols
from data.schwab_client import sd
from execution.engine import (
    _compute_indicators,
    MIN_BARS,
    DEFAULT_SCORING_WEIGHTS,
    calculate_backtest_quality_score,
    get_sector,
)
from strategies.generic import GenericStrategy
from strategies.strategy_loader import load_strategies

PORTFOLIO_FILE = "data/paper_portfolio.json"
DEFAULT_CONFIG_PATH = "config/generated_strategies.json"
POSITION_FRACTION = 0.20
MAX_POSITIONS = 5
SECTOR_CAP = 0.60


class PaperTrader:
    def __init__(self, configs: Optional[List[Dict]] = None, start_cash: float = 100000.0):
        self.start_cash = start_cash
        self.scoring_weights = DEFAULT_SCORING_WEIGHTS
        self.strategy_configs = configs if configs is not None else self._load_default_configs()
        self.strategies = self._init_strategies(self.strategy_configs)
        self.strategy_map = {s.name: s for s in self.strategies}
        self.sector_map = self._load_sector_map()
        self.state = self._load_state()
        self._rehydrate_positions()
        self.state.setdefault("pending_orders", [])

    @property
    def portfolio(self) -> Dict:
        return self.state.setdefault("positions", {})

    def _init_strategies(self, configs: Optional[List[Dict]]) -> List[GenericStrategy]:
        strategies = load_strategies(configs or [])
        if not strategies:
            strategies = [GenericStrategy({"name": "Generic", "type": "generic", "entry_rules": [], "exit_rules": [], "stop_loss_atr": 3.0, "time_stop": 70})]
        return strategies

    def _load_default_configs(self) -> List[Dict]:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r") as f:
                    return json.load(f)
            except:  # noqa: E722 - best effort fallback
                return []
        return []

    def _load_sector_map(self) -> Dict[str, str]:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/sectors.json"))
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:  # noqa: E722
                return {}
        return {}

    def _resolve_sector(self, sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in self.sector_map:
            return self.sector_map[sym_up]
        return get_sector(sym_up)

    def _default_strategy(self) -> GenericStrategy:
        return self.strategies[0] if self.strategies else GenericStrategy({"name": "Generic"})

    def _match_strategy(self, name: str) -> GenericStrategy:
        if name in self.strategy_map:
            return self.strategy_map[name]
        return self._default_strategy()

    def _rehydrate_positions(self):
        fallback = self._default_strategy()
        for sym, pos in self.portfolio.items():
            strat_name = pos.get("strategy_name") or pos.get("strategy") or fallback.name
            strat_obj = self.strategy_map.get(strat_name, fallback)
            pos["strategy_name"] = strat_name
            pos["strategy"] = strat_name
            pos["strategy_obj"] = strat_obj

        for order in self.state.get("pending_orders", []):
            strat_name = order.get("strategy_name") or order.get("strategy")
            if strat_name:
                order["strategy_obj"] = self.strategy_map.get(strat_name, fallback)
                order.setdefault("strategy_name", strat_name)

    def _load_state(self):
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                state = json.load(f)
                if "pending_orders" not in state:
                    state["pending_orders"] = []
                if "history" not in state:
                    state["history"] = []
                if "equity_curve" not in state:
                    state["equity_curve"] = [{"date": str(datetime.now().date()), "equity": self.start_cash}]
                return state
        return {
            "cash": self.start_cash,
            "equity": self.start_cash,
            "positions": {},
            "pending_orders": [],
            "history": [],
            "equity_curve": [{"date": str(datetime.now().date()), "equity": self.start_cash}],
        }

    def save_state(self):
        positions = {}
        for sym, pos in self.portfolio.items():
            sanitized = {k: v for k, v in pos.items() if k != "strategy_obj"}
            positions[sym] = sanitized

        pending_orders = []
        for order in self.state.get("pending_orders", []):
            clean_order = dict(order)
            clean_order.pop("strategy_obj", None)
            if "strategy_name" not in clean_order and "strategy" in clean_order:
                clean_order["strategy_name"] = clean_order["strategy"]
            pending_orders.append(clean_order)

        state_copy = dict(self.state)
        state_copy["positions"] = positions
        state_copy["pending_orders"] = pending_orders

        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(state_copy, f, indent=4)

    def reset_account(self):
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)
        self.__init__(configs=self.strategy_configs, start_cash=self.start_cash)

    def get_realtime_price(self, sym):
        # Priority: Quote -> History -> 0
        try:
            q = sd.get_quote(sym)
            if q and sym in q and "quote" in q[sym]:
                q_d = q[sym]["quote"]
                return float(q_d.get("lastPrice") or q_d.get("mark") or q_d.get("closePrice") or 0.0)
        except:  # noqa: E722
            pass

        # Fallback
        df = fetch_single_symbol(sym, days=5, force_fresh=False)
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
        return 0.0

    def close_position(self, symbol, reason="Manual"):
        """Manually sells a position at the current real-time price."""
        if symbol not in self.portfolio:
            return False, "Position not found"

        pos = self.portfolio[symbol]
        exit_price = self.get_realtime_price(symbol)

        if exit_price <= 0:
            return False, "Could not fetch valid price"

        proceeds = exit_price * pos["shares"]
        pnl = proceeds - (pos["entry_price"] * pos["shares"])
        pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100

        self.state["cash"] += proceeds
        self.state["history"].append(
            {
                "symbol": symbol,
                "strategy": pos.get("strategy_name", ""),
                "type": "SELL",
                "reason": reason,
                "entry_date": pos["date"],
                "exit_date": str(datetime.now().date()),
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "shares": pos["shares"],
                "pnl": pnl,
                "return_pct": pnl_pct,
            }
        )

        del self.portfolio[symbol]
        self.update_valuations()  # Refresh equity
        self.save_state()
        return True, f"Sold {symbol} at ${exit_price:.2f} (PnL: ${pnl:.2f})"

    def process_pending_orders(self, market_data: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
        """
        Fill queued Market-On-Open orders using provided market_data or live fetch.
        """
        filled_log: List[str] = []
        remaining_orders: List[Dict] = []

        for order in self.state.get("pending_orders", []):
            sym = order.get("symbol")
            if not sym:
                continue

            open_price = self._resolve_open_price(sym, market_data)
            if open_price <= 0:
                filled_log.append(f"❌ {sym}: No valid open price")
                continue

            target_value = float(order.get("target_value", 0) or 0)
            if target_value <= 0:
                target_value = float(order.get("est_price", open_price)) * float(order.get("shares", 0) or 0)

            shares = int(target_value / open_price)
            if shares <= 0:
                filled_log.append(f"❌ {sym}: Target value too low for open price")
                continue

            cost = shares * open_price
            if self.state["cash"] < cost:
                filled_log.append(f"⚠️ {sym}: Skipped, insufficient cash for ${cost:.2f}")
                continue

            success = self.buy(
                sym,
                open_price,
                shares,
                stop_price=order.get("stop_price"),
                strategy_name=order.get("strategy_name"),
                entry_index=order.get("entry_i"),
            )
            if success:
                filled_log.append(f"✅ FILLED {sym} x{shares} @ ${open_price:.2f}")

        # Clear queue after processing
        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return filled_log

    def _resolve_open_price(self, symbol: str, market_data: Optional[Dict[str, pd.DataFrame]] = None) -> float:
        """Derive open price from provided market data or fallbacks."""
        try:
            if market_data and symbol in market_data:
                df = market_data[symbol]
                if df is not None and not df.empty:
                    row = df.iloc[-1]
                    return float(row.get("open") or row.get("close"))
        except Exception:
            pass

        df_live = fetch_single_symbol(symbol, days=3, force_fresh=True)
        if df_live is not None and not df_live.empty:
            try:
                return float(df_live.iloc[-1].get("open"))
            except Exception:
                pass

        return self.get_realtime_price(symbol)

    def _current_sector_exposure(self, include_pending: bool = False) -> Dict[str, float]:
        exposure: Dict[str, float] = {}
        for sym, pos in self.portfolio.items():
            price = pos.get("current_price", pos.get("entry_price", 0))
            val = pos.get("shares", 0) * price
            sec = self._resolve_sector(sym)
            exposure[sec] = exposure.get(sec, 0.0) + val

        if include_pending:
            for order in self.state.get("pending_orders", []):
                sym = order.get("symbol")
                if not sym:
                    continue
                sec = self._resolve_sector(sym)
                target_val = float(order.get("target_value", 0) or 0)
                if target_val <= 0:
                    shares = float(order.get("shares", 0) or 0)
                    est_price = float(order.get("est_price", 0) or 0)
                    target_val = shares * est_price
                exposure[sec] = exposure.get(sec, 0.0) + target_val
        return exposure

    def update_valuations(self):
        total_value = self.state["cash"]

        for sym, pos in self.portfolio.items():
            current_price = self.get_realtime_price(sym)

            if current_price > 0:
                pos["current_price"] = current_price
                pos["unrealized_pnl"] = (current_price - pos["entry_price"]) * pos["shares"]
                pos["unrealized_pct"] = (current_price - pos["entry_price"]) / pos["entry_price"] * 100
                total_value += (current_price * pos["shares"])
            else:
                total_value += (pos["entry_price"] * pos["shares"])

        self.state["equity"] = total_value
        today = str(datetime.now().date())
        if not self.state["equity_curve"] or self.state["equity_curve"][-1]["date"] != today:
            self.state["equity_curve"].append({"date": today, "equity": total_value})
        else:
            self.state["equity_curve"][-1]["equity"] = total_value

        self.save_state()
        return self.state, []

    def buy(self, symbol: str, price: float, shares: int, stop_price: Optional[float] = None, strategy_obj=None, strategy_name: Optional[str] = None, entry_index: Optional[int] = None):
        if shares <= 0 or price <= 0:
            return False
        cost = shares * price
        if self.state["cash"] < cost:
            return False

        strat_obj = strategy_obj or self._match_strategy(strategy_name or "")
        strat_name = strategy_name or getattr(strat_obj, "name", "Unknown")
        stop_val = stop_price if stop_price is not None else price * 0.9

        self.state["cash"] -= cost
        self.portfolio[symbol] = {
            "shares": shares,
            "entry_price": price,
            "stop_price": stop_val,
            "strategy": strat_name,
            "strategy_name": strat_name,
            "strategy_obj": strat_obj,
            "date": str(datetime.now().date()),
            "current_price": price,
            "unrealized_pnl": 0.0,
            "unrealized_pct": 0.0,
            "entry_i": entry_index,
        }
        self.update_valuations()
        return True

    def queue_order(self, symbol: str, target_value: float, est_price: float, stop_price: Optional[float] = None, strategy_obj=None, strategy_name: Optional[str] = None, entry_index: Optional[int] = None) -> bool:
        if not symbol or target_value <= 0 or est_price <= 0:
            return False
        if symbol in self.portfolio or self._is_pending(symbol):
            return False

        strat_obj = strategy_obj or self._match_strategy(strategy_name or "")
        strat_name = strategy_name or getattr(strat_obj, "name", "Unknown")
        self.state.setdefault("pending_orders", [])
        self.state["pending_orders"].append(
            {
                "symbol": symbol,
                "target_value": target_value,
                "est_price": est_price,
                "stop_price": stop_price if stop_price is not None else est_price * 0.9,
                "strategy_name": strat_name,
                "strategy_obj": strat_obj,
                "entry_i": entry_index,
                "type": "MOO",
            }
        )
        self.save_state()
        return True

    def _is_pending(self, symbol: str) -> bool:
        return any((o.get("symbol") or "").upper() == symbol.upper() for o in self.state.get("pending_orders", []))

    def _normalize_candidate(self, cand: Dict) -> Optional[Dict]:
        sym = cand.get("symbol") or cand.get("Symbol")
        if not sym:
            return None

        price = cand.get("price") or cand.get("Price")
        if price is None:
            price = self.get_realtime_price(sym)
        try:
            price = float(price)
        except:  # noqa: E722
            return None
        if price <= 0:
            return None

        stop = cand.get("stop") or cand.get("Stop")
        strategy_name = cand.get("strategy_name") or cand.get("strategy") or cand.get("Strategy")
        strat_obj = self._match_strategy(strategy_name) if strategy_name else self._default_strategy()
        score = cand.get("score")
        if score is None:
            score = cand.get("Raw_Score", 50)
            if "wealth" in strat_obj.name.lower():
                score *= 1.3
        else:
            try:
                score = float(score)
            except:  # noqa: E722
                score = 50

        return {
            "symbol": sym,
            "price": price,
            "stop": stop if stop is not None else price * 0.9,
            "strategy_name": strat_obj.name,
            "strategy_obj": strat_obj,
            "score": score,
            "entry_i": cand.get("entry_i"),
        }

    def _execute_governor(self, candidates: List[Dict]) -> List[str]:
        logs = []
        normalized = []
        for cand in candidates:
            norm = self._normalize_candidate(cand)
            if norm:
                normalized.append(norm)

        normalized.sort(key=lambda x: x.get("score", 0), reverse=True)
        sector_exposure = self._current_sector_exposure(include_pending=True)

        for cand in normalized:
            if len(self.portfolio) >= MAX_POSITIONS:
                break
            if cand["symbol"] in self.portfolio or self._is_pending(cand["symbol"]):
                continue

            current_equity = self.state["cash"] + sum(sector_exposure.values())
            trade_val = current_equity * POSITION_FRACTION
            if trade_val <= 0 or self.state["cash"] < trade_val:
                continue

            sec = self._resolve_sector(cand["symbol"])
            projected_exp = (sector_exposure.get(sec, 0.0) + trade_val) / current_equity if current_equity > 0 else 1.0
            if projected_exp > SECTOR_CAP:
                continue

            shares = int(trade_val / cand["price"])
            if shares <= 0:
                continue

            queued = self.queue_order(
                cand["symbol"],
                trade_val,
                cand["price"],
                stop_price=cand["stop"],
                strategy_obj=cand["strategy_obj"],
                strategy_name=cand["strategy_name"],
                entry_index=cand.get("entry_i"),
            )
            if queued:
                sector_exposure[sec] = sector_exposure.get(sec, 0.0) + trade_val
                logs.append(f"📝 QUEUED {cand['symbol']} for MOO (${trade_val:.2f})")
        return logs

    def run_daily_scan(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, global_data: Optional[Dict[str, pd.DataFrame]] = None, scoring_weights: Optional[Dict] = None) -> List[str]:
        scoring = scoring_weights or self.scoring_weights
        vix_df = global_data.get("VIX") if global_data else None
        spy_df = global_data.get("SPY") if global_data else None
        if data_dict is None:
            tickers = get_index_symbols("S&P 1500")
            print(f"Loaded {len(tickers)} tickers from S&P 1500")
            data_dict = fetch_data_pack(tickers, days=400)
        else:
            tickers = list(data_dict.keys())
            print(f"Loaded {len(tickers)} tickers from S&P 1500")

        candidates = []
        for strat in self.strategies:
            for sym, df in data_dict.items():
                if df is None or df.empty:
                    continue
                try:
                    enriched = _compute_indicators(df.copy(), spy_df=spy_df)
                    if vix_df is not None:
                        enriched["vix"] = vix_df["close"].reindex(enriched.index).ffill().fillna(20.0)
                    else:
                        enriched["vix"] = 20.0
                    if len(enriched) <= MIN_BARS:
                        continue

                    signal_idx = len(enriched) - 2
                    trade_idx = signal_idx + 1
                    if signal_idx < 0 or trade_idx <= 0:
                        continue
                    if trade_idx < MIN_BARS:
                        continue
                    if not strat.entry(enriched, signal_idx):
                        continue

                    row_prev = enriched.iloc[signal_idx]
                    raw_score = calculate_backtest_quality_score(row_prev, strat.name, weights=scoring)
                    score = raw_score * 1.3 if "wealth" in strat.name.lower() else raw_score
                    row_curr = enriched.iloc[trade_idx]
                    open_px = float(row_curr.get("open", row_curr["close"]))
                    signal_atr = float(row_prev.get("atr14", 0))
                    current_atr = float(row_curr.get("atr14", signal_atr))
                    effective_atr = max(signal_atr, current_atr)
                    stop_mult = float(getattr(strat, "params", {}).get("stop_loss_atr", 3.0))
                    stop_price = open_px - (effective_atr * stop_mult)

                    candidates.append(
                        {
                            "symbol": sym,
                            "price": open_px,
                            "stop": stop_price,
                            "strategy_name": strat.name,
                            "strategy_obj": strat,
                            "score": score,
                            "entry_i": trade_idx,
                        }
                    )
                except:  # noqa: E722
                    continue

        return self._execute_governor(candidates)

    def execute_entries(self, candidates: List[Dict]) -> List[str]:
        return self._execute_governor(candidates)

    def process_exits(self, strategies_map=None):
        exits = []
        for sym, pos in list(self.portfolio.items()):
            strat_obj = pos.get("strategy_obj") or self._match_strategy(pos.get("strategy_name") or pos.get("strategy") or "")
            if strat_obj is None and strategies_map:
                strat_name = pos.get("strategy_name") or pos.get("strategy")
                if strat_name and strategies_map.get(strat_name):
                    strat_obj = GenericStrategy(strategies_map[strat_name])
            if strat_obj is None:
                strat_obj = self._default_strategy()
            pos["strategy_obj"] = strat_obj

            df = fetch_single_symbol(sym, days=200, force_fresh=False)
            if df is None or df.empty:
                continue

            df = _compute_indicators(df)
            current_idx = len(df) - 1
            if current_idx < 0:
                continue

            entry_i = pos.get("entry_i")
            if entry_i is None:
                try:
                    entry_dt = pd.to_datetime(pos["date"]).date()
                    today_dt = datetime.now().date()
                    days_held = (today_dt - entry_dt).days
                    entry_i = max(0, current_idx - days_held)
                except:  # noqa: E722
                    entry_i = current_idx

            stop_price = pos.get("stop_price", pos.get("entry_price", 0) * 0.9)
            if strat_obj.exit(df, current_idx, entry_i, pos["entry_price"], stop_price):
                exit_price = float(df.iloc[-1]["close"])
                if df.iloc[-1]["low"] < stop_price:
                    exit_price = stop_price

                proceeds = exit_price * pos["shares"]
                pnl = proceeds - (pos["entry_price"] * pos["shares"])
                pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

                self.state["cash"] += proceeds
                self.state["history"].append(
                    {
                        "symbol": sym,
                        "strategy": pos.get("strategy_name", strat_obj.name),
                        "type": "AUTO_EXIT",
                        "reason": "Signal/Stop",
                        "entry_date": pos["date"],
                        "exit_date": str(datetime.now().date()),
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "return_pct": pct,
                    }
                )
                del self.portfolio[sym]
                exits.append(f"SOLD {sym} at ${exit_price:.2f} ({pnl:.2f})")

        self.save_state()
        return exits

    # Alias for compatibility with newer naming
    check_exits = process_exits
