import csv
import json
import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from data.loader import fetch_single_symbol, fetch_data_pack
from data.indices import get_index_symbols
from data.schwab_client import sd
from execution.engine import (
    _compute_indicators,
    MIN_BARS,
    MIN_ENTRY_SCORE,
    DEFAULT_SCORING_WEIGHTS,
    calculate_backtest_quality_score,
    get_sector,
)
from strategies.generic import GenericStrategy
from strategies.strategy_loader import load_strategies

PORTFOLIO_FILE = "data/paper_portfolio.json"
TRADE_LEDGER_FILE = "data/sim_trade_history.csv"
TRADE_LEDGER_HEADERS = [
    "Symbol",
    "Strategy",
    "Entry_Date",
    "Exit_Date",
    "Entry_Price",
    "Exit_Price",
    "Shares",
    "PnL_$",
    "PnL_%",
    "Reason",
]
DEFAULT_CONFIG_PATH = "config/generated_strategies.json"
POSITION_FRACTION = 0.20
MAX_POSITIONS = 5
SECTOR_CAP = 0.60
MAX_RISK_PER_TRADE = 0.02  # NEW: 2% equity risk per trade
MIN_REWARD_TO_RISK = 2.0   # NEW: Minimum 2:1 reward/risk ratio


class PaperTrader:
    def __init__(self, configs: Optional[List[Dict]] = None, start_cash: float = 100000.0):
        self.start_cash = start_cash
        self.scoring_weights = DEFAULT_SCORING_WEIGHTS
        
        # --- PHASE 3: FLYWEIGHT PATTERN (Shared Strategy Objects) ---
        self.strategy_configs = configs if configs is not None else self._load_default_configs()
        self.strategies = self._init_strategies(self.strategy_configs)
        self._strategy_cache = {s.name: s for s in self.strategies}
        
        self.sector_map = self._load_sector_map()
        self.state = self._load_state()
        self._ensure_trade_ledger()
        
        self._rehydrate_strategies()
        # Validate strategy params to avoid missing config values
        self._validate_configs()

    def _validate_configs(self):
        """Validate that all strategies have required parameters."""
        for strat in self.strategies:
            params = getattr(strat, "params", {})
            if not params.get("stop_loss_atr"):
                print(f"⚠️ WARNING: {strat.name} missing stop_loss_atr, using default 3.0")
                params["stop_loss_atr"] = 3.0
            if not params.get("time_stop"):
                print(f"⚠️ WARNING: {strat.name} missing time_stop, using default 60")
                params["time_stop"] = 60
            stop_atr = float(params.get("stop_loss_atr", 3.0))
            if stop_atr < 1.0 or stop_atr > 10.0:
                print(f"❌ ERROR: {strat.name} stop_loss_atr={stop_atr} out of range [1.0, 10.0]")

    def _ensure_trade_ledger(self) -> None:
        if os.path.exists(TRADE_LEDGER_FILE):
            return
        os.makedirs(os.path.dirname(TRADE_LEDGER_FILE), exist_ok=True)
        with open(TRADE_LEDGER_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRADE_LEDGER_HEADERS)

    def _append_trade_ledger(
        self,
        symbol: str,
        strategy: str,
        entry_date: str,
        exit_date: str,
        entry_price: float,
        exit_price: float,
        shares: int,
        pnl: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        self._ensure_trade_ledger()
        with open(TRADE_LEDGER_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    symbol,
                    strategy,
                    entry_date,
                    exit_date,
                    entry_price,
                    exit_price,
                    shares,
                    pnl,
                    pnl_pct,
                    reason,
                ]
            )

    # --- INITIALIZATION HELPERS ---
    def _init_strategies(self, configs: Optional[List[Dict]]) -> List[GenericStrategy]:
        strategies = load_strategies(configs or [])
        if not strategies:
            strategies = [GenericStrategy({"name": "Generic", "type": "generic"})]
        return strategies

    def _load_default_configs(self) -> List[Dict]:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r") as f:
                    return json.load(f)
            except: 
                return []
        return []

    def _migrate_position_genomes(self, state: Dict) -> None:
        if not isinstance(state, dict):
            return

        strategy_map: Dict[str, Dict] = {}
        if os.path.exists(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r") as f:
                    configs = json.load(f)
                for cfg in configs or []:
                    name = cfg.get("name")
                    if name:
                        strategy_map[str(name)] = cfg
            except Exception:
                strategy_map = {}

        if not strategy_map:
            return

        def _normalize(name: str) -> str:
            return "".join(ch for ch in str(name).lower() if ch.isalnum())

        for pos in (state.get("positions") or {}).values():
            if not isinstance(pos, dict):
                continue
            if "genome" in pos:
                continue
            s_name = pos.get("strategy_name") or pos.get("strategy") or ""
            match = strategy_map.get(s_name)
            if match is None and s_name:
                target = _normalize(s_name)
                best_key = ""
                for key, cfg in strategy_map.items():
                    key_norm = _normalize(key)
                    if not key_norm:
                        continue
                    if key_norm in target or target in key_norm:
                        if len(key_norm) > len(best_key):
                            best_key = key_norm
                            match = cfg
            if match is not None:
                pos["genome"] = match

    def _load_sector_map(self) -> Dict[str, str]:
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config/sectors.json"))
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _resolve_sector(self, sym: str) -> str:
        sym_up = (sym or "").upper()
        if sym_up in self.sector_map:
            return self.sector_map[sym_up]
        return get_sector(sym_up)

    def _get_strategy_by_name(self, strategy_name: str) -> GenericStrategy:
        if not strategy_name:
            return self.strategies[0]
        if strategy_name in self._strategy_cache:
            return self._strategy_cache[strategy_name]
        for name, strat in self._strategy_cache.items():
            if strategy_name in name or name in strategy_name:
                return strat
        return self.strategies[0]

    def _rehydrate_strategies(self):
        for sym, pos in self.portfolio.items():
            s_name = pos.get("strategy_name") or pos.get("strategy")
            pos["strategy_obj"] = self._get_strategy_by_name(s_name)
        for order in self.state.get("pending_orders", []):
            s_name = order.get("strategy_name") or order.get("strategy")
            order["strategy_obj"] = self._get_strategy_by_name(s_name)

    # --- STATE MANAGEMENT ---
    @property
    def portfolio(self) -> Dict:
        return self.state.setdefault("positions", {})

    def _load_state(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, "r") as f:
                    state = json.load(f)
                    if "pending_orders" not in state: state["pending_orders"] = []
                    if "history" not in state: state["history"] = []
                    if "equity_curve" not in state: 
                        state["equity_curve"] = [{"date": str(datetime.now().date()), "equity": self.start_cash}]
                    self._migrate_position_genomes(state)
                    return state
            except: pass
            
        return {
            "cash": self.start_cash,
            "equity": self.start_cash,
            "positions": {},
            "pending_orders": [],
            "history": [],
            "equity_curve": [{"date": str(datetime.now().date()), "equity": self.start_cash}],
        }

    def save_state(self):
        clean_state = {
            "cash": self.state["cash"],
            "equity": self.state["equity"],
            "history": self.state["history"],
            "equity_curve": self.state["equity_curve"],
            "positions": {},
            "pending_orders": []
        }
        for sym, pos in self.portfolio.items():
            clean_pos = pos.copy()
            if "strategy_obj" in clean_pos: del clean_pos["strategy_obj"]
            clean_state["positions"][sym] = clean_pos
        for order in self.state.get("pending_orders", []):
            clean_order = order.copy()
            if "strategy_obj" in clean_order: del clean_order["strategy_obj"]
            clean_state["pending_orders"].append(clean_order)

        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(clean_state, f, indent=4)

    def reset_account(self):
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)
        self.__init__(configs=self.strategy_configs, start_cash=self.start_cash)

    # --- CORE TRADING LOGIC ---
    def get_realtime_price(self, sym):
        try:
            q = sd.get_quote(sym)
            if q and sym in q and "quote" in q[sym]:
                q_d = q[sym]["quote"]
                return float(q_d.get("lastPrice") or q_d.get("mark") or q_d.get("closePrice") or 0.0)
        except: pass

        df = fetch_single_symbol(sym, days=5, force_fresh=False)
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
        return 0.0

    def buy(self, symbol: str, price: float, shares: int, stop_price: Optional[float] = None, strategy_obj=None, strategy_name: Optional[str] = None, entry_index: Optional[int] = None, genome: Optional[Dict] = None) -> Optional[Dict]:
        """
        PHASE 3: Queues a Market-On-Open (MOO) order.
        Checks 'Reserved Cash' to prevent overdrafts.
        """
        if shares <= 0 or price <= 0:
            return None
        
        reserved_cash = sum(o.get('committed_cash', 0) for o in self.state.get('pending_orders', []))
        available_cash = self.state["cash"] - reserved_cash
        
        estimated_cost = shares * price
        
        if available_cash < estimated_cost:
            print(f"❌ REJECTED {symbol}: Insufficient funds (Need ${estimated_cost:,.0f}, Avail ${available_cash:,.0f})")
            return None

        strat_obj = strategy_obj or self._get_strategy_by_name(strategy_name or "")
        strat_name = strategy_name or getattr(strat_obj, "name", "Unknown")
        stop_val = stop_price if stop_price is not None else price * 0.9
        strat_genome = genome
        if strat_genome is None and strat_obj is not None:
            strat_genome = getattr(strat_obj, "params", None) or getattr(strat_obj, "genome", None)
        if isinstance(strat_genome, dict):
            strat_genome = dict(strat_genome)
        else:
            strat_genome = None

        order = {
            "symbol": symbol,
            "shares": shares,
            "committed_cash": estimated_cost,
            "order_price_estimate": price,
            "stop_price": stop_val,
            "strategy": strat_name,
            "strategy_name": strat_name,
            "genome": strat_genome,
            "date": str(datetime.now().date()),
            "type": "BUY_MOO",
            "status": "PENDING",
            "entry_i": entry_index,
            "queued_at": datetime.now().isoformat()
        }
        
        self.state.setdefault("pending_orders", []).append(order)
        self.save_state()
        return order

    def process_pending_orders(self) -> List[str]:
        """
        Executes pending MOO orders ONLY if market is open (9:30 AM ET).
        FIX: Recalculates Stop Loss based on ACTUAL fill price.
        """
        fill_log = []
        remaining_orders = []
        
        if not self.state.get("pending_orders"):
            return []

        # --- TIMEZONE SETUP (New York) ---
        try:
            ny_tz = pytz.timezone('America/New_York')
            now_ny = datetime.now(ny_tz)
        except ImportError:
            now_ny = datetime.now()
            print("⚠️ pytz not found, using server time")

        today_ny = now_ny.date()
        market_open_time = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        grace_deadline = market_open_time + timedelta(minutes=5)
        is_market_open = now_ny >= market_open_time
        after_grace = now_ny >= grace_deadline

        print(f"🔔 Processing {len(self.state['pending_orders'])} pending orders...")
        print(f"📍 NY Time: {now_ny.strftime('%Y-%m-%d %H:%M:%S')} | Market Open: {is_market_open}")

        for order in self.state.get("pending_orders", []):
            sym = order["symbol"]
            est_price = order.get("order_price_estimate", 0)
            fill_price = 0.0
            
            # 1. Get Fill Price
            quote_payload = None
            quote_has_price = False
            try:
                q = sd.get_quote(sym)
                if q and sym in q and "quote" in q[sym]:
                    quote_payload = q[sym]["quote"]
            except: pass
            
            if quote_payload:
                open_price = float(quote_payload.get("openPrice") or 0)
                last_price = float(quote_payload.get("lastPrice") or 0)
                mark_price = float(quote_payload.get("markPrice") or quote_payload.get("mark") or 0)
                quote_has_price = any(p > 0 for p in (open_price, last_price, mark_price))

                if open_price > 0:
                    fill_price = open_price
                elif after_grace and last_price > 0:
                    fill_price = last_price
                elif last_price > 0:
                    fill_price = last_price
                elif mark_price > 0:
                    fill_price = mark_price

            # Tier 3 Fallback: historical open only if quote is empty
            if fill_price == 0 and not quote_has_price:
                df = fetch_single_symbol(sym, days=5, force_fresh=True)
                if df is not None and not df.empty:
                    last_dt = df.index[-1]
                    # Simple date check
                    if last_dt.date() == today_ny:
                        fill_price = float(df.iloc[-1]["open"])

            # WAITING logic
            if fill_price <= 0:
                if not is_market_open:
                    reason = "Market not open"
                elif not after_grace:
                    reason = "Waiting for official open price"
                else:
                    reason = "No usable price after grace"
                fill_log.append(f"⏳ WAITING {sym}: {reason}")
                remaining_orders.append(order)
                continue

            # 2. Standardized Gap Protection (prev close vs fill)
            df_hist = fetch_single_symbol(sym, days=60, force_fresh=True)
            prev_close = None
            if df_hist is not None and not df_hist.empty:
                df_hist = df_hist.sort_index()
                if len(df_hist) >= 2:
                    prev_close = float(df_hist.iloc[-2].get("close", 0) or 0)

            if prev_close and prev_close > 0:
                gap_pct_close = (fill_price - prev_close) / prev_close
                if gap_pct_close < -0.08:
                    fill_log.append(f"❌ CANCELED {sym}: Gap {gap_pct_close*100:+.1f}% below -8.0% (prev close vs open)")
                    continue

            # 3. Cost Check
            shares = order["shares"]
            actual_cost = shares * fill_price
            if self.state["cash"] < actual_cost:
                fill_log.append(f"❌ FAILED {sym}: Insufficient cash at fill")
                continue

            # 4. Execute Fill
            self.state["cash"] -= actual_cost
            
            s_name = order.get("strategy_name") or order.get("strategy")
            s_obj = self._get_strategy_by_name(s_name)
            
            # --- SAFETY FIX: RECALCULATE STOP LOSS ---
            # Don't use the 'stop_price' from the order (based on est_price).
            # Recalculate using the ACTUAL fill price and the strategy's multiplier.
            try:
                df_ind = _compute_indicators(df_hist.copy()) if df_hist is not None else None
                atr_idx = -2 if df_ind is not None and len(df_ind) >= 2 else -1
                current_atr = float(df_ind.iloc[atr_idx].get("atr14", fill_price * 0.02)) if df_ind is not None else fill_price * 0.02

                est_price = order.get("order_price_estimate", fill_price)
                stop_mult = float(s_obj.params.get("stop_loss_atr", 3.0))

                original_stop_distance = (current_atr * stop_mult) / est_price if est_price > 0 else 0.08
                real_stop_price = fill_price * (1 - original_stop_distance)

                gap_pct_local = abs((fill_price - est_price) / est_price) if est_price > 0 else 0
                if gap_pct_local > 0.03 and fill_price < est_price:
                    max_loss_pct = original_stop_distance * 1.2
                    real_stop_price = max(real_stop_price, fill_price * (1 - max_loss_pct))
                    
            except Exception as e:  # noqa: E722
                real_stop_price = fill_price * 0.92
                print(f"⚠️ Stop calculation failed for {sym}: {e}")

            order_genome = order.get("genome")
            if order_genome is None and s_obj is not None:
                order_genome = getattr(s_obj, "params", None) or getattr(s_obj, "genome", None)
            if isinstance(order_genome, dict):
                order_genome = dict(order_genome)
            else:
                order_genome = None
            
            self.portfolio[sym] = {
                "shares": shares,
                "entry_price": fill_price,
                "stop_price": real_stop_price, # Updated Stop
                "strategy": s_name,
                "strategy_name": s_name,
                "genome": order_genome,
                "strategy_obj": s_obj,
                "date": str(today_ny),
                "current_price": fill_price,
                "unrealized_pnl": 0.0,
                "unrealized_pct": 0.0,
                "entry_i": order.get("entry_i"),
            }
            
            fill_log.append(f"✅ FILLED {sym} @ ${fill_price:.2f} (Stop: ${real_stop_price:.2f})")

        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return fill_log

    def cancel_pending_order(self, symbol: str) -> List[str]:
        if not symbol:
            return ["⚠️ CANCEL FAILED: Invalid symbol"]
        pending = self.state.get("pending_orders", [])
        remaining = [o for o in pending if o.get("symbol") != symbol]
        if len(remaining) == len(pending):
            return [f"⚠️ CANCEL FAILED: {symbol} not found"]
        self.state["pending_orders"] = remaining
        self.save_state()
        return [f"🗑️ CANCELLED: {symbol} order removed"]

    def _current_sector_exposure(self) -> Dict[str, float]:
        exposure = {}
        for sym, pos in self.portfolio.items():
            price = pos.get("current_price", pos.get("entry_price", 0))
            val = pos.get("shares", 0) * price
            sec = self._resolve_sector(sym)
            exposure[sec] = exposure.get(sec, 0.0) + val
        for order in self.state.get("pending_orders", []):
            val = order.get("committed_cash", 0)
            sec = self._resolve_sector(order["symbol"])
            exposure[sec] = exposure.get(sec, 0.0) + val
        return exposure

    def update_valuations(self):
        fill_logs = [] 
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
        return self.state, fill_logs

    def close_position(self, symbol, reason="Manual"):
        if symbol not in self.portfolio: return False, "Position not found"
        pos = self.portfolio[symbol]
        exit_price = self.get_realtime_price(symbol)
        if exit_price <= 0: return False, "Could not fetch valid price"

        proceeds = exit_price * pos["shares"]
        pnl = proceeds - (pos["entry_price"] * pos["shares"])
        pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100

        try:
            self._append_trade_ledger(
                symbol=symbol,
                strategy=pos.get("strategy_name", ""),
                entry_date=pos.get("date", ""),
                exit_date=str(datetime.now().date()),
                entry_price=pos["entry_price"],
                exit_price=exit_price,
                shares=pos["shares"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                reason=reason,
            )
        except Exception as e:
            print(f"⚠️ Ledger append failed for {symbol}: {e}")

        self.state["cash"] += proceeds
        self.state.setdefault("history", []).append({
            "symbol": symbol, "strategy": pos.get("strategy_name", ""),
            "type": "SELL", "reason": reason, "entry_date": pos["date"],
            "exit_date": str(datetime.now().date()), "entry_price": pos["entry_price"],
            "exit_price": exit_price, "shares": pos["shares"], "pnl": pnl, "return_pct": pnl_pct,
        })
        del self.portfolio[symbol]
        self.update_valuations()
        return True, f"Sold {symbol} at ${exit_price:.2f} (PnL: ${pnl:.2f})"

    def _normalize_candidate(self, cand: Dict) -> Optional[Dict]:
        sym = cand.get("symbol") or cand.get("Symbol")
        if not sym: return None
        price = cand.get("price") or cand.get("Price")
        if price is None: price = self.get_realtime_price(sym)
        try: price = float(price)
        except: return None
        if price <= 0: return None
        stop = cand.get("stop") or cand.get("Stop")
        strategy_name = cand.get("strategy_name") or cand.get("strategy") or cand.get("Strategy")
        strat_obj = self._get_strategy_by_name(strategy_name)
        genome = cand.get("genome")
        if genome is None and strat_obj is not None:
            genome = getattr(strat_obj, "params", None) or getattr(strat_obj, "genome", None)
        if isinstance(genome, dict):
            genome = dict(genome)
        else:
            genome = None
        score = cand.get("score")
        if score is None:
            score = cand.get("Raw_Score", 50)
            if "wealth" in strat_obj.name.lower(): score *= 1.3
        else:
            try: score = float(score)
            except: score = 50
        return {
            "symbol": sym, "price": price, "stop": stop if stop is not None else price * 0.9,
            "strategy_name": strat_obj.name, "strategy_obj": strat_obj, "score": score, "genome": genome,
            "entry_i": cand.get("entry_i"),
        }

    def _execute_governor(self, candidates: List[Dict]) -> Dict[str, Any]:
        assert MIN_ENTRY_SCORE == 120.0
        logs: List[str] = []
        orders: List[Dict] = []
        normalized = []
        for cand in candidates:
            norm = self._normalize_candidate(cand)
            if norm: normalized.append(norm)

        normalized.sort(key=lambda x: x.get("score", 0), reverse=True)
        sector_exposure = self._current_sector_exposure()

        pending_committed = sum(o.get("committed_cash", 0) for o in self.state.get("pending_orders", []))
        available_cash = self.state["cash"] - pending_committed

        for cand in normalized:
            if len(self.portfolio) + len(self.state.get("pending_orders", [])) >= MAX_POSITIONS:
                logs.append(f"⚠️ REJECTED {cand['symbol']}: All {MAX_POSITIONS} slots full")
                continue
            if cand["symbol"] in self.portfolio:
                logs.append(f"ℹ️ SKIPPED {cand['symbol']}: Already held")
                continue
            if any(o['symbol'] == cand['symbol'] for o in self.state.get("pending_orders", [])):
                logs.append(f"ℹ️ SKIPPED {cand['symbol']}: Already pending")
                continue

            current_equity = self.state["cash"] + sum(sector_exposure.values())
            trade_val = current_equity * POSITION_FRACTION
            if trade_val <= 0 or available_cash < trade_val:
                logs.append(
                    f"⚠️ REJECTED {cand['symbol']}: Insufficient cash "
                    f"(${available_cash:,.2f} < ${trade_val:,.2f})"
                )
                continue

            # Calculate shares based on RISK, not position size
            risk_per_trade = current_equity * MAX_RISK_PER_TRADE
            price = cand["price"]
            stop_price = cand["stop"]
            risk_per_share = price - stop_price
            if risk_per_share <= 0:
                logs.append(f"⚠️ REJECTED {cand['symbol']}: Invalid stop (stop >= price)")
                continue

            # SAFETY CHECK 1: Minimum risk distance (at least 1% of price)
            min_risk_distance = price * 0.01
            if risk_per_share < min_risk_distance:
                risk_pct = (risk_per_share / price) * 100 if price > 0 else 0
                logs.append(f"⚠️ REJECTED {cand['symbol']}: Stop too tight ({risk_pct:.2f}% < 1.0% minimum)")
                continue

            # SAFETY CHECK 2: Maximum risk distance (no more than 20% of price)
            max_risk_distance = price * 0.20
            if risk_per_share > max_risk_distance:
                logs.append(f"⚠️ REJECTED {cand['symbol']}: Stop too wide ({risk_per_share/price*100:.2f}% > 20% maximum)")
                continue

            # Calculate shares
            shares = int(risk_per_trade / risk_per_share)

            # SAFETY CHECK 3: Absolute maximum shares per position
            MAX_SHARES_PER_POSITION = 1000
            if shares > MAX_SHARES_PER_POSITION:
                logs.append(f"ℹ️ CAPPED {cand['symbol']}: Reduced from {shares} to {MAX_SHARES_PER_POSITION} shares (safety limit)")
                shares = MAX_SHARES_PER_POSITION

            # SAFETY CHECK 4: Cap total position value at 20% equity
            max_shares_by_value = int((current_equity * POSITION_FRACTION) / price)
            shares = min(shares, max_shares_by_value)

            # SAFETY CHECK 5: Minimum viable position
            if shares < 1:
                logs.append(f"⚠️ REJECTED {cand['symbol']}: Position too small (< 1 share)")
                continue

            # Now compute position value and sector exposure after sizing
            position_val = shares * price
            sec = self._resolve_sector(cand["symbol"])
            projected_exp = (sector_exposure.get(sec, 0.0) + position_val) / current_equity if current_equity > 0 else 1.0

            if projected_exp > SECTOR_CAP:
                logs.append(f"⚠️ REJECTED {cand['symbol']}: Sector {sec} would be {projected_exp*100:.0f}% (limit: {SECTOR_CAP*100:.0f}%)")
                continue

            order = self.buy(
                cand["symbol"], cand["price"], shares, stop_price=cand["stop"],
                strategy_obj=cand["strategy_obj"], strategy_name=cand["strategy_name"],
                entry_index=cand.get("entry_i"),
                genome=cand.get("genome"),
            )
            if order:
                sector_exposure[sec] = sector_exposure.get(sec, 0.0) + position_val
                available_cash -= position_val
                orders.append(order)
                logs.append(f"⏳ QUEUED {cand['symbol']} x{shares} @ ${cand['price']:.2f} (Stop: ${cand['stop']:.2f})")
        return {"count": len(orders), "orders": orders, "logs": logs}

    def run_daily_scan(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, global_data: Optional[Dict[str, pd.DataFrame]] = None, scoring_weights: Optional[Dict] = None, progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
        # PHASE 3 FIX: REAL-TIME SCANNING (Scan Today, Trade Tomorrow)
        assert MIN_ENTRY_SCORE == 120.0
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
        scan_logs: List[str] = []
        total_steps = (len(self.strategies) * len(data_dict)) if data_dict else 0
        processed = 0
        for strat in self.strategies:
            strat_genome = getattr(strat, "params", None) or getattr(strat, "genome", None)
            if isinstance(strat_genome, dict):
                strat_genome = dict(strat_genome)
            else:
                strat_genome = None
            for sym, df in data_dict.items():
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_steps)
                if df is None or df.empty: continue
                try:
                    df_ind = _compute_indicators(df.copy(), spy_df=spy_df)
                    # DIAGNOSTIC TRACE
                    last_row = df_ind.iloc[-1]
                    print(
                        f"DEBUG: {sym} | Close: {last_row['close']:.2f} | "
                        f"SMA200: {last_row['sma200']:.2f} | RSI2: {last_row['rsi2']:.2f}"
                    )
                    if vix_df is not None: df_ind["vix"] = vix_df["close"].reindex(df_ind.index).ffill().fillna(20.0)
                    else: df_ind["vix"] = 20.0
                    
                    if len(df_ind) <= MIN_BARS: continue

                    # FIX: SCAN LIVE BAR (matching app.py live mark alignment)
                    signal_idx = len(df_ind) - 1
                    if signal_idx < MIN_BARS:
                        continue
                    
                    if not strat.entry(df_ind, signal_idx): continue

                    # Use signal bar data for baseline
                    row_signal = df_ind.iloc[signal_idx]
                    raw_score = calculate_backtest_quality_score(row_signal, strat.name, weights=scoring)
                    score = raw_score * 1.3 if "wealth" in strat.name.lower() else raw_score
                    if score < MIN_ENTRY_SCORE:
                        scan_logs.append(
                            f"⚠️ REJECTED {sym}: Low score {score:.1f} < {MIN_ENTRY_SCORE:.1f}"
                        )
                        continue

                    # Store BOTH close AND open from the signal bar
                    signal_close = float(row_signal["close"])
                    signal_open = float(row_signal["open"])

                    # Stop Loss (Estimation only - Recalculated on fill)
                    atr = float(row_signal.get("atr14", signal_close * 0.02))
                    stop_mult = float(getattr(strat, "params", {}).get("stop_loss_atr", 3.0))
                    stop_price = signal_close - (atr * stop_mult)

                    candidates.append({
                        "symbol": sym,
                        "price": signal_close,
                        "signal_open": signal_open,
                        "stop": stop_price,
                        "strategy_name": strat.name,
                        "strategy_obj": strat,
                        "genome": strat_genome,
                        "score": score,
                        "entry_i": signal_idx + 1
                    })
                except: continue

        result = self._execute_governor(candidates)
        logs = scan_logs + (result.get("logs") or [])
        return {"count": result.get("count", 0), "orders": result.get("orders", []), "logs": logs}

    def execute_entries(self, candidates: List[Dict]) -> Dict[str, Any]:
        return self._execute_governor(candidates)

    def process_exits(self, strategies_map=None):
        exits = []
        for sym, pos in list(self.portfolio.items()):
            strat_obj = pos.get("strategy_obj") or self._get_strategy_by_name(pos.get("strategy_name"))
            if strat_obj is None: strat_obj = self.strategies[0]
            pos["strategy_obj"] = strat_obj # Ensure attached

            df = fetch_single_symbol(sym, days=200, force_fresh=False)
            if df is None or df.empty: continue

            df = _compute_indicators(df)
            current_idx = len(df) - 1
            if current_idx < 0: continue

            entry_i = pos.get("entry_i")
            if entry_i is None:
                try:
                    entry_dt = pd.to_datetime(pos["date"]).date()
                    today_dt = datetime.now().date()
                    days_held = (today_dt - entry_dt).days
                    entry_i = max(0, current_idx - days_held)
                except: entry_i = current_idx

            stop_price = pos.get("stop_price", pos.get("entry_price", 0) * 0.9)
            if strat_obj.exit(df, current_idx, entry_i, pos["entry_price"], stop_price):
                exit_price = float(df.iloc[-1]["close"])
                if df.iloc[-1]["low"] < stop_price: exit_price = stop_price

                proceeds = exit_price * pos["shares"]
                pnl = proceeds - (pos["entry_price"] * pos["shares"])
                pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

                try:
                    self._append_trade_ledger(
                        symbol=sym,
                        strategy=pos.get("strategy_name", strat_obj.name),
                        entry_date=pos.get("date", ""),
                        exit_date=str(datetime.now().date()),
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        shares=pos["shares"],
                        pnl=pnl,
                        pnl_pct=pct,
                        reason="Signal/Stop",
                    )
                except Exception as e:
                    print(f"⚠️ Ledger append failed for {sym}: {e}")

                self.state["cash"] += proceeds
                self.state["history"].append({
                    "symbol": sym, "strategy": pos.get("strategy_name", strat_obj.name),
                    "type": "AUTO_EXIT", "reason": "Signal/Stop",
                    "entry_date": pos["date"], "exit_date": str(datetime.now().date()),
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "return_pct": pct,
                })
                del self.portfolio[sym]
                exits.append(f"SOLD {sym} at ${exit_price:.2f} ({pnl:.2f})")

        self.save_state()
        return exits
    
    check_exits = process_exits
