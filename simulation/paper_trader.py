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
        
        # --- PHASE 3: FLYWEIGHT PATTERN (Shared Strategy Objects) ---
        self.strategy_configs = configs if configs is not None else self._load_default_configs()
        self.strategies = self._init_strategies(self.strategy_configs)
        # Create a lookup cache: Name -> Strategy Object
        self._strategy_cache = {s.name: s for s in self.strategies}
        
        self.sector_map = self._load_sector_map()
        self.state = self._load_state()
        
        # Rehydrate runtime references (attach strategy objects to positions)
        self._rehydrate_strategies()

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
        """
        Robust lookup that survives renames or missing configs.
        """
        if not strategy_name:
            return self.strategies[0]
            
        # 1. Exact Match
        if strategy_name in self._strategy_cache:
            return self._strategy_cache[strategy_name]
            
        # 2. Fuzzy/Partial Match (e.g. "Apex Wealth" -> "Apex Wealth (Gen 12)")
        for name, strat in self._strategy_cache.items():
            if strategy_name in name or name in strategy_name:
                return strat
                
        # 3. Fallback to default
        return self.strategies[0]

    def _rehydrate_strategies(self):
        """Re-attaches strategy objects to portfolio/pending after loading from JSON."""
        # 1. Portfolio
        for sym, pos in self.portfolio.items():
            s_name = pos.get("strategy_name") or pos.get("strategy")
            pos["strategy_obj"] = self._get_strategy_by_name(s_name)
            
        # 2. Pending Orders
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
        """
        Persists state to JSON. CRITICAL: Removes non-serializable objects (strategy_obj) before saving.
        """
        # Create a deep-ish copy to sanitize
        clean_state = {
            "cash": self.state["cash"],
            "equity": self.state["equity"],
            "history": self.state["history"],
            "equity_curve": self.state["equity_curve"],
            "positions": {},
            "pending_orders": []
        }

        # Sanitize Portfolio
        for sym, pos in self.portfolio.items():
            clean_pos = pos.copy()
            if "strategy_obj" in clean_pos: del clean_pos["strategy_obj"]
            clean_state["positions"][sym] = clean_pos

        # Sanitize Pending Orders
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

    def buy(self, symbol: str, price: float, shares: int, stop_price: Optional[float] = None, strategy_obj=None, strategy_name: Optional[str] = None, entry_index: Optional[int] = None):
        """
        PHASE 3: Queues a Market-On-Open (MOO) order.
        Checks 'Reserved Cash' to prevent overdrafts.
        """
        if shares <= 0 or price <= 0: return False
        
        # 1. Calculate Reserved Cash
        reserved_cash = sum(o.get('committed_cash', 0) for o in self.state.get('pending_orders', []))
        available_cash = self.state["cash"] - reserved_cash
        
        estimated_cost = shares * price
        
        # 2. Strict Funds Check
        if available_cash < estimated_cost:
            print(f"❌ REJECTED {symbol}: Insufficient funds (Need ${estimated_cost:,.0f}, Avail ${available_cash:,.0f})")
            return False

        strat_obj = strategy_obj or self._get_strategy_by_name(strategy_name or "")
        strat_name = strategy_name or getattr(strat_obj, "name", "Unknown")
        stop_val = stop_price if stop_price is not None else price * 0.9

        # 3. Create Queue Order (MOO)
        # Note: We do NOT store strategy_obj here to keep JSON clean. It's rehydrated by name later.
        order = {
            "symbol": symbol,
            "shares": shares,
            "committed_cash": estimated_cost,
            "order_price_estimate": price,
            "stop_price": stop_val,
            "strategy": strat_name,
            "strategy_name": strat_name,
            "date": str(datetime.now().date()),
            "type": "BUY_MOO",
            "status": "PENDING",
            "entry_i": entry_index,
            "queued_at": datetime.now().isoformat()
        }
        
        self.state.setdefault("pending_orders", []).append(order)
        self.save_state()
        return True

    def process_pending_orders(self) -> List[str]:
        """
        Executes pending MOO orders using current market data.
        Features Gap Protection (±5%) and actual cash debit.
        """
        fill_log = []
        remaining_orders = []
        
        # If nothing pending, return
        if not self.state.get("pending_orders"):
            return []

        print(f"🔔 Processing {len(self.state['pending_orders'])} pending orders...")

        for order in self.state.get("pending_orders", []):
            sym = order["symbol"]
            est_price = order.get("order_price_estimate", 0)
            
            # 1. Get Fill Price (Market Open logic)
            fill_price = 0.0
            
            # Try Schwab Quote first (Real-time)
            try:
                q = sd.get_quote(sym)
                if q and sym in q and "quote" in q[sym]:
                    fill_price = float(q[sym]["quote"].get("openPrice", 0) or q[sym]["quote"].get("lastPrice", 0))
            except: pass
            
            # Fallback to recent data
            if fill_price == 0:
                df = fetch_single_symbol(sym, days=5, force_fresh=True)
                if df is not None and not df.empty:
                    fill_price = float(df.iloc[-1]["open"]) # Use OPEN for MOO

            if fill_price <= 0:
                fill_log.append(f"❌ SKIPPED {sym}: No price data")
                remaining_orders.append(order)
                continue

            # 2. Gap Protection (±5%)
            gap_pct = ((fill_price - est_price) / est_price) * 100 if est_price > 0 else 0
            if abs(gap_pct) > 5.0:
                fill_log.append(f"❌ CANCELED {sym}: Gap too large ({gap_pct:+.1f}%)")
                continue # Do not fill, do not keep (Cancel)

            # 3. Cost Check
            shares = order["shares"]
            actual_cost = shares * fill_price
            
            if self.state["cash"] < actual_cost:
                fill_log.append(f"❌ FAILED {sym}: Insufficient cash at fill")
                continue # Cancel

            # 4. Execute Fill
            self.state["cash"] -= actual_cost
            
            # Rehydrate strategy for active position
            s_name = order.get("strategy_name") or order.get("strategy")
            s_obj = self._get_strategy_by_name(s_name)
            
            self.portfolio[sym] = {
                "shares": shares,
                "entry_price": fill_price,
                "stop_price": order["stop_price"],
                "strategy": s_name,
                "strategy_name": s_name,
                "strategy_obj": s_obj,
                "date": str(datetime.now().date()),
                "current_price": fill_price,
                "unrealized_pnl": 0.0,
                "unrealized_pct": 0.0,
                "entry_i": order.get("entry_i"),
            }
            
            fill_log.append(f"✅ FILLED {sym} @ ${fill_price:.2f} (Gap: {gap_pct:+.1f}%)")

        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return fill_log

    def _current_sector_exposure(self) -> Dict[str, float]:
        exposure = {}
        # Count Active Positions
        for sym, pos in self.portfolio.items():
            price = pos.get("current_price", pos.get("entry_price", 0))
            val = pos.get("shares", 0) * price
            sec = self._resolve_sector(sym)
            exposure[sec] = exposure.get(sec, 0.0) + val
            
        # Count Pending Orders (Reserved Capital)
        for order in self.state.get("pending_orders", []):
            val = order.get("committed_cash", 0)
            sec = self._resolve_sector(order["symbol"])
            exposure[sec] = exposure.get(sec, 0.0) + val
            
        return exposure

    def update_valuations(self):
        # We process orders explicitly via button now, but we can check if any are leftover
        # fill_logs = self.process_pending_orders() 
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

    # ... (Keep existing close_position, _execute_governor, run_daily_scan, process_exits) ...
    # CRITICAL: _execute_governor uses self.buy(), so it inherits the fixes automatically.
    # We just need to ensure the methods are preserved below.

    def close_position(self, symbol, reason="Manual"):
        if symbol not in self.portfolio: return False, "Position not found"
        pos = self.portfolio[symbol]
        exit_price = self.get_realtime_price(symbol)
        if exit_price <= 0: return False, "Could not fetch valid price"

        proceeds = exit_price * pos["shares"]
        pnl = proceeds - (pos["entry_price"] * pos["shares"])
        pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100

        self.state["cash"] += proceeds
        self.state["history"].append({
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
        score = cand.get("score")
        if score is None:
            score = cand.get("Raw_Score", 50)
            if "wealth" in strat_obj.name.lower(): score *= 1.3
        else:
            try: score = float(score)
            except: score = 50
        return {
            "symbol": sym, "price": price, "stop": stop if stop is not None else price * 0.9,
            "strategy_name": strat_obj.name, "strategy_obj": strat_obj, "score": score,
            "entry_i": cand.get("entry_i"),
        }

    def _execute_governor(self, candidates: List[Dict]) -> List[str]:
        logs = []
        normalized = []
        for cand in candidates:
            norm = self._normalize_candidate(cand)
            if norm: normalized.append(norm)

        normalized.sort(key=lambda x: x.get("score", 0), reverse=True)
        sector_exposure = self._current_sector_exposure()

        # Calculate committed cash from pending orders
        pending_committed = sum(o.get("committed_cash", 0) for o in self.state.get("pending_orders", []))
        available_cash = self.state["cash"] - pending_committed

        for cand in normalized:
            if len(self.portfolio) + len(self.state.get("pending_orders", [])) >= MAX_POSITIONS: break
            if cand["symbol"] in self.portfolio: continue
            if any(o['symbol'] == cand['symbol'] for o in self.state.get("pending_orders", [])): continue

            current_equity = self.state["cash"] + sum(sector_exposure.values())
            trade_val = current_equity * POSITION_FRACTION
            if trade_val <= 0 or available_cash < trade_val: continue

            sec = self._resolve_sector(cand["symbol"])
            projected_exp = (sector_exposure.get(sec, 0.0) + trade_val) / current_equity if current_equity > 0 else 1.0
            if projected_exp > SECTOR_CAP: continue

            shares = int(trade_val / cand["price"])
            if shares <= 0: continue

            success = self.buy(
                cand["symbol"], cand["price"], shares, stop_price=cand["stop"],
                strategy_obj=cand["strategy_obj"], strategy_name=cand["strategy_name"],
                entry_index=cand.get("entry_i"),
            )
            if success:
                sector_exposure[sec] = exposure_update = sector_exposure.get(sec, 0.0) + (shares * cand["price"])
                sector_exposure[sec] = exposure_update
                available_cash -= (shares * cand["price"])
                logs.append(f"⏳ QUEUED {cand['symbol']} x{shares} (Est. ${cand['price']:.2f})")
        return logs

    def run_daily_scan(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, global_data: Optional[Dict[str, pd.DataFrame]] = None, scoring_weights: Optional[Dict] = None) -> List[str]:
        # (Keeping original logic, just ensuring it calls _execute_governor which calls updated buy)
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
                if df is None or df.empty: continue
                try:
                    enriched = _compute_indicators(df.copy(), spy_df=spy_df)
                    if vix_df is not None: enriched["vix"] = vix_df["close"].reindex(enriched.index).ffill().fillna(20.0)
                    else: enriched["vix"] = 20.0
                    if len(enriched) <= MIN_BARS: continue

                    signal_idx = len(enriched) - 2
                    trade_idx = signal_idx + 1
                    if signal_idx < 0 or trade_idx <= 0: continue
                    if trade_idx < MIN_BARS: continue
                    if not strat.entry(enriched, signal_idx): continue

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

                    candidates.append({
                        "symbol": sym, "price": open_px, "stop": stop_price,
                        "strategy_name": strat.name, "strategy_obj": strat,
                        "score": score, "entry_i": trade_idx,
                    })
                except: continue

        return self._execute_governor(candidates)

    def execute_entries(self, candidates: List[Dict]) -> List[str]:
        return self._execute_governor(candidates)

    def process_exits(self, strategies_map=None):
        exits = []
        # Copy items to allow deletion during iteration
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
