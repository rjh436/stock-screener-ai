
import json
import os
import pandas as pd
from datetime import datetime
from data.loader import fetch_single_symbol
from data.schwab_client import sd
from execution.engine import _compute_indicators
from strategies.generic import GenericStrategy

PORTFOLIO_FILE = "data/paper_portfolio.json"

class PaperTrader:
    def __init__(self, start_cash=100000.0):
        self.start_cash = start_cash
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                state = json.load(f)
                if "pending_orders" not in state: state["pending_orders"] = []
                return state
        return {
            "cash": self.start_cash, "equity": self.start_cash,
            "positions": {}, "pending_orders": [], "history": [],
            "equity_curve": [{"date": str(datetime.now().date()), "equity": self.start_cash}]
        }

    def save_state(self):
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.state, f, indent=4)

    def reset_account(self):
        if os.path.exists(PORTFOLIO_FILE): os.remove(PORTFOLIO_FILE)
        self.__init__(self.start_cash)

    def get_realtime_price(self, sym):
        # Priority: Quote -> History -> 0
        try:
            q = sd.get_quote(sym)
            if q and sym in q and 'quote' in q[sym]:
                q_d = q[sym]['quote']
                return float(q_d.get('lastPrice') or q_d.get('mark') or q_d.get('closePrice') or 0.0)
        except: pass
        
        # Fallback
        df = fetch_single_symbol(sym, days=5, force_fresh=False)
        if df is not None and not df.empty:
            return float(df.iloc[-1]["close"])
        return 0.0

    def close_position(self, symbol, reason="Manual"):
        """Manually sells a position at the current real-time price."""
        if symbol not in self.state["positions"]: return False, "Position not found"
        
        pos = self.state["positions"][symbol]
        exit_price = self.get_realtime_price(symbol)
        
        if exit_price <= 0: return False, "Could not fetch valid price"
        
        proceeds = exit_price * pos["shares"]
        pnl = proceeds - (pos["entry_price"] * pos["shares"])
        pnl_pct = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
        
        self.state["cash"] += proceeds
        self.state["history"].append({
            "symbol": symbol,
            "strategy": pos["strategy"],
            "type": "SELL",
            "reason": reason,
            "entry_date": pos["date"],
            "exit_date": str(datetime.now().date()),
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": pos["shares"],
            "pnl": pnl,
            "return_pct": pnl_pct
        })
        
        del self.state["positions"][symbol]
        self.update_valuations() # Refresh equity
        self.save_state()
        return True, f"Sold {symbol} at ${exit_price:.2f} (PnL: ${pnl:.2f})"

    def process_pending_orders(self):
        filled_log = []
        remaining_orders = []
        
        for order in self.state.get("pending_orders", []):
            sym = order["symbol"]
            order_date = order["date"]
            fill_price = None
            fill_date = None
            
            # 1. Try History
            df = fetch_single_symbol(sym, days=5, force_fresh=True)
            if df is not None and not df.empty:
                last_dt = str(df.index[-1].date())
                if last_dt > order_date:
                    fill_price = float(df.iloc[-1]["open"])
                    fill_date = last_dt

            # 2. Try Real-Time Quote
            if fill_price is None:
                try:
                    q = sd.get_quote(sym)
                    if q and sym in q and 'quote' in q[sym]:
                        q_data = q[sym]['quote']
                        open_px = q_data.get('openPrice')
                        if open_px and open_px > 0:
                            fill_price = float(open_px)
                            fill_date = str(datetime.now().date())
                except: pass

            if fill_price and fill_date and fill_date > order_date:
                committed = order["committed_cash"]
                shares = int(committed / fill_price)
                if shares > 0 and self.state["cash"] >= (shares * fill_price):
                    self.state["cash"] -= (shares * fill_price)
                    self.state["positions"][sym] = {
                        "shares": shares, "entry_price": fill_price,
                        "stop_price": order["stop_price"], "strategy": order["strategy"],
                        "date": fill_date, "current_price": fill_price,
                        "unrealized_pnl": 0.0, "unrealized_pct": 0.0
                    }
                    filled_log.append(f"✅ FILLED {sym} at ${fill_price:.2f}")
                else: filled_log.append(f"❌ FAILED {sym}: Cash")
            else:
                remaining_orders.append(order)
        
        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return filled_log

    def update_valuations(self):
        fill_logs = self.process_pending_orders()
        total_value = self.state["cash"]
        
        for sym, pos in self.state["positions"].items():
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

    def process_exits(self, strategies_map):
        exits = []
        for sym, pos in list(self.state["positions"].items()):
            strat_name = pos["strategy"].split(" + ")[0]
            strat_config = strategies_map.get(strat_name)
            if not strat_config: continue

            strat_logic = GenericStrategy(strat_config)
            df = fetch_single_symbol(sym, days=200, force_fresh=False)
            if df is None or df.empty: continue

            df = _compute_indicators(df)
            current_idx = len(df) - 1
            
            try:
                entry_dt = pd.to_datetime(pos["date"]).date()
                today_dt = datetime.now().date()
                days_held = (today_dt - entry_dt).days
                entry_i = max(0, current_idx - days_held)
            except: entry_i = current_idx

            if strat_logic.exit(df, current_idx, entry_i, pos["entry_price"], pos["stop_price"]):
                exit_price = float(df.iloc[-1]["close"])
                if df.iloc[-1]["low"] < pos["stop_price"]: exit_price = pos["stop_price"]

                proceeds = exit_price * pos["shares"]
                pnl = proceeds - (pos["entry_price"] * pos["shares"])
                pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

                self.state["cash"] += proceeds
                self.state["history"].append({
                    "symbol": sym, "strategy": strat_name, "type": "AUTO_EXIT",
                    "reason": "Signal/Stop", "entry_date": pos["date"], 
                    "exit_date": str(datetime.now().date()),
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "return_pct": pct
                })
                del self.state["positions"][sym]
                exits.append(f"SOLD {sym} at ${exit_price:.2f} ({pnl:.2f})")

        self.save_state()
        return exits

    def execute_entries(self, candidates):
        logs = []
        merged = {}
        for cand in candidates:
            sym = cand["Symbol"]
            score = cand.get("Raw_Score", 50)
            if sym in merged:
                merged[sym]["Score"] += score + 1000
                merged[sym]["Strategy"] += " + " + cand["Strategy"]
            else:
                cand["Score"] = score
                merged[sym] = cand
        
        final_list = list(merged.values())
        final_list.sort(key=lambda x: x["Score"], reverse=True)
        
        max_pos = 5
        current_count = len(self.state["positions"]) + len(self.state["pending_orders"])
        target_size = self.state["equity"] * 0.20 

        for cand in final_list:
            if current_count >= max_pos: break
            if self.state["cash"] < target_size: break
            
            if cand["Symbol"] in self.state["positions"]: continue
            if any(o["symbol"] == cand["Symbol"] for o in self.state["pending_orders"]): continue

            self.state["pending_orders"].append({
                "symbol": cand["Symbol"], "committed_cash": target_size,
                "stop_price": cand["Stop"], "strategy": cand["Strategy"],
                "date": str(datetime.now().date()), "status": "PENDING_OPEN"
            })
            logs.append(f"⏳ QUEUED {cand['Symbol']}")
            current_count += 1
            
        self.save_state()
        return logs
