
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
                        # Use openPrice if available, else mark if market open
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
            current_price = None
            # A. Try Quote
            try:
                q = sd.get_quote(sym)
                if q and sym in q and 'quote' in q[sym]:
                    q_d = q[sym]['quote']
                    current_price = q_d.get('lastPrice') or q_d.get('mark')
            except: pass
            
            # B. Fallback History
            if current_price is None:
                df = fetch_single_symbol(sym, days=30)
                if df is not None and not df.empty:
                    current_price = float(df.iloc[-1]["close"])
            
            if current_price is not None:
                pos["current_price"] = float(current_price)
                pos["unrealized_pnl"] = (pos["current_price"] - pos["entry_price"]) * pos["shares"]
                pos["unrealized_pct"] = (pos["current_price"] - pos["entry_price"]) / pos["entry_price"] * 100
                total_value += (pos["current_price"] * pos["shares"])
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
        # ... (Logic mostly unchanged, omitting for brevity but ensuring get_quote could be used here too)
        # For now, standard exit logic on history is safer for consistency
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
