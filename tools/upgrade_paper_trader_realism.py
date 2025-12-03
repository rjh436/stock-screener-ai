"""Upgrade paper trader realism with pending MOO orders and super-signal handling."""

import os


def upgrade_paper_trader_realism():
    print("🎮 UPGRADING SIMULATOR (Realism + Super Signals)...")

    path = "simulation/paper_trader.py"

    code = """
import json
import os
import pandas as pd
from datetime import datetime
from data.loader import fetch_single_symbol
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
                # Ensure backward compatibility
                if "pending_orders" not in state:
                    state["pending_orders"] = []
                return state
        return {
            "cash": self.start_cash,
            "equity": self.start_cash,
            "positions": {},  # Active Holdings
            "pending_orders": [], # Orders waiting for Market Open
            "history": [],    # Closed trades
            "equity_curve": [{"date": str(datetime.now().date()), "equity": self.start_cash}]
        }

    def save_state(self):
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(self.state, f, indent=4)

    def reset_account(self):
        if os.path.exists(PORTFOLIO_FILE):
            os.remove(PORTFOLIO_FILE)
        self.__init__(self.start_cash)

    def process_pending_orders(self):
        \"\"\"Checks if we can fill pending 'Market On Open' orders.\"\"\"
        filled_log = []
        remaining_orders = []
        
        today_str = str(datetime.now().date())
        
        for order in self.state.get("pending_orders", []):
            sym = order["symbol"]
            order_date = order["date"]
            
            # fetch fresh data to see if a new bar exists
            df = fetch_single_symbol(sym, days=5, force_fresh=True)
            if df is None or df.empty:
                remaining_orders.append(order)
                continue
                
            # Check if we have data NEWER than the order date
            last_dt_str = str(df.index[-1].date())
            
            if last_dt_str > order_date:
                # WE HAVE A NEW BAR! EXECUTE AT OPEN.
                fill_price = float(df.iloc[-1]["open"])
                
                # Calculate shares based on the cash committed
                committed_cash = order["committed_cash"]
                shares = int(committed_cash / fill_price)
                
                if shares > 0:
                    cost = shares * fill_price
                    
                    if self.state["cash"] >= cost:
                        self.state["cash"] -= cost
                        self.state["positions"][sym] = {
                            "shares": shares,
                            "entry_price": fill_price,
                            "stop_price": order["stop_price"],
                            "strategy": order["strategy"],
                            "date": last_dt_str,
                            "current_price": fill_price,
                            "unrealized_pnl": 0.0,
                            "unrealized_pct": 0.0
                        }
                        filled_log.append(f"✅ FILLED {sym} at ${fill_price:.2f} (MOO)")
                    else:
                        filled_log.append(f"❌ FAILED {sym}: Insufficient Cash at Fill")
            else:
                # Still waiting for next day
                remaining_orders.append(order)
        
        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return filled_log

    def update_valuations(self):
        \"\"\"Updates prices AND checks for pending fills.\"\"\"
        # 1. Try to fill pending orders first
        fill_logs = self.process_pending_orders()
        
        # 2. Update Active Positions
        total_value = self.state["cash"]
        
        # Note: We do not count 'pending_orders' committed cash as equity until filled
        # to avoid double counting or complexity, but it remains in 'cash'.

        for sym, pos in self.state["positions"].items():
            df = fetch_single_symbol(sym, days=30, force_fresh=True)
            if df is not None and not df.empty:
                current_price = float(df.iloc[-1]["close"])
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
            strat_name = pos["strategy"]
            
            # Handle potential merged names like "Strategy A + Strategy B"
            # We just pick the first one for exit logic, or default
            primary_strat = strat_name.split(" + ")[0]
            
            strat_config = strategies_map.get(primary_strat)
            if not strat_config: continue

            strat_logic = GenericStrategy(strat_config)
            df = fetch_single_symbol(sym, days=200, force_fresh=False)
            if df is None or df.empty: continue

            df = _compute_indicators(df)
            current_idx = len(df) - 1
            
            # Estimate Entry Index
            try:
                # Approximate days held
                entry_dt = pd.to_datetime(pos["date"]).date()
                today_dt = datetime.now().date()
                days_held = (today_dt - entry_dt).days
                entry_i = max(0, current_idx - days_held)
            except:
                entry_i = current_idx

            if strat_logic.exit(df, current_idx, entry_i, pos["entry_price"], pos["stop_price"]):
                exit_price = float(df.iloc[-1]["close"])
                if df.iloc[-1]["low"] < pos["stop_price"]:
                    exit_price = pos["stop_price"] 

                proceeds = exit_price * pos["shares"]
                pnl = proceeds - (pos["entry_price"] * pos["shares"])

                self.state["cash"] += proceeds
                self.state["history"].append({
                    "symbol": sym,
                    "strategy": strat_name,
                    "entry_date": pos["date"],
                    "exit_date": str(datetime.now().date()),
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
                })
                del self.state["positions"][sym]
                exits.append(f"SOLD {sym} at ${exit_price:.2f} ({pnl:.2f})")

        self.save_state()
        return exits

    def execute_entries(self, candidates):
        \"\"\"
        Queues 'Market On Open' orders. 
        Merges duplicates to create SUPER SIGNALS.
        \"\"\"
        logs = []
        
        # 1. MERGE DUPLICATES (Super Signal Logic)
        merged = {}
        for cand in candidates:
            sym = cand["Symbol"]
            score = cand.get("Raw_Score", 50)
            
            if sym in merged:
                # SUPER SIGNAL: Boost Score & Combine Names
                merged[sym]["Score"] += score + 1000 # Force to top
                merged[sym]["Strategy"] += " + " + cand["Strategy"]
                # Keep tighter stop? or average? Let's keep the first one (usually safer)
            else:
                cand["Score"] = score
                merged[sym] = cand
        
        final_list = list(merged.values())
        
        # 2. SORT
        final_list.sort(key=lambda x: x["Score"], reverse=True)

        max_pos = 5
        current_count = len(self.state["positions"]) + len(self.state["pending_orders"])
        target_size = self.state["equity"] * 0.20 

        for cand in final_list:
            if current_count >= max_pos: break
            if self.state["cash"] < target_size: break
            
            # Check if already owned or pending
            if cand["Symbol"] in self.state["positions"]: continue
            if any(o["symbol"] == cand["Symbol"] for o in self.state["pending_orders"]): continue

            # QUEUE ORDER
            self.state["pending_orders"].append({
                "symbol": cand["Symbol"],
                "committed_cash": target_size,
                "stop_price": cand["Stop"],
                "strategy": cand["Strategy"],
                "date": str(datetime.now().date()),
                "status": "PENDING_OPEN"
            })
            logs.append(f"⏳ QUEUED {cand['Symbol']} (Buy Next Open) - Score: {cand['Score']:.0f}")
            current_count += 1

        self.save_state()
        return logs
"""
    
    with open(path, "w") as f:
        f.write(code)
    print("   ✅ Simulator Upgraded: Next-Day Open Execution + Super Signal Priority.")


if __name__ == "__main__":
    upgrade_paper_trader_realism()
