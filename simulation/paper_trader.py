import json
import os
import pandas as pd
from datetime import datetime
from data.loader import fetch_single_symbol
from execution.engine import _compute_indicators, calculate_backtest_quality_score
from strategies.generic import GenericStrategy

PORTFOLIO_FILE = "data/paper_portfolio.json"


class PaperTrader:
    def __init__(self, start_cash=100000.0):
        self.start_cash = start_cash
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        return {
            "cash": self.start_cash,
            "equity": self.start_cash,
            "positions": {},  # Symbol -> {shares, entry_price, stop_price, strategy, date}
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

    def update_valuations(self):
        """Fetches current prices to update unrealized PnL and Total Equity."""
        total_value = self.state["cash"]

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
        if self.state["equity_curve"][-1]["date"] != today:
            self.state["equity_curve"].append({"date": today, "equity": total_value})

        self.save_state()
        return self.state

    def process_exits(self, strategies_map):
        """Checks held positions against their strategy's exit logic."""
        exits = []
        for sym, pos in list(self.state["positions"].items()):
            strat_name = pos["strategy"]
            if strat_name not in strategies_map:
                continue

            strat_logic = GenericStrategy(strategies_map[strat_name])
            df = fetch_single_symbol(sym, days=200, force_fresh=False)
            if df is None or df.empty:
                continue

            df = _compute_indicators(df)
            current_idx = len(df) - 1

            # entry_i set to 0 for simplified paper trading; time-based exits may be loose.
            if strat_logic.exit(df, current_idx, 0, pos["entry_price"], pos["stop_price"]):
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
        """Buys the top candidates if cash is available."""
        candidates.sort(key=lambda x: x["Score"], reverse=True)

        logs = []
        max_pos = 5
        target_size = self.state["equity"] * 0.20

        for cand in candidates:
            if len(self.state["positions"]) >= max_pos:
                break
            if self.state["cash"] < target_size:
                break
            if cand["Symbol"] in self.state["positions"]:
                continue

            price = cand["Price"]
            shares = int(target_size / price)
            cost = shares * price

            if shares > 0 and self.state["cash"] >= cost:
                self.state["cash"] -= cost
                self.state["positions"][cand["Symbol"]] = {
                    "shares": shares,
                    "entry_price": price,
                    "stop_price": cand["Stop"],
                    "strategy": cand["Strategy"],
                    "date": str(datetime.now().date()),
                    "current_price": price,
                    "unrealized_pnl": 0.0,
                    "unrealized_pct": 0.0
                }
                logs.append(f"BOUGHT {cand['Symbol']} at ${price:.2f} ({cand['Strategy']})")

        self.save_state()
        return logs
