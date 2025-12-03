"""Add quote support and improve simulator pending fills."""

import os


def fix_simulator_fills():
    print("🔧 FIXING SIMULATOR FILLS (Adding Real-Time Quotes)...")

    # --- 1. UPGRADE SCHWAB CLIENT (Add Quotes) ---
    schwab_path = "data/schwab_client.py"
    schwab_code = """
import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load env vars
load_dotenv(override=True)

try:
    import streamlit as st
except ImportError:
    class DummySt:
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def stop(self): raise SystemExit("Streamlit Stop")
        def cache_data(self, **kwargs):
            def decorator(func): return func
            return decorator
    st = DummySt()

try:
    from schwab.auth import easy_client
except ImportError:
    pass 

def _first_env(*keys, default=""):
    for k in keys:
        v = os.getenv(k, "")
        if v and v.strip():
            return v.strip()
    return default

class SchwabData:
    def __init__(self):
        self.cid = _first_env("SCHWAB_CLIENT_ID", "SCHWAB_APP_KEY")
        self.ck = _first_env("SCHWAB_CLIENT_KEY", "SCHWAB_CLIENT_SECRET")
        self.redir = _first_env(
            "SCHWAB_REDIRECT_URI", "SCHWAB_CALLBACK_URL", "CALLBACK_URL", "REDIRECT_URI",
            default="http://127.0.0.1:8000",
        )
        self.creds = _first_env("SCHWAB_CRED_PATH", default="data/.schwab_creds.json")
        self.port = int(_first_env("SCHWAB_PORT", default="8182"))
        os.makedirs("data", exist_ok=True)

        self._cli = None
        self._tok = "data/schwab_tokens.json"
        self.signature_used = None

    def _ensure(self):
        if self._cli is not None:
            return
        errors = []
        def _try(fn, label):
            try:
                cli = fn()
                self.signature_used = label
                return cli
            except Exception as e:
                errors.append(f"{label}: {e}")
                return None

        self._cli = _try(
            lambda: easy_client(
                api_key=self.cid, client_secret=self.ck, redirect_uri=self.redir,
                credentials_path=self.creds, token_path=self._tok,
                make_webdriver=lambda: None, headless=True, port=self.port,
            ), "v1:new-keywords")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(
                    app_key=self.cid, app_secret=self.ck, callback_url=self.redir,
                    creds_path=self.creds), "v2:classic-kw")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(self.cid, self.ck, self.redir, self.creds),
                "v4:positional")

        if self._cli is None:
            raise RuntimeError(f"Schwab Auth Failed. Errors: {errors}")

    def price_daily(self, symbol, start_datetime=None, end_datetime=None):
        self._ensure()
        r = self._cli.get_price_history_every_day(
            symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=False,
            need_previous_close=False,
        )
        j = r.json()
        return j["candles"] if isinstance(j, dict) and "candles" in j else j

    def get_quote(self, symbol):
        \"\"\"Fetch real-time quote for a single symbol.\"\"\"
        self._ensure()
        # API expects a list of symbols
        r = self._cli.get_quote([symbol])
        return r.json()

    def health_check(self, symbol="VOO"):
        try:
            self._ensure()
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=10)
            r = self.price_daily(symbol, start_datetime=start, end_datetime=end)
            n = len(r) if isinstance(r, list) else 0
            last_dt = None
            if n:
                ts = r[-1].get("datetime")
                if ts is not None:
                    last_dt = pd.to_datetime(ts, unit="ms", utc=True)
            return {
                "ok": True, "signature_used": self.signature_used, "symbol": symbol,
                "candles": n, "last_bar_utc": str(last_dt) if last_dt is not None else None,
            }
        except Exception as e:
            return {"ok": False, "error": str(e), "signature_used": self.signature_used}

sd = SchwabData()
"""
    with open(schwab_path, "w") as f:
        f.write(schwab_code)
    print("   ✅ Schwab Client Upgraded: Added get_quote().")


    # --- 2. UPGRADE SIMULATOR (Quote Fallback) ---
    trader_path = "simulation/paper_trader.py"
    trader_code = """
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
                if "pending_orders" not in state:
                    state["pending_orders"] = []
                return state
        return {
            "cash": self.start_cash,
            "equity": self.start_cash,
            "positions": {},
            "pending_orders": [],
            "history": [],
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
        
        for order in self.state.get("pending_orders", []):
            sym = order["symbol"]
            order_date = order["date"]
            
            fill_price = None
            fill_date = None
            
            # 1. Try Historical Data (The preferred, official record)
            df = fetch_single_symbol(sym, days=5, force_fresh=True)
            if df is not None and not df.empty:
                last_dt_str = str(df.index[-1].date())
                if last_dt_str > order_date:
                    fill_price = float(df.iloc[-1]["open"])
                    fill_date = last_dt_str

            # 2. Try Real-Time Quote (If history is stale but market is open)
            if fill_price is None:
                try:
                    q = sd.get_quote(sym)
                    # Schwab format: { 'SYMBOL': { 'quote': { 'openPrice': 123.4 ... } } }
                    if q and sym in q and 'quote' in q[sym]:
                        q_data = q[sym]['quote']
                        open_px = q_data.get('openPrice')
                        # Ensure it's valid and not 0.0
                        if open_px and open_px > 0:
                            fill_price = float(open_px)
                            fill_date = str(datetime.now().date())
                except: pass

            # 3. Execute Fill if Price Found
            if fill_price and fill_date and fill_date > order_date:
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
                            "date": fill_date,
                            "current_price": fill_price,
                            "unrealized_pnl": 0.0,
                            "unrealized_pct": 0.0
                        }
                        filled_log.append(f"✅ FILLED {sym} at ${fill_price:.2f} (Open)")
                    else:
                        filled_log.append(f"❌ FAILED {sym}: Insufficient Cash")
            else:
                remaining_orders.append(order)
        
        self.state["pending_orders"] = remaining_orders
        self.save_state()
        return filled_log

    def update_valuations(self):
        # 1. Try Fill
        fill_logs = self.process_pending_orders()
        
        # 2. Update Active
        total_value = self.state["cash"]
        for sym, pos in self.state["positions"].items():
            # Use quote for active positions too if possible, else history
            # Falling back to history for speed in bulk, but could use quotes
            df = fetch_single_symbol(sym, days=30, force_fresh=False)
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

                self.state["cash"] += proceeds
                self.state["history"].append({
                    "symbol": sym, "strategy": strat_name,
                    "entry_date": pos["date"], "exit_date": str(datetime.now().date()),
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "pnl": pnl, "return_pct": (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
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
            logs.append(f"⏳ QUEUED {cand['Symbol']} (Buy Next Open)")
            current_count += 1

        self.save_state()
        return logs
"""
    with open(trader_path, "w") as f:
        f.write(trader_code)
    print("   ✅ Simulator Fixed: Now uses Real-Time Quotes to fill MOO orders.")


if __name__ == "__main__":
    fix_simulator_fills()
