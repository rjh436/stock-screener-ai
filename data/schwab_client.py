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
    # Dummy st for CLI usage
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
    pass # Handle in app.py or main entry point

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

        # Try all known schwab-py signatures
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
                lambda: easy_client(
                    app_key=self.cid, app_secret=self.ck, redirect_uri=self.redir,
                    credentials_path=self.creds), "v3:classic-alt-kw")
        if self._cli is None:
            self._cli = _try(
                lambda: easy_client(self.cid, self.ck, self.redir, self.creds),
                "v4:positional")

        if self._cli is None:
            # Raise error but let the caller handle it (e.g. Streamlit)
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

# Initialize one global client
sd = SchwabData()
