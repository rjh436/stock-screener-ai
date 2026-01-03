
import os
import random
import threading
import time
from contextlib import contextmanager
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
        self._rate_lock = threading.Lock()
        self._rate_sem = threading.BoundedSemaphore(self._get_rate_limit_concurrency())
        self._next_allowed_ts = 0.0
        self._base_min_interval_s = self._get_min_request_interval_s()
        self._dynamic_min_interval_s = self._base_min_interval_s

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

    @staticmethod
    def _get_rate_limit_concurrency() -> int:
        try:
            v = int(os.getenv("SCHWAB_API_CONCURRENCY", "6"))
        except Exception:
            v = 6
        return max(1, min(v, 16))

    @staticmethod
    def _get_min_request_interval_s() -> float:
        rpm = os.getenv("SCHWAB_API_RPM", "").strip()
        if rpm:
            try:
                rpm_v = int(rpm)
                if rpm_v > 0:
                    return max(0.0, 60.0 / float(rpm_v))
            except Exception:
                pass

        try:
            v = float(os.getenv("SCHWAB_API_MIN_INTERVAL", "0.50"))
        except Exception:
            v = 0.50
        return max(0.0, v)

    @staticmethod
    def _get_max_retries() -> int:
        try:
            v = int(os.getenv("SCHWAB_API_MAX_RETRIES", "6"))
        except Exception:
            v = 6
        return max(1, min(v, 20))

    @staticmethod
    def _get_backoff_max_s() -> float:
        try:
            v = float(os.getenv("SCHWAB_API_BACKOFF_MAX", "30.0"))
        except Exception:
            v = 30.0
        return max(0.0, v)

    @contextmanager
    def _rate_limited(self):
        self._rate_sem.acquire()
        try:
            # Use an adaptive interval that increases after 429s.
            min_interval = max(self._base_min_interval_s, self._dynamic_min_interval_s)
            if min_interval > 0.0:
                with self._rate_lock:
                    now = time.monotonic()
                    sleep_s = max(0.0, self._next_allowed_ts - now)
                    self._next_allowed_ts = max(self._next_allowed_ts, now) + min_interval
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
            yield
        finally:
            self._rate_sem.release()

    def _note_success(self) -> None:
        # Slowly relax back toward the base interval after sustained success.
        with self._rate_lock:
            self._dynamic_min_interval_s = max(
                self._base_min_interval_s,
                self._dynamic_min_interval_s * 0.97,
            )

    def _note_rate_limit(self) -> None:
        # Aggressively slow down after 429s to avoid repeated bursts.
        with self._rate_lock:
            self._dynamic_min_interval_s = min(
                self._get_backoff_max_s(),
                max(self._dynamic_min_interval_s * 1.5, self._dynamic_min_interval_s + 0.25),
            )

    @staticmethod
    def _looks_rate_limited(err: Exception | str) -> bool:
        msg = str(err).lower()
        return ("too many requests" in msg) or ("rate limit" in msg) or ("429" in msg)

    @staticmethod
    def _retry_sleep_s(attempt: int, *, base_s: float = 1.0, max_s: float = 30.0) -> float:
        attempt = max(1, int(attempt))
        delay = min(max_s, base_s * (2 ** (attempt - 1)))
        # light jitter to avoid thundering herd
        delay += random.uniform(0.0, min(0.25, delay * 0.15))
        return float(delay)

    def price_daily(self, symbol, start_datetime=None, end_datetime=None):
        self._ensure()
        last_exc: Exception | None = None
        max_tries = self._get_max_retries()
        backoff_max = self._get_backoff_max_s()

        for attempt in range(1, max_tries + 1):
            try:
                with self._rate_limited():
                    r = self._cli.get_price_history_every_day(
                        symbol,
                        start_datetime=start_datetime,
                        end_datetime=end_datetime,
                        need_extended_hours_data=False,
                        need_previous_close=False,
                    )

                status = getattr(r, "status_code", None)
                if status == 429:
                    self._note_rate_limit()
                    retry_after = None
                    headers = getattr(r, "headers", None) or {}
                    ra = headers.get("Retry-After") or headers.get("retry-after")
                    if ra is not None:
                        try:
                            retry_after = float(ra)
                        except Exception:
                            retry_after = None

                    sleep_s = retry_after if retry_after is not None else self._retry_sleep_s(attempt, max_s=backoff_max)
                    time.sleep(sleep_s)
                    continue

                if isinstance(status, int) and status >= 500:
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                j = r.json()
                if isinstance(j, dict) and "candles" in j and isinstance(j["candles"], list):
                    self._note_success()
                    return j["candles"]

                # Schwab sometimes returns an error payload as a dict; treat it as a retryable failure
                # if it looks like rate limiting.
                if isinstance(j, dict):
                    if self._looks_rate_limited(j):
                        time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                        continue
                    raise RuntimeError(f"Schwab price history error for {symbol}: {j}")

                if isinstance(j, list):
                    self._note_success()
                    return j

                # Unknown payload type: treat as empty.
                self._note_success()
                return []

            except Exception as e:
                last_exc = e
                if attempt >= max_tries:
                    break

                if self._looks_rate_limited(e):
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                time.sleep(min(1.0, backoff_max))

        raise RuntimeError(f"Schwab price history failed for {symbol} after {max_tries} attempts: {last_exc}")

    def get_quote(self, symbol):
        """Fetch real-time quote for a single symbol."""
        self._ensure()
        # API expects a list of symbols
        last_exc: Exception | None = None
        max_tries = self._get_max_retries()
        backoff_max = self._get_backoff_max_s()

        for attempt in range(1, max_tries + 1):
            try:
                with self._rate_limited():
                    r = self._cli.get_quote(symbol)

                status = getattr(r, "status_code", None)
                if status == 429:
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                if isinstance(status, int) and status >= 500:
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                j = r.json()
                if isinstance(j, dict) and self._looks_rate_limited(j):
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue
                self._note_success()
                return j

            except Exception as e:
                last_exc = e
                if attempt >= max_tries:
                    break

                if self._looks_rate_limited(e):
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                time.sleep(min(1.0, backoff_max))

        raise RuntimeError(f"Schwab quote failed for {symbol} after {max_tries} attempts: {last_exc}")

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
