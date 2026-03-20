
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
    # Keep a defined name to avoid NameError during auth.
    easy_client = None

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
        self._ensure_lock = threading.Lock()
        self._rate_sem = threading.BoundedSemaphore(self._get_rate_limit_concurrency())
        self._next_allowed_ts = 0.0
        self._base_min_interval_s = self._get_min_request_interval_s()
        self._dynamic_min_interval_s = self._base_min_interval_s
        self._auth_invalid = False
        self._auth_invalid_detail = ""

    def _ensure(self):
        if self._auth_invalid:
            detail = self._auth_invalid_detail or "token_invalid"
            raise RuntimeError(f"Schwab auth unavailable: {detail}")
        if self._cli is not None:
            return
        with self._ensure_lock:
            if self._auth_invalid:
                detail = self._auth_invalid_detail or "token_invalid"
                raise RuntimeError(f"Schwab auth unavailable: {detail}")
            if self._cli is not None:
                return

            # STRICT MODE: Only allow project-level token path
            # This prevents "Ghost Tokens" from being created in ~/.schwab
            print(f"🔐 Authenticating with token at: {self._tok}")
            try:
                if easy_client is None:
                    raise RuntimeError("schwab.auth.easy_client is unavailable (missing Schwab SDK).")
                self._cli = easy_client(
                    api_key=self.cid,
                    app_secret=self.ck,
                    callback_url=self.redir,
                    token_path=self._tok
                )
                self.signature_used = "strict_kwargs"
            except Exception as e:
                print(f"❌ AUTHENTICATION FAILED: {e}")
                print(f"👉 Please run 'python3 Reset_Auth_Final.py' to fix this.")
                raise e

    def _mark_auth_invalid(self, detail: Exception | str) -> None:
        msg = str(detail).strip() or "token_invalid"
        self._auth_invalid = True
        self._auth_invalid_detail = msg

    @staticmethod
    def _looks_auth_invalid(err: Exception | str) -> bool:
        msg = str(err).lower()
        return (
            "token_invalid" in msg
            or "invalid token" in msg
            or "access token" in msg
            or "401" in msg
            or "unauthorized" in msg
            or "invalid_client" in msg
        )

    @staticmethod
    def _get_rate_limit_concurrency() -> int:
        try:
            v = int(os.getenv("SCHWAB_API_CONCURRENCY", "10"))
        except Exception:
            v = 10
        return max(1, min(v, 24))

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
            v = float(os.getenv("SCHWAB_API_MIN_INTERVAL", "0.20"))
        except Exception:
            v = 0.20
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
                if status == 401:
                    self._mark_auth_invalid(f"HTTP 401 for {symbol}")
                    break
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
                    if self._looks_auth_invalid(j):
                        self._mark_auth_invalid(j)
                        break
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
                if self._looks_auth_invalid(e):
                    self._mark_auth_invalid(e)
                    break
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
                if status == 401:
                    self._mark_auth_invalid(f"HTTP 401 for {symbol}")
                    break
                if status == 429:
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                if isinstance(status, int) and status >= 500:
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                j = r.json()
                if self._looks_auth_invalid(j):
                    self._mark_auth_invalid(j)
                    break
                if isinstance(j, dict) and self._looks_rate_limited(j):
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue
                self._note_success()
                return j

            except Exception as e:
                last_exc = e
                if self._looks_auth_invalid(e):
                    self._mark_auth_invalid(e)
                    break
                if attempt >= max_tries:
                    break

                if self._looks_rate_limited(e):
                    self._note_rate_limit()
                    time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                    continue

                time.sleep(min(1.0, backoff_max))

        raise RuntimeError(f"Schwab quote failed for {symbol} after {max_tries} attempts: {last_exc}")

    def get_quotes(self, symbols: list) -> dict:
        """Fetch real-time quotes for a list of symbols in batches."""
        self._ensure()
        if not symbols:
            return {}

        results = {}
        chunk_size = 150
        max_tries = self._get_max_retries()
        backoff_max = self._get_backoff_max_s()

        print(f"📡 Fetching {len(symbols)} quotes in chunks of {chunk_size}...")
        for i in range(0, len(symbols), chunk_size):
            chunk = [s for s in symbols[i : i + chunk_size] if s]
            if not chunk:
                continue

            for attempt in range(1, max_tries + 1):
                try:
                    with self._rate_limited():
                        r = self._cli.quote(chunk)

                    status = getattr(r, "status_code", None)
                    if status == 401:
                        self._mark_auth_invalid(f"HTTP 401 during quote batch {i // chunk_size + 1}")
                        break
                    if status == 429:
                        self._note_rate_limit()
                        time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                        continue

                    if isinstance(status, int) and status >= 500:
                        time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                        continue

                    j = r.json()
                    if self._looks_auth_invalid(j):
                        self._mark_auth_invalid(j)
                        break
                    if isinstance(j, dict) and self._looks_rate_limited(j):
                        self._note_rate_limit()
                        time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                        continue

                    if isinstance(j, dict):
                        results.update(j)
                    elif isinstance(j, list):
                        for item in j:
                            if not isinstance(item, dict):
                                continue
                            key = item.get("symbol") or item.get("symbolId") or item.get("key")
                            if key:
                                results[str(key)] = item

                    self._note_success()
                    break

                except Exception as e:
                    if self._looks_auth_invalid(e):
                        self._mark_auth_invalid(e)
                        break
                    if attempt >= max_tries:
                        break

                    if self._looks_rate_limited(e):
                        self._note_rate_limit()
                        time.sleep(self._retry_sleep_s(attempt, max_s=backoff_max))
                        continue

                    time.sleep(min(1.0, backoff_max))

            if self._auth_invalid:
                break

        return results

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
