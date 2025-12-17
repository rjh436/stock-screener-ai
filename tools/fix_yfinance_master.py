import yfinance as yf
import shutil
import os
import sys
import time

def patch_yfinance_invalid_crumb():
    """
    Workaround for Yahoo 'Invalid Crumb' issues in some networks/DNS setups.
    Replaces yfinance's default cookie bootstrap URL (fc.yahoo.com) with a
    consent endpoint that still sets the required A3 cookie.
    """
    try:
        from yfinance.data import YfData
        from curl_cffi import requests as curl_requests
    except Exception as e:
        print(f"⚠️ Unable to patch yfinance cookie bootstrap: {e}")
        return False

    def _get_cookie_basic(self, timeout=30):
        if getattr(self, "_cookie", None) is not None:
            return True

        try:
            if self._load_cookie_curlCffi():
                return True
        except Exception:
            pass

        try:
            self._session.get(
                url="https://guce.yahoo.com/consent",
                timeout=timeout,
                allow_redirects=False,
            )
        except curl_requests.exceptions.DNSError:
            return False
        except Exception:
            return False

        try:
            cookies = self._session.cookies.jar._cookies
            yahoo_domains = [d for d in cookies.keys() if "yahoo" in d]
            if len(yahoo_domains) > 1:
                yahoo_domains = [d for d in yahoo_domains if "consent" not in d]
            if not yahoo_domains:
                return False
            domain = yahoo_domains[0]
            cookie = cookies[domain]["/"].get("A3")
            if cookie is None:
                return False
            self._cookie = cookie
        except Exception:
            return False

        try:
            self._save_cookie_curlCffi()
        except Exception:
            pass

        return True

    YfData._get_cookie_basic = _get_cookie_basic
    return True

def nuke_cache():
    """Clear corrupted Yahoo cookies/crumbs"""
    print("\n🛠️  STEP 1: Nuke Cache...")
    cache_roots = [
        os.path.expanduser("~/Library/Caches/py-yfinance"),
        os.path.expanduser("~/.cache/py-yfinance")
    ]
    for root in cache_roots:
        if os.path.exists(root):
            try:
                shutil.rmtree(root)
                print(f"   - Deleted: {root}")
            except Exception as e:
                print(f"   - Failed to delete {root}: {e}")
        else:
            print(f"   - Clean: {root}")

def test_fetch():
    """Verify connection works"""
    print("\n🔍 STEP 2: Testing Connection (AAPL)...")
    try:
        patch_yfinance_invalid_crumb()
        # Simple fetch - let yfinance 0.2.66 handle the session logic
        stock = yf.Ticker("AAPL")
        info = stock.info
        
        # Check for key data points
        pe = info.get('forwardPE')
        debt = info.get('debtToEquity')
        
        if pe is not None or debt is not None:
            print(f"✅ SUCCESS: Data received!")
            print(f"   - Forward PE: {pe}")
            print(f"   - Debt/Eq:    {debt}")
            return True
        else:
            print("⚠️  WARNING: Connection made, but data empty.")
            return False
            
    except Exception as e:
        print(f"❌ FAILURE: {e}")
        return False

if __name__ == "__main__":
    nuke_cache()
    # Sleep briefly to ensure file system updates
    time.sleep(1) 
    test_fetch()
