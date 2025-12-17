import yfinance as yf
import shutil
import os
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
    """Force delete the yfinance cache to fix 'Invalid Crumb' errors"""
    cache_dirs = [
        os.path.expanduser("~/Library/Caches/py-yfinance"),
        os.path.expanduser("~/.cache/py-yfinance")
    ]
    print("\n🧹 Cleaning Cache...")
    for d in cache_dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"   - Deleted: {d}")
            except Exception as e:
                print(f"   - Failed to delete {d}: {e}")
        else:
            print(f"   - Already clean: {d}")

def check_fundamentals(ticker):
    print(f"\n🔍 Analyzing {ticker} (Standard Mode)...")
    try:
        # Standard Call (Let yfinance handle the session)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        debt_eq = info.get('debtToEquity')
        fwd_pe = info.get('forwardPE')
        
        print(f"   - Debt/Eq:   {debt_eq}")
        print(f"   - PE Fwd:    {fwd_pe}")
        
        if debt_eq is not None and debt_eq > 300:
            print("❌ REJECT: Excessive Debt")
        elif fwd_pe is None:
            print("❌ REJECT: Zombie (No Earnings)")
        else:
            print("✅ PASS: Solid Fundamentals")
            
    except Exception as e:
        print(f"⚠️ Failed: {e}")

if __name__ == "__main__":
    nuke_cache()
    patch_yfinance_invalid_crumb()
    check_fundamentals("AAPL")
    time.sleep(1)
    check_fundamentals("PTON")
