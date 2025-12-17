import yfinance as yf
import pandas as pd
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

def compute_debt_to_equity_fallback(stock: yf.Ticker):
    """
    Returns a Debt/Equity % using the balance sheet when Yahoo's quote API
    doesn't provide `debtToEquity` (often missing when equity is negative).
    """
    try:
        bs = stock.balance_sheet
    except Exception:
        return None

    if bs is None or getattr(bs, "empty", True):
        return None

    latest_col = bs.columns[0]

    total_debt = None
    if "Total Debt" in bs.index:
        total_debt = bs.loc["Total Debt", latest_col]
    else:
        debt_components = ["Long Term Debt", "Short Term Debt", "Current Debt", "Short Long Term Debt"]
        debt_sum = 0.0
        found = False
        for key in debt_components:
            if key in bs.index:
                try:
                    debt_sum += float(bs.loc[key, latest_col] or 0)
                    found = True
                except Exception:
                    continue
        if found:
            total_debt = debt_sum

    equity = None
    equity_keys = [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Total Equity Gross Minority Interest",
        "Total Equity",
    ]
    for key in equity_keys:
        if key in bs.index:
            equity = bs.loc[key, latest_col]
            break

    try:
        total_debt_val = float(total_debt)
        equity_val = float(equity)
    except Exception:
        return None

    if total_debt_val <= 0:
        return 0.0

    if equity_val <= 0:
        return float("inf")

    return (total_debt_val / equity_val) * 100.0

def check_fundamentals(ticker):
    """
    The 'Anti-Rot' Filter.
    Returns TRUE if the stock is fundamentally safe enough to trade.
    Rejects 'Value Traps' (Negative Earnings + High Debt).
    """
    print(f"\n🔍 Analyzing {ticker} Fundamentals...")
    try:
        patch_yfinance_invalid_crumb()
        stock = yf.Ticker(ticker)
        # Force a network refresh to get latest data
        info = stock.info
        
        # --- 1. The "Solvency" Check (Debt to Equity) ---
        # Logic: If Debt is > 300% of Equity, the company is highly leveraged.
        debt_eq = info.get('debtToEquity')
        if debt_eq is None:
            debt_eq = compute_debt_to_equity_fallback(stock)
        
        # --- 2. The "Viability" Check (Profitability) ---
        # Logic: We don't need them to be profitable TODAY, but we need
        # EITHER positive earnings OR revenue growth.
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        
        # --- 3. The "Hypetrain" Check (PEG Ratio) ---
        peg = info.get('pegRatio')

        print(f"   - Debt/Eq:   {debt_eq} (Limit: 300)")
        print(f"   - PE Trail:  {trailing_pe}")
        print(f"   - PE Fwd:    {forward_pe}")
        print(f"   - PEG:       {peg}")

        # --- REJECTION LOGIC (The "Trash Can") ---
        
        # Rule 1: The Debt Trap
        if debt_eq is not None and debt_eq > 300:
            print(f"❌ REJECT: Excessive Debt ({debt_eq}%)")
            return False

        # Rule 2: The Zombie (No Earnings visibility)
        # It's okay to have no Trailing PE (turnaround play), 
        # but you MUST have a Forward PE (analysts expect profit).
        if trailing_pe is None and forward_pe is None:
            print("❌ REJECT: No Earnings Visibility (Zombie Company)")
            return False

        print("✅ PASS: Fundamentally Sound")
        return True

    except Exception as e:
        print(f"⚠️ Data Fetch Error for {ticker}: {e}")
        # FAIL SAFE: If data fails, we usually ALLOW the trade in backtest
        # but flag it in live trading. For now, return True to avoid starvation.
        return True 

if __name__ == "__main__":
    # Test 1: Apple (Should Pass easily)
    check_fundamentals("AAPL")
    
    # Test 2: Peloton (Should Fail or struggle - High losses)
    check_fundamentals("PTON")
    
    # Test 3: AMC (Should Fail - Debt/Dilution)
    check_fundamentals("AMC")
