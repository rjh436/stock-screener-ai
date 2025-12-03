import sys
import os
import json
import time

# Ensure we can find the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from data.schwab_client import sd
    print("📡 Testing Schwab Quote Connection...")

    symbol = "NWL"  # One of your pending stocks
    print(f"   Fetching quote for: {symbol}")

    try:
        # Force a fresh auth check
        sd._ensure()
        print(f"   Auth Status: {sd.signature_used}")

        # Raw call
        quote = sd.get_quote(symbol)
        print("\n📄 RAW QUOTE DATA:")
        print(json.dumps(quote, indent=2))

        # Logic Check
        if quote and symbol in quote:
            q = quote[symbol].get('quote', {})
            open_px = q.get('openPrice')
            print(f"\n✅ Parsed Open Price: {open_px}")

            if open_px and open_px > 0:
                print("   CONCLUSION: Logic *should* have filled this. Check dates.")
            else:
                print("   ❌ CONCLUSION: 'openPrice' is missing or zero.")
        else:
            print("   ❌ CONCLUSION: Symbol not found in response keys.")

    except Exception as e:
        print(f"\n❌ API ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

except ImportError as e:
    print("❌ Could not import data.schwab_client. Run from project root.")
    print(f"Import error: {e}")
    import traceback
    traceback.print_exc()
