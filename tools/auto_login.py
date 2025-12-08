import os
import sys

# Add project root to system path so we can import 'data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.schwab_client import sd


def main() -> None:
    print("🔍 Pre-Flight Check: Verifying Schwab Token...")
    try:
        sd.health_check(symbol="SPY")
    except Exception:
        print("❌ Login Failed.")
        sys.exit(1)

    print("✅ Token Valid. Starting Engine...")
    sys.exit(0)


if __name__ == "__main__":
    main()
