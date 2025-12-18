import os
import sys
import json
import time

# Add project root to path so `python tools/fetch_sectors.py` works reliably.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.chdir(PROJECT_ROOT)


def _find_venv_python() -> str | None:
    candidates = [
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _ensure_venv() -> None:
    venv_python = _find_venv_python()
    if not venv_python:
        return
    if os.path.realpath(sys.executable) == os.path.realpath(venv_python):
        return
    if os.environ.get("APEX_VENV_REEXEC") == "1":
        return
    os.environ["APEX_VENV_REEXEC"] = "1"
    os.execv(venv_python, [venv_python] + sys.argv)


_ensure_venv()


try:
    import yfinance as yf
except Exception:
    yf = None

try:
    from data.indices import get_index_symbols
except Exception as e:
    print(f"❌ Failed to import project modules: {e}")
    print("   Try running: .venv/bin/python tools/fetch_sectors.py")
    sys.exit(1)

SECTOR_FILE = "config/sectors.json"
BATCH_SIZE = 100
SLEEP_SECONDS = 0.25


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    if yf is None:
        print("❌ Missing dependency: yfinance")
        print("   Try running: .venv/bin/python tools/fetch_sectors.py")
        return

    # Work around Yahoo 'Invalid Crumb' failures seen in some environments.
    try:
        from strategies.generic import _patch_yfinance_invalid_crumb

        if _patch_yfinance_invalid_crumb():
            print("✅ Patched yfinance cookie bootstrap (Invalid Crumb workaround)")
    except Exception:
        pass

    symbols = get_index_symbols("S&P 1500") or []
    symbols = [str(s).strip().upper() for s in symbols if s]
    symbols = sorted(set(symbols))

    if not symbols:
        print("❌ No symbols returned for S&P 1500.")
        return

    sector_map = {}
    total = len(symbols)
    processed = 0

    print(f"📥 Fetching sector data for {total} symbols via yfinance (batch={BATCH_SIZE})...")

    for batch in _chunked(symbols, BATCH_SIZE):
        try:
            tickers = yf.Tickers(" ".join(batch))
        except Exception as e:
            print(f"⚠️ Batch init failed for {len(batch)} tickers: {e}")
            for sym in batch:
                sector_map[sym] = "Unknown"
                processed += 1
            continue

        for sym in batch:
            try:
                t = tickers.tickers.get(sym)
                info = (t.info or {}) if t is not None else {}
                sector = info.get("sector", "Unknown") or "Unknown"
            except Exception:
                sector = "Unknown"

            sector_map[sym] = sector
            processed += 1

            if processed % 100 == 0 or processed == total:
                print(f"Processed {processed}/{total}...")

        time.sleep(SLEEP_SECONDS)

    out_path = os.path.join(PROJECT_ROOT, SECTOR_FILE)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sector_map, f, indent=4, sort_keys=True)

    unknowns = sum(1 for v in sector_map.values() if (v or "Unknown") == "Unknown")
    print(f"✅ Saved sector map: {out_path}")
    print(f"   - Unknown sectors: {unknowns}/{total}")


if __name__ == "__main__":
    main()
