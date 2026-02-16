Russell 3000 Point-In-Time (PIT) Membership Data
=================================================

Purpose
-------
This folder is used to remove survivorship bias in long-horizon Russell 3000
backtests by providing point-in-time index membership.

Supported formats
-----------------
1) Snapshot files (preferred when you have periodic constituent lists):
   - Place CSV files in this folder.
   - Filename must contain a date, for example:
     - `russell3000_2006-06-30.csv`
     - `iwv_holdings_2014_12_31.csv`
   - Required CSV column:
     - `symbol` or `ticker`
   - Loader behavior:
     - For an `as_of_date`, the engine selects the latest snapshot dated
       on or before that date.

2) Membership ranges file (preferred when you have add/remove dates):
   - Set env var `RUSSELL3000_PIT_MEMBERSHIP_CSV` to a CSV path.
   - Required columns:
     - `symbol` (or `ticker`)
     - `start_date` (or `from_date`) in `YYYY-MM-DD`
   - Optional columns:
     - `end_date` (or `to_date`) in `YYYY-MM-DD` (blank means still active)

Quick start
-----------
1) Copy the template:
   - `data/russell3000_membership/template_membership_ranges.csv`
2) Fill it with your PIT records.
3) Export:
   - `export RUSSELL3000_PIT_MEMBERSHIP_CSV=/absolute/path/to/your_ranges.csv`
4) Validate:
   - `./.venv/bin/python tools/validate_pit_universe.py --strict`

Accuracy guardrails
-------------------
- Streamlit and CLI paths can enforce PIT-only Russell 3000 backtests when:
  `APEX_REQUIRE_PIT_UNIVERSE=1` (recommended default).
- If PIT data is missing, long-horizon backtests should be considered biased.

Symbol coverage helpers
-----------------------
- Optional alias file:
  - `data/russell3000_membership/symbol_aliases.json`
  - Use this for ticker formatting variants (for example class-share punctuation).
- Historical fallback providers (non-blocking):
  - `DATA_ENABLE_FALLBACK_HISTORY=1` (default on)
  - `DATA_FALLBACK_PROVIDERS=yahoo,stooq` (default order)
  - `DATA_FALLBACK_MAX_WORKERS=8`
  - `DATA_FALLBACK_TIMEOUT_SEC=8`
  - `DATA_FALLBACK_MIN_BARS=40`
