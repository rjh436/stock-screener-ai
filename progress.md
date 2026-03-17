# Progress Log

## 2026-03-16
- Fixed `tests/test_superperformance_vcp_extension.py` so it adds the repo root to `sys.path`, matching the rest of the direct-run test files.
- Attempted a `tests/sitecustomize.py` bootstrap, confirmed it was not loaded for direct file execution here, and replaced it with explicit repo-root bootstrapping in the older direct-run test modules.
- Verified direct test execution with `./.venv/bin/python` for:
  - `tests/test_optimizer_fitness.py`
  - `tests/test_robustness.py`
  - `tests/test_walkforward.py`
  - `tests/test_autonomous_promotion_gates.py`
  - `tests/test_repo_hygiene_audit.py`
  - `tests/test_superperformance_vcp_extension.py`
- Checked the running Streamlit app on `http://127.0.0.1:8501`; the first load was building the hybrid snapshot, and a reload confirmed the cached hybrid snapshot rendered successfully for `2026-03-16`.
- Fixed a Superperformance parameter-clamping bug that was silently forcing several configurable thresholds back up to legacy defaults in live strategy evaluation.
- Added support for `adaptive_green_min_avg_dollar_volume_50` and regression coverage for both adaptive green ADV50 relaxation and sub-default EP gap thresholds.
- Re-ran targeted Alpha B4 diagnostics on the PIT sample window `2016-03-02` to `2021-12-31` with the prepared-cache sample (`182` symbols from the seeded Russell 3000 subset):
  - Current B4 sample: `-0.94%` CAGR, `6.18%` max DD, `5` trades, `0.86` trades/year, stitched OOS CAGR `-1.51%`.
  - Gate forensics on `217,037` checked rows showed `trend_gate`, `adr_gate`, `runup_gate`, and `liquidity` as the largest rejection buckets, but a controlled `classic` trend / lower-ADR probe did not improve realized sample performance.
- Reverted the unvalidated Alpha B4 config relaxation after the controlled sample remained negative; kept the code-level bug fix and tests only.
- Traced the realized-trade bottleneck for Alpha B4 on the same PIT sample and found that `40` accepted VCP setups were collapsing to `5` next-day fills, with `35` rejected only because the next open gapped above the `3%` stop-limit band.
- Benchmarked the VCP entry-path variants and promoted `vcp_entry_mode="same_day_close"` with `vcp_close_extension_max_pct=0.08` for Alpha B4 after it outperformed the current B4 config and the archived frontier/practical configs on the same 10-year sampled PIT window.
- Rechecked the promoted Alpha B4 config on the sampled `2016-03-02` to `2026-03-02` PIT window:
  - Promoted B4 sample: `0.97%` CAGR, `11.84%` max DD, `12` trades, `1.2` trades/year, stitched OOS CAGR `1.13%`.
  - Previous B4 sample on the same window: `0.83%` CAGR, `10.97%` max DD, `8` trades, `0.8` trades/year, stitched OOS CAGR `0.91%`.
