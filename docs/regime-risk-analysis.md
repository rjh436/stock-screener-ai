# Apex Swing: Regime Risk Analysis

## Production Truth

- **10Y strict PIT validation:** `2016-03-21` to `2026-03-20`
- **Validated operating envelope:** `35.52%` CAGR / `24.08%` max drawdown / `226` total trades (`22.6` trades per year)
- **Max Verified strict PIT window:** `2006-09-29` to `2026-03-20`
- **Long-window disclosure:** `16.67%` CAGR / `30.06%` max drawdown / `376` total trades (`19.3` trades per year)

## What Max Verified Means

`Max Verified` is a strict long-window regime-risk disclosure, not the primary production benchmark.

The shipped Apex Swing engine is production-worthy on the validated 10Y operating window. Over the full Russell 3000 PIT-supported history, it remains profitable but materially weaker, which indicates regime dependence rather than a broken execution model.

## Entry-Gate Hardening Experiments

All bounded hardening experiments were run on the same strict PIT Max Verified path and all failed to improve long-run risk-adjusted returns.

| Gate | Full CAGR | Max DD | Trades | Verdict |
| --- | ---: | ---: | ---: | --- |
| Baseline | 16.67% | 30.06% | 376 | Reference |
| Traffic-light hard cash | 14.66% | 28.89% | 325 | CAGR cost exceeded DD benefit |
| SPY SMA50/200 + VIX < 25 | 10.90% | 30.69% | 203 | Destroyed CAGR and worsened DD |
| Spot VIX < 25 only | 13.87% | 28.91% | 353 | CAGR cost still exceeded DD benefit |

## Final Bounded Test

The final council-approved branch was the spot-VIX-only entry gate. It still failed:

- **Full Max Verified window:** `13.87%` CAGR / `28.91%` max drawdown / `353` trades
- **Late-era only (`2017-01-03` to `2026-03-20`):** `32.46%` CAGR / `20.58%` max drawdown / `216` trades

That late-era result is below the council's immediate-abort threshold (`37%` CAGR), which means the VIX filter degraded the strong regime instead of preserving it.

## Conclusion

No static market-level entry filter improved 20Y risk-adjusted returns for Apex Swing on the strict PIT Russell 3000 path.

The practical conclusion is:

- Keep Apex Swing shipped as the current 10Y validated production strategy.
- Treat Max Verified as a regime-risk disclosure for users, not an optimization promise.
- Stop further single-strategy market-gating optimization on this architecture.
- If long-window robustness becomes a product requirement, pursue a complementary sleeve or portfolio router rather than more entry-gate tuning inside Apex Swing.
