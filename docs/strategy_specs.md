Qullamaggie and Minervini Strategy Specs (Daily Bar Approximation)

Purpose
- Provide a concise, implementation-ready spec for the Qullamaggie Breakout, Qullamaggie Episodic Pivot (EP), and Minervini SEPA setups.
- Align to after-close scanning and next-day stop order execution.

Sources
- Public summaries of the Minervini SEPA method and buy-stop entry guidance.
- Public summaries of Qullamaggie breakout and EP characteristics.
- Reference: chartmill.com documentation on Minervini SEPA and buy-stop entries.

Common Assumptions
- Daily-bar approximation only (no intraday).
- Signals are generated after market close on day T.
- Orders are placed for day T+1 as buy-stop-limit orders.
- Fundamentals are not enforced for SEPA unless a reliable data source is added.

Qullamaggie Breakout (Long)
1) Pre-move
   - Stock is in the top momentum bucket over the last 3-6 months.
   - Default: momentum_rank >= 98 (top ~2%).
2) Base / Consolidation
   - Contraction in range: range_pct_5 < range_pct_10 < range_pct_20 < range_pct_40.
   - Volume dry-up: vol_ma10 < vol_ma50.
3) Liquidity / Volatility
   - ADR (Qullamaggie definition) >= 3.0 by default.
4) Trigger
   - Buy-stop at pivot (default high_20_prev * stop_buy_mult).
5) Stop
   - Stop at signal-day low (LOD).
   - Reject trade if stop width > max_stop_pct.
6) Partial Profit
   - Time-based partial after 3-5 days if trade is profitable.
7) Exit
   - Trail by 10- or 20-day SMA after partial.

Qullamaggie Episodic Pivot (EP)
1) Gap + Volume
   - Gap >= 10% (gap_pct).
   - Volume >= 2.0x vol_ma50 (configurable).
2) Trigger
   - Buy-stop at gap-day high (signal-day high).
3) Stop
   - Stop at gap-day low.
   - Reject trade if stop width > max_stop_pct.
4) Partial Profit / Exit
   - Same as breakout (time-based partial, then SMA trail).

Minervini SEPA (Long)
1) Trend Template
   - Price > SMA150 and SMA200.
   - SMA150 > SMA200.
   - SMA200 rising over ~1 month.
   - SMA50 > SMA150 and SMA200.
   - Price > SMA50.
   - Price within 25% of 52-week high.
   - Price at least 30% above 52-week low.
   - RS rating >= 70 (configurable).
2) VCP Proxy
   - Contraction in range (range_pct_5 < range_pct_10 < range_pct_20 < range_pct_40).
   - Volume dry-up (vol_ma10 < vol_ma50).
3) Trigger
   - Buy-stop at pivot (default high_20_prev * stop_buy_mult).
4) Stop
   - Stop at signal-day low.
   - Reject trade if stop width > max_stop_pct (default 8%).
5) Exit
   - Trail by SMA10 or SMA20; optionally loosen to SMA50 after partial.

Execution Notes
- Backtests must simulate stop-order fills:
  - If next-day high does not reach trigger, no fill.
  - If next-day open gaps above limit, no fill.
  - Otherwise, fill at open (if open >= trigger) or at trigger.
