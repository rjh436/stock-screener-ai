# Apex Sniper AI (stock-screener-ai)

Apex Sniper AI is a Streamlit-based quant research and execution lab focused on growth/momentum strategies (Superperformance + Minervini/SEPA variants) with three operating modes:

- `Live Screener`: end-of-day opportunity scan
- `Backtest`: historical simulation with accuracy gating
- `Simulator`: paper-trading workflow (scan -> queue -> fill/exit)

## Current Status

- Primary app entrypoint: `app.py`
- Data source stack: Schwab-first with fallback/cached workflows
- Accuracy model: point-in-time (PIT) Russell 3000 support + coverage scorecards
- UX and regression testing: Playwright-driven desktop/mobile audit scripts

## Key Features

- Strategy-level scoring, entry/exit modeling, and risk controls
- Regime-aware backtesting and simulator position governance
- PIT universe handling for Russell 3000 windows
- Accuracy gate and diagnostics (coverage, shallow-history, stale data checks)
- CSV export, baseline locking, and repeatability drift checks
- Automated browser smoke tests and forensic UX/functionality audits

## Quick Start

### 1) Create/activate environment

```bash
cd "/Users/rob/Documents/Gemini Learning Stock Screener"
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure local env

Copy and edit local runtime variables:

```bash
cp .env.example .env
```

Do not commit `.env` or token files.

### 3) Launch app

```bash
./Run_Screener.command
```

Or:

```bash
./.venv/bin/python -m streamlit run app.py
```

## Accuracy and Data Integrity Notes

- For Russell 3000 baselines, prefer PIT-enabled runs.
- Use strict/verified mode for baseline-quality backtests.
- Treat low-coverage runs as non-baseline research only.
- Review generated missing-symbol reports before locking new baselines.

## Canonical Research Workflow

Prefer the realistic, cache-backed PIT evaluation scripts for candidate selection:

1. Rank seed candidates under realistic friction:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/evaluate_low_turnover_alpha_frontier.py --out tmp/alpha_frontier_latest.json
```

2. Expand around the current frontier with realistic costs:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_autonomous_cagr_search.py --out tmp/autonomous_cagr_search_latest.json
PYTHONPATH=. ./.venv/bin/python scripts/optimize_alpha_neighborhood_realistic.py --out tmp/alpha_neighborhood_latest.json
```

3. Run walk-forward robustness on finalists before promoting configs:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_superperformance_walkforward.py --config config/superperformance_multi_sleeve_allocator_v1.json
```

Notes:

- These scripts use realistic friction settings, PIT Russell 3000 membership where applicable, and robustness-aware ranking.
- Treat `optimization/optimizer.py` and the older `Run_Optimizer.command` path as exploratory single-window search only. They should not be the final authority for promoting a strategy.

## Automated QA / Product Audit

Run all primary checks:

```bash
./.venv/bin/python tools/run_product_audit.py --base-url http://localhost:8501
```

Individual suites:

```bash
./.venv/bin/python tools/test_app_ux_functional_e2e.py --base-url http://localhost:8501
./.venv/bin/python tools/test_live_simulator_e2e.py --base-url http://localhost:8501 --universe RUSSELL3000
./.venv/bin/python tools/test_mobile_mode_snapshots.py --base-url http://localhost:8501
```

Outputs are written under `output/playwright/`.

## Repo Hygiene Audit

Run:

```bash
./.venv/bin/python tools/repo_hygiene_audit.py
```

This checks for:

- tracked secrets/runtime artifacts
- root documentation gaps
- broad `except` usage hotspots

## Project Layout

- `app.py`: Streamlit UI and orchestration
- `data/`: loaders, universe resolution, broker integration
- `execution/`: backtest engine/parity/shared logic
- `simulation/`: paper trader and portfolio state transitions
- `strategies/`: strategy implementations + loader
- `optimization/`: optimizer/evolution workflows
- `tools/`: diagnostics, QA, utilities

## Security

See `SECURITY.md`.

At minimum:

- never commit real credentials
- rotate broker/API credentials if exposed
- keep local token files out of git

## Contributing

See `CONTRIBUTING.md` for branch, testing, and PR expectations.
