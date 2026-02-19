# Contributing Guide

## Branching

- Create feature/fix branches from the active base branch.
- Keep branch names descriptive.

## Before Opening a PR

1. Run syntax checks:
   - `python3 -m py_compile app.py simulation/paper_trader.py`
2. Run product QA suite:
   - `./.venv/bin/python tools/run_product_audit.py --base-url http://localhost:8501`
3. Run hygiene audit:
   - `./.venv/bin/python tools/repo_hygiene_audit.py`
4. Confirm no secrets/artifacts are staged:
   - `git status --short`

## Code Quality Expectations

- Prefer explicit error handling over broad `except: pass`.
- Add concise comments only where needed for non-obvious logic.
- Preserve point-in-time data integrity in backtest paths.
- Do not weaken accuracy gates for baseline-quality workflows.

## UI/UX Changes

- Keep desktop and mobile layouts usable.
- Add/update screenshot artifacts in `output/playwright/` when validating.
- Re-run forensic UI test scripts after visual changes.

## Commit Hygiene

- Keep commits focused and reviewable.
- Avoid committing large generated artifacts, caches, or logs.
- Never commit credentials or local token files.

