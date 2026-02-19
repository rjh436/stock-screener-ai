# Security Policy

## Supported Scope

This repository includes local tooling for broker-connected workflows. It must be treated as a sensitive codebase even when used for paper trading.

## Sensitive Data (Never Commit)

- `.env` and `.env.*` (except `.env.example`)
- broker tokens/credentials (for example `data/schwab_tokens.json`)
- local auth/session artifacts
- local database snapshots with account history

## Credential Handling

1. Store secrets only in local environment files that are gitignored.
2. Use least-privilege broker/API credentials where possible.
3. Rotate credentials immediately after accidental exposure.

## If You Suspect a Secret Leak

1. Revoke/rotate exposed credentials immediately.
2. Remove the secret from the current working tree.
3. Confirm it is not tracked:
   - `git ls-files | rg -n "\\.env|schwab_tokens|credentials|secret|token"`
4. Run hygiene audit:
   - `./.venv/bin/python tools/repo_hygiene_audit.py`
5. If already pushed publicly, rotate first, then clean history.

## Operational Hardening

- Keep Streamlit app bound to trusted local network contexts.
- Avoid exposing local app instances to public internet without authentication.
- Review logs and exports before sharing externally.

