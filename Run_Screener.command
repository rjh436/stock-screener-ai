#!/bin/bash
# Thin wrapper: invoke the real launcher via /bin/bash to avoid direct-exec instability.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec /bin/bash "$SCRIPT_DIR/scripts/run_screener_launcher.sh"
