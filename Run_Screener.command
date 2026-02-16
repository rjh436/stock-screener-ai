#!/bin/bash
# RJH Custom Screener Launcher with Pre-Flight Auth + Port Cleanup

set -e
export PYTHONUTF8=1
set -o pipefail

# Move to this script’s folder
cd "$(dirname "$0")"

# Startup log (helps diagnose external SIGKILL / OOM events).
mkdir -p logs
LAUNCH_LOG="logs/streamlit_launch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
echo "📝 Launch log: $LAUNCH_LOG"

# Detect Python (Try 3.12 first, then fallback to system default)
if command -v python3.12 &> /dev/null; then
    PY_EXEC="python3.12"
elif command -v python3 &> /dev/null; then
    PY_EXEC="python3"
else
    echo "❌ Error: Python 3 not found."
    exit 1
fi

echo "✅ Using Python: $PY_EXEC"

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PY_EXEC -m venv .venv
fi

# Activate environment
source .venv/bin/activate

# Install requirements if marker missing
if [ ! -f ".venv/.deps_installed" ]; then
    echo "⬆️ Upgrading pip..."
    pip install --upgrade pip

    echo "📥 Installing requirements..."
    pip install -r requirements.txt

    touch .venv/.deps_installed
fi

# Conservative runtime defaults to reduce sudden memory pressure spikes.
# You can override any of these in your shell/.env.
export DATA_FETCH_WORKERS="${DATA_FETCH_WORKERS:-12}"
export SCHWAB_API_CONCURRENCY="${SCHWAB_API_CONCURRENCY:-4}"
export APEX_BACKTEST_CACHE_MAX_ENTRIES="${APEX_BACKTEST_CACHE_MAX_ENTRIES:-1}"
export APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY="${APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY:-1}"
export APEX_BACKTEST_CACHE_ONLY_WHEN_CLOSED="${APEX_BACKTEST_CACHE_ONLY_WHEN_CLOSED:-1}"
export APEX_BACKTEST_REFRESH_WHEN_CLOSED="${APEX_BACKTEST_REFRESH_WHEN_CLOSED:-0}"
export APEX_INDICATOR_CACHE_MAX_BYTES="${APEX_INDICATOR_CACHE_MAX_BYTES:-6442450944}" # 6 GiB
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Clear any processes holding port 8182 to avoid startup conflicts
PORT_PIDS=$(lsof -ti:8182 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    echo "🧹 Clearing port 8182..."
    echo "$PORT_PIDS" | xargs kill -9 || true
fi

echo "🛫 Running pre-flight auth check..."
if python tools/auto_login.py; then
    echo "🚀 Launching Streamlit App..."
    python -m streamlit run app.py --server.fileWatcherType=none
else
    echo "⚠️ Launch Aborted."
    read -n 1 -p "Press any key to close..."
fi
