#!/bin/bash
# RJH Custom Screener Launcher with Pre-Flight Auth + Port Cleanup

set -e
export PYTHONUTF8=1
set -o pipefail

# Move to project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# Early pre-log (survives even if process is killed before full logger init).
mkdir -p logs
PRELOG="logs/launcher_preflight_$(date +%Y%m%d_%H%M%S).log"
{
    echo "=== Apex Launcher Preflight ==="
    echo "Timestamp: $(date)"
    echo "User: $(whoami)"
    echo "PWD: $(pwd)"
} >> "$PRELOG"

# Startup log (helps diagnose external SIGKILL / OOM events).
LAUNCH_LOG="logs/streamlit_launch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
echo "📝 Launch log: $LAUNCH_LOG"
echo "📝 Preflight log: $PRELOG"

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

# Verify selected Python can actually execute code.
if ! "$PY_EXEC" -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo "⚠️ Primary Python failed runtime check: $PY_EXEC"
    if command -v python3 >/dev/null 2>&1; then
        PY_EXEC="python3"
        echo "↪ Falling back to: $PY_EXEC"
    fi
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PY_EXEC -m venv .venv
fi

# Activate environment
source .venv/bin/activate

# Clear quarantine xattrs in venv to avoid macOS library-load policy issues.
xattr -dr com.apple.quarantine .venv 2>/dev/null || true

# Install requirements if marker missing
if [ ! -f ".venv/.deps_installed" ]; then
    echo "⬆️ Upgrading pip..."
    pip install --upgrade pip

    echo "📥 Installing requirements..."
    pip install -r requirements.txt

    touch .venv/.deps_installed
fi

# Validate core runtime modules. Rebuild venv once if integrity is broken.
if ! python -c "import streamlit, pandas, numpy" >/dev/null 2>&1; then
    echo "⚠️ Python environment integrity check failed. Rebuilding .venv..."
    deactivate 2>/dev/null || true
    rm -rf .venv
    $PY_EXEC -m venv .venv
    source .venv/bin/activate
    xattr -dr com.apple.quarantine .venv 2>/dev/null || true
    pip install --upgrade pip
    pip install -r requirements.txt
    touch .venv/.deps_installed
fi

# Conservative runtime defaults to reduce sudden memory pressure spikes.
# You can override any of these in your shell/.env.
export DATA_FETCH_WORKERS="${DATA_FETCH_WORKERS:-8}"
export SCHWAB_API_CONCURRENCY="${SCHWAB_API_CONCURRENCY:-2}"
export APEX_BACKTEST_CACHE_MAX_ENTRIES="${APEX_BACKTEST_CACHE_MAX_ENTRIES:-1}"
export APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY="${APEX_BACKTEST_FORCE_REFRESH_FOR_ACCURACY:-1}"
export APEX_BACKTEST_CACHE_ONLY_WHEN_CLOSED="${APEX_BACKTEST_CACHE_ONLY_WHEN_CLOSED:-1}"
export APEX_BACKTEST_REFRESH_WHEN_CLOSED="${APEX_BACKTEST_REFRESH_WHEN_CLOSED:-0}"
export APEX_INDICATOR_CACHE_MAX_BYTES="${APEX_INDICATOR_CACHE_MAX_BYTES:-3221225472}" # 3 GiB
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export APEX_SAFE_MODE="${APEX_SAFE_MODE:-1}"

if [ "$APEX_SAFE_MODE" = "1" ]; then
    # In safe mode, avoid loading large pickled indicator cache blobs into RAM.
    export APEX_DISABLE_INDICATOR_CACHE="${APEX_DISABLE_INDICATOR_CACHE:-1}"
    echo "🛡️ Safe mode enabled (memory-conservative runtime)."
fi

echo "Runtime: DATA_FETCH_WORKERS=$DATA_FETCH_WORKERS SCHWAB_API_CONCURRENCY=$SCHWAB_API_CONCURRENCY"
echo "Runtime: APEX_INDICATOR_CACHE_MAX_BYTES=$APEX_INDICATOR_CACHE_MAX_BYTES APEX_SAFE_MODE=$APEX_SAFE_MODE"

# Clear any stale Streamlit processes holding key ports.
for PORT in 8501 8182; do
    PORT_PIDS=$(lsof -ti:"$PORT" 2>/dev/null || true)
    if [ -n "$PORT_PIDS" ]; then
        echo "🧹 Clearing port $PORT..."
        echo "$PORT_PIDS" | xargs kill -15 || true
        sleep 1
        PORT_PIDS=$(lsof -ti:"$PORT" 2>/dev/null || true)
        if [ -n "$PORT_PIDS" ]; then
            echo "$PORT_PIDS" | xargs kill -9 || true
        fi
    fi
done

echo "🛫 Running pre-flight auth check..."
if python tools/auto_login.py; then
    # If Streamlit is already up, avoid duplicate launch.
    if lsof -ti:8501 >/dev/null 2>&1; then
        echo "✅ Streamlit already running on http://localhost:8501"
        exit 0
    fi

    RUNTIME_LOG="logs/streamlit_runtime_$(date +%Y%m%d_%H%M%S).log"
    PID_FILE="logs/streamlit.pid"
    echo "🚀 Launching Streamlit App (detached)..."
    echo "📝 Runtime log: $RUNTIME_LOG"
    nohup python -m streamlit run app.py --server.fileWatcherType=none >> "$RUNTIME_LOG" 2>&1 &
    STREAMLIT_PID=$!
    echo "$STREAMLIT_PID" > "$PID_FILE"
    sleep 2
    if kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "✅ Streamlit started (PID: $STREAMLIT_PID)"
        echo "🌐 Open: http://localhost:8501"
        exit 0
    fi

    echo "❌ Streamlit failed to stay running. Last log lines:"
    tail -n 80 "$RUNTIME_LOG" || true
    exit 1
else
    echo "⚠️ Launch Aborted."
    read -n 1 -p "Press any key to close..."
fi
