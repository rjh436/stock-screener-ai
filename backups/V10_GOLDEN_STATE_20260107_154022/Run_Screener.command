#!/bin/bash
# RJH Custom Screener Launcher with Pre-Flight Auth + Port Cleanup

set -e
export PYTHONUTF8=1

# Move to this script’s folder
cd "$(dirname "$0")"

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

# Clear any processes holding port 8182 to avoid startup conflicts
PORT_PIDS=$(lsof -ti:8182 2>/dev/null || true)
if [ -n "$PORT_PIDS" ]; then
    echo "🧹 Clearing port 8182..."
    echo "$PORT_PIDS" | xargs kill -9 || true
fi

echo "🛫 Running pre-flight auth check..."
if python tools/auto_login.py; then
    echo "🚀 Launching Streamlit App..."
    streamlit run app.py
else
    echo "⚠️ Launch Aborted."
    read -n 1 -p "Press any key to close..."
fi
