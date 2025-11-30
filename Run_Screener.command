#!/bin/zsh
# RJH Custom Screener Launcher (Optimized for M3 Max + Python 3.12)

set -e
export PYTHONUTF8=1

# Move to this script’s folder
cd "$(dirname "$0")"

# Path to Python 3.12 (auto-detected)
PY312=$(which python3)

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python 3.12 virtual environment..."
    $PY312 -m venv .venv
fi

# Activate environment
source .venv/bin/activate

# Only install requirements if missing (NOT every run)
if [ ! -f ".venv/.deps_installed" ]; then
    echo "⬆️ Upgrading pip..."
    pip install --upgrade pip

    echo "📥 Installing requirements..."
    pip install -r requirements.txt

    # Marker to avoid reinstalling every launch
    touch .venv/.deps_installed
fi

# Launch Streamlit
echo "🚀 Launching Streamlit App..."
streamlit run app.py
