#!/bin/bash
cd "$(dirname "$0")"

# Define path to virtual environment python
VENV_PYTHON=".venv/bin/python"

echo "🧠 Starting Smart Optimizer (Trend+Expectancy)..."

if [ -f "$VENV_PYTHON" ]; then
    echo "Using virtual environment..."
    "$VENV_PYTHON" optimize_smart.py
else
    echo "Virtual environment not found at .venv/bin/python"
    echo "Attempting to use system python3..."
    python3 optimize_smart.py
fi

echo ""
echo "Optimization complete. Press any key to close..."
read -n 1 -s
