#!/bin/bash
cd "$(dirname "$0")"

# Define path to virtual environment python
VENV_PYTHON=".venv/bin/python"

echo "Starting Apex Genetic Algorithm..."

if [ -f "$VENV_PYTHON" ]; then
    echo "Using virtual environment..."
    "$VENV_PYTHON" optimization/optimize_ga.py
else
    echo "Virtual environment not found at .venv/bin/python"
    echo "Attempting to use system python3..."
    python3 optimization/optimize_ga.py
fi

echo ""
echo "GA complete. Press any key to close..."
read -n 1 -s
