#!/bin/bash
cd "$(dirname "$0")"

# Define path to virtual environment python
VENV_PYTHON=".venv/bin/python"

echo "🧪 Starting Apex Strategy Lab (Safe Sandbox Mode)..."
echo "---------------------------------------------------"
echo "Target: optimization/lab_optimizer.py"
echo "Goal: Evolve 'Super Signal' Exits without overwriting config."
echo "---------------------------------------------------"

if [ -f "$VENV_PYTHON" ]; then
    echo "✅ Using virtual environment..."
    "$VENV_PYTHON" optimization/lab_optimizer.py
else
    echo "⚠️ Virtual environment not found at .venv/bin/python"
    echo "Attempting to use system python3..."
    python3 optimization/lab_optimizer.py
fi

echo ""
echo "Analysis complete. Check 'config/lab_candidates.json' for results."
echo "Press any key to close..."
read -n 1 -s
