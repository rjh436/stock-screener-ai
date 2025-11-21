import optimization.optimizer
import backtest_runner
import os

print("1. Checking optimizer.py...")
try:
    with open(optimization.optimizer.__file__, 'r') as f:
        content = f.read()
        if "sharpe_score = min(sharpe, 3.0) * 40.0" in content:
            print("✅ Optimizer: Sharpe Logic Found")
        else:
            print("❌ Optimizer: STILL BROKEN")
            print("Debug: Content snippet: ", content[:200])
except Exception as e:
    print(f"❌ Optimizer Check Failed: {e}")

print("\n2. Checking backtest_runner.py...")
try:
    with open(backtest_runner.__file__, 'r') as f:
        content = f.read()
        if "_precalculate_signals" in content:
            print("✅ Backtester: Vectorization Found")
        else:
            print("❌ Backtester: STILL BROKEN")
            print("Debug: Content snippet: ", content[:200])
except Exception as e:
    print(f"❌ Backtester Check Failed: {e}")
