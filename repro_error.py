import pandas as pd
import numpy as np
from execution.engine import _compute_indicators
import traceback

def test_indicators(df, name):
    print(f"Testing {name} with shape {df.shape}...")
    try:
        _compute_indicators(df)
        print(f"✅ {name} passed")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        # traceback.print_exc()

# 1. Empty DataFrame
df_empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
test_indicators(df_empty, "Empty DF")

# 2. Small DataFrame (fewer than window size)
dates = pd.date_range(start="2023-01-01", periods=10)
df_small = pd.DataFrame({
    "open": np.random.rand(10) * 100,
    "high": np.random.rand(10) * 100,
    "low": np.random.rand(10) * 100,
    "close": np.random.rand(10) * 100,
    "volume": np.random.randint(100, 1000, 10)
}, index=dates)
test_indicators(df_small, "Small DF (10 rows)")

# 3. Large DataFrame
dates = pd.date_range(start="2023-01-01", periods=300)
df_large = pd.DataFrame({
    "open": np.random.rand(300) * 100,
    "high": np.random.rand(300) * 100,
    "low": np.random.rand(300) * 100,
    "close": np.random.rand(300) * 100,
    "volume": np.random.randint(100, 1000, 300)
}, index=dates)
test_indicators(df_large, "Large DF (300 rows)")
