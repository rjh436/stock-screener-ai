import os
import sys
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import joblib

# Add project root to path so `python tools/run_ai_scanner.py` works reliably.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import data.loader as loader
import data.indices as indices
from execution import engine
from strategies.strategy_loader import load_strategies

MODEL_PATH = "models/apex_neural_v1.pkl"
STRATEGY_PATH = "config/training_strategies.json"
CONFIDENCE_THRESHOLD = 0.60
LOOKBACK_DAYS = 300

FEATURE_COLS = ["rsi2", "adx", "atr_pct", "dist_sma50", "dist_sma200", "vol_rel"]


def get_latest_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    try:
        # Ensure indicators are present
        if df is None or df.empty:
            return None
        if "rsi2" not in df.columns:
            df = engine._compute_indicators(df)

        row = df.iloc[-1]
        close = row["close"]
        if close == 0:
            return None

        # Features
        rsi2 = row.get("rsi2", 50)
        adx = row.get("adx", 0)
        atr_pct = (row.get("atr14", 0) / close) if close > 0 else 0

        sma50 = row.get("sma50", close)
        sma200 = row.get("sma200", close)
        dist_sma50 = (close - sma50) / close
        dist_sma200 = (close - sma200) / close

        vol_rel = row.get("volume", 0) / (row.get("vol_ma20", 1) + 1)

        features = np.array([rsi2, adx, atr_pct, dist_sma50, dist_sma200, vol_rel], dtype=float)
        if not np.all(np.isfinite(features)):
            return None

        # Return as 2D array
        return features.reshape(1, -1)
    except Exception:
        return None


def _fetch_symbol(sym: str) -> Tuple[str, Optional[pd.DataFrame]]:
    try:
        df = loader.fetch_single_symbol(sym, days=LOOKBACK_DAYS, force_fresh=True)
        if df is None or df.empty:
            return sym, None
        return sym, df
    except Exception:
        return sym, None


def _load_strategy() -> Optional[object]:
    if not os.path.exists(STRATEGY_PATH):
        print(f"❌ Strategy config not found: {STRATEGY_PATH}")
        return None

    try:
        with open(STRATEGY_PATH, "r") as f:
            configs = json.load(f) or []
    except Exception as e:
        print(f"❌ Failed to read strategy config: {e}")
        return None

    if not configs:
        print(f"❌ No strategies found in: {STRATEGY_PATH}")
        return None

    strategies = load_strategies([configs[0]])
    if not strategies:
        print("❌ Failed to instantiate strategy.")
        return None

    strategy = strategies[0]

    # Allow calling `strategy.entry(df, -1)` for the latest bar.
    try:
        params = getattr(strategy, "genome", None)
        if isinstance(params, dict):
            params["warmup_bars"] = -1
    except Exception:
        pass

    return strategy


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    print(f"🧠 Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    print("📋 Loading strategy...")
    strategy = _load_strategy()
    if strategy is None:
        return

    print("📥 Loading S&P 1500 symbols...")
    symbols = [s for s in (indices.get_index_symbols("S&P 1500") or []) if s]
    if not symbols:
        print("❌ No symbols returned for S&P 1500.")
        return

    print(f"⚡ Fetching {len(symbols)} symbols ({LOOKBACK_DAYS}d lookback) in parallel (joblib)...")
    fetched: List[Tuple[str, Optional[pd.DataFrame]]] = joblib.Parallel(n_jobs=-1, backend="threading")(
        joblib.delayed(_fetch_symbol)(sym) for sym in symbols
    )
    data: Dict[str, pd.DataFrame] = {sym: df for sym, df in fetched if df is not None and not df.empty}
    print(f"✅ Data loaded: {len(data)} symbols")

    hits: List[Dict] = []

    for sym, df in data.items():
        try:
            df_ind = engine._compute_indicators(df.copy())
            df_ind["ticker"] = sym
        except Exception:
            continue

        try:
            signal = strategy.entry(df_ind, -1)
        except Exception:
            continue

        if not signal:
            continue

        features = get_latest_features(df_ind)
        if features is None:
            continue

        try:
            prob = float(model.predict_proba(features)[0][1])
        except Exception:
            continue

        if prob < CONFIDENCE_THRESHOLD:
            continue

        row = df_ind.iloc[-1]
        try:
            price = float(row.get("close", np.nan))
        except Exception:
            continue
        if not np.isfinite(price) or price <= 0:
            continue

        hits.append(
            {
                "Symbol": sym,
                "Signal Price": price,
                "Probability": prob,
                "Sector": engine.get_sector(sym),
            }
        )

    if not hits:
        print(f"🔎 No high-confidence hits found (P >= {CONFIDENCE_THRESHOLD:.2f}).")
        return

    df_hits = pd.DataFrame(hits).sort_values("Probability", ascending=False)
    df_hits["Signal Price"] = df_hits["Signal Price"].map(lambda x: f"{x:,.2f}")
    df_hits["Probability"] = df_hits["Probability"].map(lambda x: f"{x:.2f}")

    print(f"\n✅ AI Scanner Hits (P >= {CONFIDENCE_THRESHOLD:.2f}) — {len(df_hits)} candidates")
    print(df_hits.to_string(index=False))


if __name__ == "__main__":
    main()
