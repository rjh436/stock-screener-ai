import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def main() -> None:
    data_path = "ml_training_data.csv"
    if not os.path.exists(data_path):
        print("❌ Training data not found: ml_training_data.csv")
        return

    df = pd.read_csv(data_path)

    features = [
        "rsi2",
        "adx",
        "atr_pct",
        "dist_sma50",
        "dist_sma200",
        "vol_rel",
        "rs_ratio",
        "rs_trend",
        "rs_mom20",
        "spy_regime",
    ]

    df = df.dropna(subset=features + ["profit_pct"])
    df = df[(df["profit_pct"] > 0.5) | (df["profit_pct"] < -1.5)]
    if df.empty:
        print("❌ No decisive trades after filtering. Check profit_pct thresholds.")
        return

    X = df[features]
    y = (df["profit_pct"] > 0.5).astype(int)

    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.07,
        max_iter=400,
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    save_path = "models/apex_neural_v2.pkl"
    joblib.dump(model, save_path)
    print(f"✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()
