import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURE_COLS = [
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
    "vix_rel20",
]
SUCCESS_PCT = 5.0
FAILURE_PCT = -7.5


def main() -> None:
    data_path = "ml_training_data.csv"
    if not os.path.exists(data_path):
        print("ERROR: Training data not found: ml_training_data.csv")
        return

    df = pd.read_csv(data_path)
    missing = [col for col in FEATURE_COLS + ["profit_pct"] if col not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return

    df = df.dropna(subset=FEATURE_COLS + ["profit_pct"]).copy()
    df["profit_pct"] = pd.to_numeric(df["profit_pct"], errors="coerce")
    df = df.dropna(subset=["profit_pct"])
    if df.empty:
        print("ERROR: No valid rows after filtering for required columns.")
        return

    success_mask = df["profit_pct"] > SUCCESS_PCT
    failure_mask = df["profit_pct"] < FAILURE_PCT
    label_mask = success_mask | failure_mask

    df = df.loc[label_mask].copy()
    if df.empty:
        print("ERROR: No rows left after applying label thresholds.")
        return

    y = (df["profit_pct"] > SUCCESS_PCT).astype(int)
    X = df[FEATURE_COLS].astype(float)

    class_counts = y.value_counts().to_dict()
    min_class = min(class_counts.values()) if class_counts else 0
    if min_class < 2:
        print("ERROR: Not enough samples per class for calibration.")
        print(f"Class counts: {class_counts}")
        return

    cv_folds = min(5, min_class)
    if cv_folds < 2:
        print("ERROR: Need at least 2 folds for calibration.")
        print(f"Class counts: {class_counts}")
        return

    base_model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.07,
        max_iter=400,
        random_state=42,
    )

    model = CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=cv_folds,
    )
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    save_path = "models/apex_neural_v6.pkl"
    joblib.dump(model, save_path)

    print("Training complete.")
    print(f"Samples used: {len(df)}")
    print(f"Class counts: {class_counts}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
