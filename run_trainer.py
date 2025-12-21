import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split


def train_model() -> None:
    print("🧠 Starting Neural Training (Random Forest)...")

    data_path = "ml_training_data.csv"
    if not os.path.exists(data_path):
        print("❌ Training data not found. Generate it from the app with ML export enabled.")
        return

    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} samples.")

    feature_cols = [
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
    target_col = "outcome"

    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)

    print("\n🏆 TRAINING RESULTS")
    print("===================")
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.1%} (When we predict Win, how often are we right?)")
    print("\nFeature Importance:")

    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {name}: {imp:.4f}")

    probs = model.predict_proba(X_test)[:, 1]
    y_true = np.asarray(y_test)

    print("\n🔍 CONFIDENCE THRESHOLD ANALYSIS")
    print("==========================================")
    print("Threshold  | Win Rate   | Trades Found   ")
    print("------------------------------------------")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for thr in thresholds:
        mask = probs >= thr
        trades_found = int(np.sum(mask))
        if trades_found == 0:
            win_rate = 0.0
        else:
            win_rate = float(np.mean(y_true[mask] == 1)) * 100.0
        print(f"{thr:0.2f}       | {win_rate:0.1f}%      | {trades_found:<14d}")

    os.makedirs("models", exist_ok=True)
    save_path = "models/apex_neural_v2.pkl"
    joblib.dump(model, save_path)
    print(f"\n💾 Model saved to: {save_path}")
    print("TRAINING COMPLETE: models/apex_neural_v2.pkl is ready.")


if __name__ == "__main__":
    train_model()
