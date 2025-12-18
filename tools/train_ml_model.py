import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, classification_report

def train_model():
    print("🧠 Starting Neural Training (Random Forest)...")
    
    # 1. Load Data
    data_path = "ml_training_data.csv"
    if not os.path.exists(data_path):
        print("❌ Training data not found. Run generate_ml_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"📊 Loaded {len(df)} samples.")
    
    # 2. Define Features & Target
    # These must match what we logged in engine.py
    feature_cols = ["rsi2", "adx", "atr_pct", "dist_sma50", "dist_sma200", "vol_rel"]
    target_col = "outcome" # 1 = Win, 0 = Loss
    
    # Clean Data (Drop rows with missing values)
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols]
    y = df[target_col]
    
    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train Model
    # Using Random Forest as it handles non-linear relationships well (like 'RSI is good unless ADX is low')
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    
    print("\n🏆 TRAINING RESULTS")
    print("===================")
    print(f"Accuracy:  {acc:.1%}")
    print(f"Precision: {prec:.1%} (When we predict Win, how often are we right?)")
    print("\nFeature Importance:")
    
    # Show what matters
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True):
        print(f"   - {name}: {imp:.4f}")

    # 5b. Confidence Threshold Analysis
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
        
    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    save_path = "models/apex_neural_v1.pkl"
    joblib.dump(model, save_path)
    print(f"\n💾 Model saved to: {save_path}")

if __name__ == "__main__":
    train_model()
