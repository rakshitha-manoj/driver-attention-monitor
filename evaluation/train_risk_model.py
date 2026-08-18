"""
Trains and saves the drowsiness risk classifier - now trained on the
full real feature set (EAR, MAR, head_pitch, head_yaw, PERCLOS,
yawn_rate) since Sheethal's Decision module is real. Previously
EAR+MAR only. Run after nthu_extraction.py.
"""
import os
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- ROBUST PATH FIX ---
# Gets the absolute directory where this training script sits (evaluation/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves absolute paths perfectly relative to the script location
CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "nthu_features.csv"))
SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "drowsiness_risk_model.pkl"))

FEATURE_COLS = ["EAR", "MAR", "head_pitch", "head_yaw", "PERCLOS", "yawn_rate"]

def train_risk_model(csv_path=CSV_PATH, save_path=SAVE_PATH):
    print(f"Loading features from: {csv_path}")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"Error: {csv_path} is missing or empty. Please run nthu_extraction.py completely first.")
        return None

    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "drowsy").astype(int)

    X = df[FEATURE_COLS].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Optimization: probability=False, cache_size=2000, and verbose=True to prevent hour-long silent hangs
    clf = SVC(kernel="rbf", probability=False, cache_size=2000, random_state=42, verbose=True)
    
    print("Training drowsiness risk model classifier...")
    clf.fit(X_train, y_train)
    print("Training complete!")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nRisk model successfully trained on {FEATURE_COLS}")
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")

    joblib.dump(clf, save_path)
    print(f"Saved model configuration to {save_path}")
    return clf

if __name__ == "__main__":
    train_risk_model()