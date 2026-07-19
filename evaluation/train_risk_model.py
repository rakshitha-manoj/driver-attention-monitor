"""
Trains and saves the drowsiness risk classifier - now trained on the
full real feature set (EAR, MAR, head_pitch, head_yaw, PERCLOS,
yawn_rate) since Sheethal's Decision module is real. Previously
EAR+MAR only. Run after nthu_extraction.py.
"""
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

FEATURE_COLS = ["EAR", "MAR", "head_pitch", "head_yaw", "PERCLOS", "yawn_rate"]

def train_risk_model(csv_path="nthu_features.csv", save_path="drowsiness_risk_model.pkl"):
    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "drowsy").astype(int)

    X = df[FEATURE_COLS].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Risk model trained on {FEATURE_COLS}")
    print(f"Accuracy: {acc:.4f}  F1: {f1:.4f}")

    joblib.dump(clf, save_path)
    print(f"Saved to {save_path}")
    return clf

if __name__ == "__main__":
    train_risk_model()
