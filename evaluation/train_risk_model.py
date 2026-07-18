"""
Trains an SVM classifier on REAL NTHU features (EAR + MAR, no synthetic
data) and saves it so it can be loaded live for the risk-prediction demo.
Run this once nthu_features.csv exists (from nthu_extraction.py).
"""
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def train_risk_model(csv_path="nthu_features.csv", save_path="drowsiness_risk_model.pkl"):
    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "drowsy").astype(int)

    X = df[["EAR", "MAR"]].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Risk model trained — accuracy: {acc:.4f}  F1: {f1:.4f}")

    joblib.dump(clf, save_path)
    print(f"Saved to {save_path}")
    return clf

if __name__ == "__main__":
    train_risk_model()
