"""
4-condition ablation study - now fully real.

Previously conditions 3 and 4 (headpose, all_four) used synthetic
pitch/yaw/PERCLOS since Sheethal's Decision module wasn't built yet.
Now that nthu_extraction.py produces real head_pitch, head_yaw, and
PERCLOS columns from her actual logic, all 4 conditions run on real
data. No synthetic_stubs import anywhere in this file.
"""
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from mlflow_utils import setup_experiment
import mlflow

def load_features(csv_path="nthu_features.csv"):
    df = pd.read_csv(csv_path)
    df["label_bin"] = (df["label"] == "drowsy").astype(int)
    return df

def build_condition_features(df, condition):
    X = df[["EAR"]].copy()

    if condition in ("ear_mar", "ear_mar_headpose", "all_four"):
        X["MAR"] = df["MAR"]

    if condition in ("ear_mar_headpose", "all_four"):
        X["head_pitch"] = df["head_pitch"]
        X["head_yaw"] = df["head_yaw"]

    if condition == "all_four":
        X["PERCLOS"] = df["PERCLOS"]

    return X.values

def run_ablation():
    df = load_features()
    conditions = ["ear_only", "ear_mar", "ear_mar_headpose", "all_four"]

    setup_experiment("ablation_study")
    fig, ax = plt.subplots(figsize=(6, 6))
    results = {}

    for cond in conditions:
        X = build_condition_features(df, cond)
        y = df["label_bin"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        clf = SVC(kernel="rbf", probability=True, random_state=42)
        clf.fit(X_train, y_train)
        y_scores = clf.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        results[cond] = roc_auc

        ax.plot(fpr, tpr, label=f"{cond} (AUC={roc_auc:.3f})")

        with mlflow.start_run(run_name=f"ablation_{cond}"):
            mlflow.log_param("condition", cond)
            mlflow.log_metric("roc_auc", roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ablation study - signal combinations (all real data)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("ablation_roc_curves.png", dpi=150)
    print("\nResults:", results)

    return results

if __name__ == "__main__":
    run_ablation()
