import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from synthetic_stubs import synthetic_head_pitch, synthetic_head_yaw, synthetic_perclos
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
        print("SYNTHETIC - injecting fake head pitch/yaw for 'headpose' condition")
        X["head_pitch"] = synthetic_head_pitch(len(df), df["label_bin"].values)
        X["head_yaw"] = synthetic_head_yaw(len(df))

    if condition == "all_four":
        print("SYNTHETIC - injecting approximated PERCLOS for 'all_four' condition")
        X["PERCLOS"] = synthetic_perclos(df["EAR"].values)

    return X.values

def run_ablation():
    df = load_features()
    conditions = ["ear_only", "ear_mar", "ear_mar_headpose", "all_four"]

    setup_experiment("ablation_study")
    fig, ax = plt.subplots(figsize=(6, 6))
    results = {}

    for cond in conditions:
        is_synthetic = cond in ("ear_mar_headpose", "all_four")
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

        label = f"{cond} (AUC={roc_auc:.3f})"
        if is_synthetic:
            label += " [SYNTHETIC inputs]"
        ax.plot(fpr, tpr, label=label)

        with mlflow.start_run(run_name=f"ablation_{cond}"):
            mlflow.log_param("condition", cond)
            mlflow.log_param("contains_synthetic_features", is_synthetic)
            mlflow.log_metric("roc_auc", roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ablation study - signal combinations\n(dashed conditions use synthetic head pose/PERCLOS)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ablation_roc_curves.png", dpi=150)
    print("\nResults:", results)
    print("\nREMINDER: 'ear_mar_headpose' and 'all_four' conditions used SYNTHETIC")
    print("head pose / PERCLOS data. Re-run once Sheethal's Decision module is real")
    print("before reporting these two conditions as final results.")

    return results

if __name__ == "__main__":
    run_ablation()
