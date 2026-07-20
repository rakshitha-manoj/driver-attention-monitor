"""
4-condition ablation study - now fully real.

Previously conditions 3 and 4 (headpose, all_four) used synthetic
pitch/yaw/PERCLOS since Sheethal's Decision module wasn't built yet.
Now that nthu_extraction.py produces real head_pitch, head_yaw, and
PERCLOS columns from her actual logic, all 4 conditions run on real
data. No synthetic_stubs import anywhere in this file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Set the environment variable immediately before any MLflow initializations
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow_utils import setup_experiment
import mlflow

# --- ROBUST PATH FIX ---
# Gets the absolute directory where this script sits (evaluation/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves absolute paths perfectly relative to the script location
CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "nthu_features.csv"))
PLOT_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "ablation_roc_curves.png"))

def load_features(csv_path=CSV_PATH):
    print(f"Loading features from: {csv_path}")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"Error: {csv_path} is missing or empty. Please run nthu_extraction.py first.")
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
        print(f"\nEvaluating condition loop branch: [{cond}]...")
        X = build_condition_features(df, cond)
        y = df["label_bin"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Performance optimization: probability=False and cache_size=2000 to eliminate evaluation block hanging
        clf = SVC(kernel="rbf", probability=False, cache_size=2000, random_state=42, verbose=True)
        clf.fit(X_train, y_train)
        
        # Corrected: using decision_function since probability=False
        y_scores = clf.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        results[cond] = roc_auc

        ax.plot(fpr, tpr, label=f"{cond} (AUC={roc_auc:.3f})")

        try:
            with mlflow.start_run(run_name=f"ablation_{cond}"):
                mlflow.log_param("condition", cond)
                mlflow.log_metric("roc_auc", roc_auc)
            print(f"Logged run metrics for condition [{cond}] successfully.")
        except Exception as e:
            print(f"MLflow tracking encountered an error for {cond}: {e}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ablation study - signal combinations (all real data)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved combined comparison plot to {PLOT_SAVE_PATH}")
    print("\nFinal Results:", results)

    return results

if __name__ == "__main__":
    run_ablation()