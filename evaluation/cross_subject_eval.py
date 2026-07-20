"""
Cross-subject generalisation - extended to the full real feature set
now that Sheethal's head pose/PERCLOS/nod/distraction are real,
instead of the earlier EAR+MAR-only version.
"""
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Set the environment variable immediately before any MLflow initializations
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow_utils import setup_experiment, log_classification_run

# --- ROBUST PATH FIX ---
# Gets the absolute directory where this evaluation script sits (evaluation/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves absolute paths perfectly relative to the script location
CSV_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "nthu_features.csv"))

FEATURE_COLS = ["EAR", "MAR", "head_pitch", "head_yaw", "PERCLOS", "yawn_rate", "nod_count"]

def load_features(csv_path=CSV_PATH):
    print(f"Loading features from: {csv_path}")
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"Error: {csv_path} is missing or empty. Please run nthu_extraction.py first.")
    return pd.read_csv(csv_path)

def cross_subject_split(df, n_test_subjects=12, seed=42):
    subjects = df["subject_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    
    # Handle constraints gracefully if the dataset contains fewer total subjects
    actual_test_count = min(n_test_subjects, len(subjects) - 1) if len(subjects) > 1 else 0
    
    test_subjects = set(subjects[:actual_test_count])
    train_subjects = set(subjects[actual_test_count:])
    
    train_df = df[df["subject_id"].isin(train_subjects)]
    test_df = df[df["subject_id"].isin(test_subjects)]
    return train_df, test_df

def run_cross_subject():
    df = load_features()
    df["label_bin"] = (df["label"] == "drowsy").astype(int)
    train_df, test_df = cross_subject_split(df)

    print(f"Train subjects: {train_df['subject_id'].nunique()}  "
          f"Test subjects: {test_df['subject_id'].nunique()}")
    print(f"Features used: {FEATURE_COLS}")

    X_train, y_train = train_df[FEATURE_COLS].values, train_df["label_bin"].values
    X_test, y_test = test_df[FEATURE_COLS].values, test_df["label_bin"].values

    setup_experiment("cross_subject_eval")

    # Fast configuration architectures: probability=False and high cache allocations
    models = [
        ("svm_cross_subject_full", SVC(kernel="rbf", probability=False, cache_size=2000, random_state=42, verbose=True)),
        ("mlp_cross_subject_full", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42))
    ]

    for name, clf in models:
        print(f"\nTraining model: {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Conditional Score Routing: Use decision boundary scores for SVM, probabilities for MLP
        if isinstance(clf, SVC):
            y_scores = clf.decision_function(X_test)
        else:
            y_scores = clf.predict_proba(X_test)[:, 1]
            
        try:
            log_classification_run(
                run_name=name,
                params={"n_train_subjects": train_df["subject_id"].nunique(),
                        "n_test_subjects": test_df["subject_id"].nunique(),
                        "features": FEATURE_COLS},
                y_true=y_test, y_pred=y_pred, y_scores=y_scores
            )
            print(f"Successfully logged metrics for {name} to MLflow.")
        except Exception as e:
            print(f"MLflow tracking logging failed for {name}: {e}")

    # Same-subject comparison pipeline block
    print("\nRunning same-subject control baseline split...")
    X_all, y_all = df[FEATURE_COLS].values, df["label_bin"].values
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    clf_same = SVC(kernel="rbf", probability=False, cache_size=2000, random_state=42, verbose=True)
    clf_same.fit(Xs_train, ys_train)
    same_acc = accuracy_score(ys_test, clf_same.predict(Xs_test))
    
    print(f"\nSame-subject SVM accuracy (full features): {same_acc:.4f}")
    print("Compare this to cross-subject accuracy above - the gap is your generalisation result.")

if __name__ == "__main__":
    run_cross_subject()