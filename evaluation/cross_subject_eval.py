"""
Cross-subject generalisation - extended to the full real feature set
now that Sheethal's head pose/PERCLOS/nod/distraction are real,
instead of the earlier EAR+MAR-only version.
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from mlflow_utils import setup_experiment, log_classification_run

FEATURE_COLS = ["EAR", "MAR", "head_pitch", "head_yaw", "PERCLOS", "yawn_rate", "nod_count"]

def load_features(csv_path="nthu_features.csv"):
    return pd.read_csv(csv_path)

def cross_subject_split(df, n_test_subjects=12, seed=42):
    subjects = df["subject_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    test_subjects = set(subjects[:n_test_subjects])
    train_subjects = set(subjects[n_test_subjects:])
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

    for name, clf in [
        ("svm_cross_subject_full", SVC(kernel="rbf", probability=True, random_state=42)),
        ("mlp_cross_subject_full", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)[:, 1]
        log_classification_run(
            run_name=name,
            params={"n_train_subjects": train_df["subject_id"].nunique(),
                     "n_test_subjects": test_df["subject_id"].nunique(),
                     "features": FEATURE_COLS},
            y_true=y_test, y_pred=y_pred, y_scores=y_scores
        )

    # same-subject comparison, to see the generalisation gap
    X_all, y_all = df[FEATURE_COLS].values, df["label_bin"].values
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    clf_same = SVC(kernel="rbf", probability=True, random_state=42)
    clf_same.fit(Xs_train, ys_train)
    same_acc = accuracy_score(ys_test, clf_same.predict(Xs_test))
    print(f"\nSame-subject SVM accuracy (full features): {same_acc:.4f}")
    print("Compare this to cross-subject accuracy above - the gap is your generalisation result.")

if __name__ == "__main__":
    run_cross_subject()
