import os, glob
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from mlflow_utils import setup_experiment, log_classification_run
import mlflow

# ADJUST once you've inspected the actual CEW folder layout
CEW_OPEN_DIR = "../data/cew/OpenFace"
CEW_CLOSED_DIR = "../data/cew/ClosedFace"
IMG_SIZE = (64, 64)

def extract_hog_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), block_norm="L2-Hys")
    return features

def build_dataset():
    X, y = [], []
    for p in glob.glob(os.path.join(CEW_OPEN_DIR, "*.jpg")):
        f = extract_hog_features(p)
        if f is not None:
            X.append(f); y.append(1)  # open
    for p in glob.glob(os.path.join(CEW_CLOSED_DIR, "*.jpg")):
        f = extract_hog_features(p)
        if f is not None:
            X.append(f); y.append(0)  # closed
    return np.array(X), np.array(y)

def run_baseline():
    print("Extracting HOG features...")
    X, y = build_dataset()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)[:, 1]

    setup_experiment("hog_svm_cew")
    metrics = log_classification_run(
        run_name="hog_svm_baseline",
        params={"kernel": "rbf", "C": 1.0, "img_size": IMG_SIZE, "n_train": len(X_train)},
        y_true=y_test, y_pred=y_pred, y_scores=y_scores
    )

    import joblib
    joblib.dump(clf, "hog_svm_model.pkl")
    with mlflow.start_run(run_id=mlflow.last_active_run().info.run_id):
        mlflow.log_artifact("hog_svm_model.pkl")

    return clf, metrics

if __name__ == "__main__":
    run_baseline()
