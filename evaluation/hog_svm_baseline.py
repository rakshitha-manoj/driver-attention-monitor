import os, glob
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, accuracy_score, f1_score

from mlflow_utils import setup_experiment, log_classification_run
import mlflow

# Allow local file system tracking for MLflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# --- ROBUST PATH FIX ---
# Gets the absolute directory where this script actually sits (evaluation/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Resolves relative paths perfectly, regardless of where you call the command from
CEW_OPEN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/cew/open"))
CEW_CLOSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/cew/closed"))
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
    
    print(f"Checking Path: {CEW_OPEN_DIR} -> {'EXISTS' if os.path.exists(CEW_OPEN_DIR) else 'NOT FOUND'}")
    print(f"Checking Path: {CEW_CLOSED_DIR} -> {'EXISTS' if os.path.exists(CEW_CLOSED_DIR) else 'NOT FOUND'}")

    # Use recursive=True and '**/*.jpg' to catch images nested inside subdirectories
    open_pattern = os.path.join(CEW_OPEN_DIR, "**", "*.jpg")
    closed_pattern = os.path.join(CEW_CLOSED_DIR, "**", "*.jpg")
    
    open_paths = glob.glob(open_pattern, recursive=True)
    closed_paths = glob.glob(closed_pattern, recursive=True)
    
    print(f"Found {len(open_paths)} open eye images.")
    print(f"Found {len(closed_paths)} closed eye images.")
    
    for p in open_paths:
        f = extract_hog_features(p)
        if f is not None:
            X.append(f); y.append(1)  # open
            
    for p in closed_paths:
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

    import joblib
    joblib.dump(clf, "hog_svm_model.pkl")
    print("Successfully saved model locally to hog_svm_model.pkl")

    # Log metrics AND the model artifact in the SAME run, no need to
    # look up "last active run" after closing it
    with mlflow.start_run(run_name="hog_svm_baseline") as run:
        mlflow.log_params({"kernel": "rbf", "C": 1.0, "img_size": IMG_SIZE, "n_train": len(X_train)})

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        print(f"[hog_svm_baseline] acc={acc:.4f}  f1={f1:.4f}")

        mlflow.log_artifact("hog_svm_model.pkl")
        print("Model artifact logged to MLflow successfully.")

    return clf, {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    run_baseline()