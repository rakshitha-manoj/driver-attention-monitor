import os, glob
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from mlflow_utils import setup_experiment, log_classification_run
import mlflow

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

# --- ROBUST PATH FIX ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

    open_pattern = os.path.join(CEW_OPEN_DIR, "**", "*.jpg")
    closed_pattern = os.path.join(CEW_CLOSED_DIR, "**", "*.jpg")
    
    open_paths = glob.glob(open_pattern, recursive=True)
    closed_paths = glob.glob(closed_pattern, recursive=True)
    
    print(f"Found {len(open_paths)} open eye images.")
    print(f"Found {len(closed_paths)} closed eye images.")
    
    for p in open_paths:
        f = extract_hog_features(p)
        if f is not None:
            X.append(f); y.append(1)
            
    for p in closed_paths:
        f = extract_hog_features(p)
        if f is not None:
            X.append(f); y.append(0)
            
    return np.array(X), np.array(y)

def run_baseline():
    print("Extracting HOG features...")
    X, y = build_dataset()
    
    if X.ndim < 2 or X.shape[0] == 0:
        print("\nError: Dataset array is empty! Feature matrix cannot be split.")
        print("Please check if the printed paths above actually contain files.")
        return None, None

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features each")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Performance optimization: probability=False and cache_size=2000
    clf = SVC(kernel="rbf", C=1.0, probability=False, cache_size=2000, random_state=42, verbose=True)
    
    print("Training SVM classifier (optimization process running)...")
    clf.fit(X_train, y_train)
    print("Training complete!")

    import joblib
    model_path = "hog_svm_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Successfully saved model locally to: {os.path.abspath(model_path)}")

    print("Running evaluation...")
    y_pred = clf.predict(X_test)
    
    # Using decision_function boundary distances since probability=False
    y_scores = clf.decision_function(X_test)

    try:
        setup_experiment("hog_svm_cew")
        
        # Unpack both the run context and the metrics dictionary cleanly
        active_run, metrics = log_classification_run(
            run_name="hog_svm_baseline",
            params={"kernel": "rbf", "C": 1.0, "img_size": IMG_SIZE, "n_train": len(X_train)},
            y_true=y_test, y_pred=y_pred, y_scores=y_scores
        )

        # Log artifact using the explicit run tracking token
        with mlflow.start_run(run_id=active_run.info.run_id):
            mlflow.log_artifact(model_path)
        print("Logged metrics and model to MLflow successfully.")
    except Exception as e:
        print(f"MLflow logging failed, but your model is safe! Error: {e}")
        metrics = None

    return clf, metrics

if __name__ == "__main__":
    run_baseline()