import sys, os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Set the environment variable immediately before any other MLflow processes call tracking hooks
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow_utils import setup_experiment, log_classification_run

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_ear, LEFT_EYE, RIGHT_EYE  # reuse Hafsa's exact logic, unmodified

# --- ROBUST PATH FIX ---
# Anchor paths dynamically to the location of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CEW_OPEN_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/cew/open"))
CEW_CLOSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/cew/closed"))

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5
)

def get_ear_from_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img.shape[:2]
    left = compute_ear(landmarks, LEFT_EYE, w, h)
    right = compute_ear(landmarks, RIGHT_EYE, w, h)
    return (left + right) / 2.0

def run_validation(threshold=0.25):
    print(f"Checking Path: {CEW_OPEN_DIR} -> {'EXISTS' if os.path.exists(CEW_OPEN_DIR) else 'NOT FOUND'}")
    print(f"Checking Path: {CEW_CLOSED_DIR} -> {'EXISTS' if os.path.exists(CEW_CLOSED_DIR) else 'NOT FOUND'}")

    # Use recursive matching with '**/*.jpg' to catch images nested inside subdirectories
    open_paths = glob.glob(os.path.join(CEW_OPEN_DIR, "**", "*.jpg"), recursive=True)
    closed_paths = glob.glob(os.path.join(CEW_CLOSED_DIR, "**", "*.jpg"), recursive=True)

    open_ears, closed_ears = [], []
    for p in open_paths:
        ear = get_ear_from_image(p)
        if ear is not None:
            open_ears.append(ear)
    for p in closed_paths:
        ear = get_ear_from_image(p)
        if ear is not None:
            closed_ears.append(ear)

    print(f"Valid samples - open: {len(open_ears)}  closed: {len(closed_ears)}")

    if len(open_ears) == 0 and len(closed_ears) == 0:
        print("Error: No images were processed successfully by MediaPipe Face Mesh.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(open_ears, bins=40, alpha=0.6, label="open (ground truth)")
    ax.hist(closed_ears, bins=40, alpha=0.6, label="closed (ground truth)")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("EAR")
    ax.legend()
    ax.set_title("EAR distribution - CEW")
    plt.savefig("cew_ear_distribution.png", dpi=150)
    plt.close(fig)

    y_true = [1] * len(open_ears) + [0] * len(closed_ears)
    y_scores = open_ears + closed_ears
    y_pred = [1 if e > threshold else 0 for e in y_scores]

    try:
        setup_experiment("cew_ear_validation")
        
        # Unpack the active run object and metrics returned from the tracking module
        active_run, metrics = log_classification_run(
            run_name=f"ear_threshold_{threshold}",
            params={"threshold": threshold, "n_open": len(open_ears), "n_closed": len(closed_ears)},
            y_true=y_true, y_pred=y_pred, y_scores=y_scores
        )
        print("Logged validation metrics and figures to MLflow successfully.")
    except Exception as e:
        print(f"MLflow tracking encountered an error: {e}")

if __name__ == "__main__":
    run_validation(threshold=0.25)