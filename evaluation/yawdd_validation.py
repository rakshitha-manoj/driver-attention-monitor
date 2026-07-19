import sys, os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_mar, MOUTH

from mlflow_utils import setup_experiment, log_classification_run

# ADJUST once you've inspected the actual YawDD folder layout
YAWDD_DIR = "../data/yawdd"
FRAME_SAMPLE_RATE = 5  # extract every Nth frame to keep this fast

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def extract_mar_series(video_path):
    cap = cv2.VideoCapture(video_path)
    mars = []
    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        if frame_i % FRAME_SAMPLE_RATE != 0:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        mars.append(compute_mar(landmarks, MOUTH, w, h))
    cap.release()
    return mars

def run_validation(threshold=0.6):
    # ADJUST: label videos by filename convention once you know it
    yawn_videos = glob.glob(os.path.join(YAWDD_DIR, "**", "*yawn*.avi"), recursive=True)
    normal_videos = glob.glob(os.path.join(YAWDD_DIR, "**", "*normal*.avi"), recursive=True)

    yawn_mars, normal_mars = [], []
    for v in yawn_videos:
        yawn_mars.extend(extract_mar_series(v))
    for v in normal_videos:
        normal_mars.extend(extract_mar_series(v))

    print(f"MAR samples - yawn videos: {len(yawn_mars)}  normal videos: {len(normal_mars)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_mars, bins=40, alpha=0.6, label="normal (ground truth)")
    ax.hist(yawn_mars, bins=40, alpha=0.6, label="yawn video frames (ground truth)")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    ax.legend()
    ax.set_title("MAR distribution - YawDD")
    plt.savefig("yawdd_mar_distribution.png", dpi=150)

    y_true = [0] * len(normal_mars) + [1] * len(yawn_mars)
    y_scores = normal_mars + yawn_mars
    y_pred = [1 if m > threshold else 0 for m in y_scores]

    setup_experiment("yawdd_mar_validation")
    log_classification_run(
        run_name=f"mar_threshold_{threshold}",
        params={"threshold": threshold, "sample_rate": FRAME_SAMPLE_RATE},
        y_true=y_true, y_pred=y_pred, y_scores=y_scores
    )

if __name__ == "__main__":
    run_validation(threshold=0.6)
