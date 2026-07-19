import sys, os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Set the environment variable immediately before any MLflow initializations
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from mlflow_utils import setup_experiment, log_classification_run

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_mar, MOUTH  # reuse exact logic, unmodified

# --- ROBUST PATH FIX ---
# Anchor the path dynamically to the directory where this validation script sits
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YAWDD_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/yawdd"))

FRAME_SAMPLE_RATE = 3  # denser sampling helps catch the actual yawn peak

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def get_peak_mar(video_path):
    cap = cv2.VideoCapture(video_path)
    max_mar = 0.0
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
        mar = compute_mar(landmarks, MOUTH, w, h)
        max_mar = max(max_mar, mar)
    cap.release()
    return max_mar

def run_validation(threshold=0.6):
    print(f"Target dataset directory: {YAWDD_DIR}")
    print(f"Directory exists: {os.path.exists(YAWDD_DIR)}")

    # Broad recursive search to find videos across all nested structures
    yawn_pattern = os.path.join(YAWDD_DIR, "**", "*yawn*.avi")
    normal_pattern = os.path.join(YAWDD_DIR, "**", "*normal*.avi")
    
    yawn_videos = glob.glob(yawn_pattern, recursive=True)
    normal_videos = glob.glob(normal_pattern, recursive=True)

    print(f"Discovered {len(yawn_videos)} yawn videos.")
    print(f"Discovered {len(normal_videos)} normal videos.")

    if len(yawn_videos) == 0 and len(normal_videos) == 0:
        print("Error: No videos matched the name patterns in the directory structure.")
        return

    yawn_peaks, normal_peaks = [], []
    for v in yawn_videos:
        peak = get_peak_mar(v)
        yawn_peaks.append(peak)
        print(f"  {os.path.basename(v)}: peak MAR = {peak:.3f}")
        
    for v in normal_videos:
        normal_peaks.append(get_peak_mar(v))

    print(f"\nVideo-level peaks - yawn videos: {len(yawn_peaks)}  normal videos: {len(normal_peaks)}")

    if len(yawn_peaks) == 0 and len(normal_peaks) == 0:
        print("Error: Zero frames were processed successfully from the video list.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_peaks, bins=20, alpha=0.6, label="normal videos (peak MAR)")
    ax.hist(yawn_peaks, bins=20, alpha=0.6, label="yawn videos (peak MAR)")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold}")
    ax.legend()
    ax.set_title("Peak MAR per video - YawDD (video-level)")
    plt.savefig("yawdd_mar_video_level.png", dpi=150)
    plt.close(fig)

    y_true = [1] * len(yawn_peaks) + [0] * len(normal_peaks)
    y_scores = yawn_peaks + normal_peaks
    y_pred = [1 if m > threshold else 0 for m in y_scores]

    try:
        setup_experiment("yawdd_mar_validation")
        
        # Unpack the active run context and the metrics dictionary cleanly
        active_run, metrics = log_classification_run(
            run_name=f"mar_threshold_{threshold}_video_level",
            params={"threshold": threshold, "level": "video (peak MAR)"},
            y_true=y_true, y_pred=y_pred, y_scores=y_scores
        )
        print("Logged validation run results to MLflow successfully.")
    except Exception as e:
        print(f"MLflow tracking encountered an error: {e}")

if __name__ == "__main__":
    run_validation(threshold=0.6)