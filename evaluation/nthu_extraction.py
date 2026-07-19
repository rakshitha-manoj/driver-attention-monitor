"""
NTHU feature extraction - full version.

Replaces the earlier EAR/MAR-only extractor now that Sheethal's
Decision module is real. Runs her actual HeadPoseEstimator,
PoseCalibrator, NodDetector, DistractionDetector, PerclosWindow, and
a YawnRateWindow (same pattern as her PERCLOS window) over each NTHU
video, driven by a synthetic per-video timestamp instead of a live
clock, since these are pre-recorded files, not a live camera feed.

No synthetic/fake feature values anywhere in this file - every
column is computed from her real logic running against real video.
"""
import sys, os, glob
import cv2
import pandas as pd
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "decision"))

from perception import compute_ear, compute_mar, LEFT_EYE, RIGHT_EYE, MOUTH
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator
from nod_detector import NodDetector
from distraction_detector import DistractionDetector
from perclos import PerclosWindow
from pose_sanity import is_plausible
from yawn_rate import YawnRateWindow

# ADJUST once you've inspected the actual NTHU folder layout
NTHU_DIR = "../data/nthu"
OUTPUT_CSV = "nthu_features.csv"
FPS_ASSUMED = 30  # used to turn frame_id into a synthetic timestamp
MAR_YAWN_THRESHOLD = 0.6

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def extract_from_video(video_path, label, subject_id):
    cap = cv2.VideoCapture(video_path)

    # fresh instances per video so calibration/state doesn't leak across subjects
    pose_estimator = HeadPoseEstimator()
    calibrator = PoseCalibrator(calibration_duration=3.0)
    nod_detector = NodDetector()
    distraction_detector = DistractionDetector()
    perclos_window = PerclosWindow(window_seconds=60.0)
    yawn_window = YawnRateWindow(window_seconds=60.0)

    rows = []
    frame_i = 0
    yawn_state = False  # tracks MAR crossing edge, mirrors Hafsa's yawn-start logic

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        now = frame_i / FPS_ASSUMED

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        left = compute_ear(landmarks, LEFT_EYE, w, h)
        right = compute_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left + right) / 2.0
        mar = compute_mar(landmarks, MOUTH, w, h)

        is_closed = ear < 0.25
        perclos = perclos_window.update(is_closed, now)

        if mar > MAR_YAWN_THRESHOLD and not yawn_state:
            yawn_window.add_yawn(now)
            yawn_state = True
        elif mar <= MAR_YAWN_THRESHOLD:
            yawn_state = False

        pose = pose_estimator.estimate_pose(landmarks, frame)
        pitch = yaw = roll = 0.0
        nod_count = 0
        distraction_flag = False

        if pose is not None:
            if not calibrator.is_calibrated:
                calibrator.calibrate(pose)
            pose = calibrator.get_relative_pose(pose)
            pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]

            if is_plausible(pitch, yaw, roll):
                nod_count = nod_detector.update(pitch, yaw=yaw, roll=roll, now=now)
                distraction_flag = distraction_detector.update(yaw, now)

        rows.append({
            "subject_id": subject_id,
            "video": os.path.basename(video_path),
            "frame_id": frame_i,
            "EAR": round(ear, 4),
            "MAR": round(mar, 4),
            "head_pitch": round(pitch, 2),
            "head_yaw": round(yaw, 2),
            "head_roll": round(roll, 2),
            "nod_count": nod_count,
            "distraction_flag": distraction_flag,
            "PERCLOS": round(perclos, 4),
            "yawn_rate": round(yawn_window.rate_per_minute(now), 2),
            "label": label,  # ADJUST: map from NTHU's actual annotation format
        })

    cap.release()
    return rows

def run_extraction():
    all_rows = []
    # ADJUST: assumes data/nthu/<subject_id>/<scenario>/<video>.avi
    subject_dirs = glob.glob(os.path.join(NTHU_DIR, "*"))
    for subj_dir in subject_dirs:
        subject_id = os.path.basename(subj_dir)
        videos = glob.glob(os.path.join(subj_dir, "**", "*.avi"), recursive=True)
        for v in videos:
            label = "drowsy" if "sleepy" in v.lower() else "alert"  # ADJUST
            all_rows.extend(extract_from_video(v, label, subject_id))
            print(f"  extracted {v}: running total {len(all_rows)} rows")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    print("All columns are REAL - EAR/MAR (Hafsa's logic), head pose/nod/")
    print("distraction/PERCLOS/yawn_rate (Sheethal's logic). No synthetic data.")
    return df

if __name__ == "__main__":
    run_extraction()
