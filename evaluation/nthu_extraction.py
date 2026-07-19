import sys, os, glob
import cv2
import pandas as pd
import mediapipe as mp

# --- ROBUST PATH FIX ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NTHU_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/nthu"))
OUTPUT_CSV = os.path.abspath(os.path.join(SCRIPT_DIR, "nthu_features.csv"))

# Append internal project folders to sys.path
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "perception")))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..", "decision")))

from perception import compute_ear, compute_mar, LEFT_EYE, RIGHT_EYE, MOUTH
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator
from nod_detector import NodDetector
from distraction_detector import DistractionDetector
from perclos import PerclosWindow
from pose_sanity import is_plausible
from yawn_rate import YawnRateWindow

FPS_ASSUMED = 30  
MAR_YAWN_THRESHOLD = 0.6

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def extract_from_image_sequence(image_paths, label, sequence_name):
    """Processes a list of sorted images as a single continuous chronological stream."""
    pose_estimator = HeadPoseEstimator()
    calibrator = PoseCalibrator(calibration_duration=3.0)
    nod_detector = NodDetector()
    distraction_detector = DistractionDetector()
    perclos_window = PerclosWindow(window_seconds=60.0)
    yawn_window = YawnRateWindow(window_seconds=60.0)

    rows = []
    frame_i = 0
    yawn_state = False  

    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
            
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

        # Try parsing out subject ID from filename e.g. "001_glasses..."
        base_name = os.path.basename(path)
        subject_id = f"subject_{base_name[0:3]}" if base_name[0:3].isdigit() else "unknown"

        rows.append({
            "subject_id": subject_id,
            "video": sequence_name,
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
            "label": label,
        })
        
    return rows

def run_extraction():
    all_rows = []
    print(f"Target NTHU directory: {NTHU_DIR}")
    
    image_extensions = (".jpg", ".jpeg", ".png")
    
    # 1. FIND ALL DROWSY SUBFOLDERS
    drowsy_base_dir = glob.glob(os.path.join(NTHU_DIR, "**", "drowsy"), recursive=True)
    if drowsy_base_dir:
        # Find directories inside the main drowsy folder (like yawning, slowBlinkWithNodding)
        subdirs = [d for d in glob.glob(os.path.join(drowsy_base_dir[0], "*")) if os.path.isdir(d)]
        
        for subdir in subdirs:
            seq_name = os.path.basename(subdir)
            all_files = glob.glob(os.path.join(subdir, "*"))
            images = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
            
            if images:
                print(f"Processing drowsy sequence [{seq_name}] with {len(images)} images...")
                all_rows.extend(extract_from_image_sequence(images, "drowsy", f"drowsy_{seq_name}"))

    # 2. FIND NOTDROWSY DIRECT FILES
    notdrowsy_base_dir = glob.glob(os.path.join(NTHU_DIR, "**", "notdrowsy"), recursive=True)
    if notdrowsy_base_dir:
        all_files = glob.glob(os.path.join(notdrowsy_base_dir[0], "*"))
        images = sorted([f for f in all_files if f.lower().endswith(image_extensions)])
        
        if images:
            print(f"Processing alert sequence [notdrowsy] with {len(images)} images...")
            all_rows.extend(extract_from_image_sequence(images, "alert", "notdrowsy_root"))

    # 3. VERIFY AND WRITE
    if len(all_rows) == 0:
        print("Error: No data rows were extracted successfully. Check image paths.")
        return None

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    run_extraction()