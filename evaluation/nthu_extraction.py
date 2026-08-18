import sys, os, glob
import cv2
import pandas as pd
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_ear, compute_mar, LEFT_EYE, RIGHT_EYE, MOUTH

# ADJUST once you've inspected the actual NTHU folder layout
NTHU_DIR = "../data/nthu"
OUTPUT_CSV = "nthu_features.csv"
FRAME_SAMPLE_RATE = 3

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def extract_from_video(video_path, label, subject_id):
    cap = cv2.VideoCapture(video_path)
    rows = []
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
        left = compute_ear(landmarks, LEFT_EYE, w, h)
        right = compute_ear(landmarks, RIGHT_EYE, w, h)
        ear = (left + right) / 2.0
        mar = compute_mar(landmarks, MOUTH, w, h)
        rows.append({
            "subject_id": subject_id,
            "video": os.path.basename(video_path),
            "frame_id": frame_i,
            "EAR": round(ear, 4),
            "MAR": round(mar, 4),
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
    return df

if __name__ == "__main__":
    run_extraction()
