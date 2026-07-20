"""
Live drowsiness RISK prediction demo v3 - now runs Hafsa's real
PerceptionModule and Sheethal's real head-pose/PERCLOS pipeline together live.

Features a robust programmatic fallback for SVM models trained with probability=False
using plane distance sigmoid maps to prevent terminal attribute crashes.
"""
import sys
import os
import cv2
import numpy as np
import joblib
import time
from collections import deque
import mediapipe as mp

# Handle path resolutions gracefully regardless of terminal execution root location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "drowsiness_risk_model.pkl")
TREND_LEN = 90

sys.path.append(os.path.join(SCRIPT_DIR, "..", "perception"))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "decision"))

try:
    from perception import PerceptionModule
    from head_pose import HeadPoseEstimator
    from calibrator import PoseCalibrator
    from perclos import PerclosWindow
    from pose_sanity import is_plausible
    from yawn_rate import YawnRateWindow
except ImportError as e:
    print(f"[CRITICAL] Architecture path loading failed: {e}")
    print("Verify that 'perception' and 'decision' folders are structured properly in root.")
    sys.exit(1)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

def draw_trend_graph(frame, history, x0=10, y0=180, w=300, h=80):
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 1)
    if len(history) < 2:
        return
    pts = []
    for i, val in enumerate(history):
        px = x0 + int(i / TREND_LEN * w)
        py = y0 + h - int(val * h)
        pts.append((px, py))
    for i in range(1, len(pts)):
        color = (0, 0, 255) if history[i] > 0.6 else \
                (0, 165, 255) if history[i] > 0.3 else (0, 255, 0)
        cv2.line(frame, pts[i-1], pts[i], color, 2)

def run_risk_demo():
    if not os.path.exists(MODEL_PATH):
        print(f"[CRITICAL] Target binary model payload missing at: {MODEL_PATH}")
        print("Train weights via train_risk_model.py before starting live demo window.")
        sys.exit(1)

    print("Loading trained risk model...")
    clf = joblib.load(MODEL_PATH)
    
    # CRITICAL FALLBACK CHECK: Verify model classification capabilities
    has_probability = True
    try:
        # Check if the internal sklearn availability wrapper flag is active
        _ = clf.predict_proba(np.zeros((1, 6)))
    except AttributeError:
        print("[WARN] SVM model was trained with probability=False.")
        print("       Activating Sigmoid Decision-Function approximation fallbacks safely...")
        has_probability = False
    except Exception:
        # Catch secondary instantiation checks gracefully
        pass

    perception = PerceptionModule()
    pose_estimator = HeadPoseEstimator()
    calibrator = PoseCalibrator(calibration_duration=3.0)
    perclos_window = PerclosWindow(window_seconds=60.0)
    yawn_window = YawnRateWindow(window_seconds=60.0)
    _last_yawn_count = 0

    cap = cv2.VideoCapture(0)
    risk_history = deque(maxlen=TREND_LEN)

    print("Live drowsiness risk prediction (full feature set) running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Mirror naturally for interactive desktop rendering loops
        frame = cv2.flip(frame, 1)
        now = time.time()

        perception_output = perception.process_frame(frame.copy())
        ear = perception_output["EAR"]
        mar = perception_output["MAR"]
        is_closed = perception_output["blink_state"] == "closed"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        pitch = yaw = 0.0
        if results.multi_face_landmarks:
            landmarks = face_results = results.multi_face_landmarks[0].landmark
            pose = pose_estimator.estimate_pose(landmarks, frame)
            if pose is not None:
                if not calibrator.is_calibrated:
                    calibrator.calibrate(pose)
                pose = calibrator.get_relative_pose(pose)
                if is_plausible(pose["pitch"], pose["yaw"], pose["roll"]):
                    pitch, yaw = pose["pitch"], pose["yaw"]

        perclos = perclos_window.update(is_closed if ear is not None else False, now)

        yawn_count = perception.yawn_count
        if yawn_count > _last_yawn_count:
            for _ in range(yawn_count - _last_yawn_count):
                yawn_window.add_yawn(now)
            _last_yawn_count = yawn_count
        yawn_rate = yawn_window.rate_per_minute(now)

        risk_prob = None
        if ear is not None and mar is not None:
            X = np.array([[ear, mar, pitch, yaw, perclos, yawn_rate]])
            
            # Tiered execution handler branch logic
            if has_probability:
                risk_prob = clf.predict_proba(X)[0][1]
            else:
                # Extract hyper-plane metric distance
                decision_score = clf.decision_function(X)[0]
                # Map smoothly into normalized [0.0 - 1.0] probability values via a standard Logistic Sigmoid function
                risk_prob = 1.0 / (1.0 + np.exp(-decision_score))
                
            risk_history.append(risk_prob)

            cv2.putText(frame, f"EAR:{ear:.2f} MAR:{mar:.2f} Pitch:{pitch:.1f} Yaw:{yaw:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"PERCLOS:{perclos*100:.0f}% YawnRate:{yawn_rate:.1f}/min",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            risk_history.append(0.0)
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if risk_prob is not None:
            risk_pct = risk_prob * 100
            risk_color = (0, 0, 255) if risk_prob > 0.6 else \
                         (0, 165, 255) if risk_prob > 0.3 else (0, 255, 0)
            cv2.putText(frame, f"DROWSINESS RISK: {risk_pct:.1f}%",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
            bar_w = int(300 * risk_prob)
            cv2.rectangle(frame, (10, 105), (310, 125), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 105), (10 + bar_w, 125), risk_color, -1)

        draw_trend_graph(frame, list(risk_history))
        cv2.putText(frame, "ML Drowsiness Risk (full features) - Raks",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        cv2.imshow("Drowsiness Risk Predictor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_risk_demo()