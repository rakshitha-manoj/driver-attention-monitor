"""
Live drowsiness RISK prediction demo - Raks's distinct contribution.

This is not eye-state detection (Hafsa) or head-pose tracking (Sheethal).
This runs Raks's own trained SVM classifier live, predicting a continuous
drowsiness RISK PROBABILITY (0-100%) from EAR+MAR, and plots the trend
over time. The model was trained on real NTHU data, this is what a
trained classifier adds beyond a fixed threshold or hand-tuned formula.
"""
import sys, os
import cv2
import numpy as np
import joblib
from collections import deque

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_ear, compute_mar, LEFT_EYE, RIGHT_EYE, MOUTH
import mediapipe as mp

MODEL_PATH = "drowsiness_risk_model.pkl"
TREND_LEN = 90  # ~3 seconds of trend history at 30fps

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def draw_trend_graph(frame, history, x0=10, y0=150, w=300, h=80):
    """Draws a simple line graph of risk probability over time."""
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

    warn_y = y0 + h - int(0.3 * h)
    crit_y = y0 + h - int(0.6 * h)
    cv2.line(frame, (x0, warn_y), (x0 + w, warn_y), (0, 165, 255), 1)
    cv2.line(frame, (x0, crit_y), (x0 + w, crit_y), (0, 0, 255), 1)

def run_risk_demo():
    print("Loading trained risk model...")
    clf = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    risk_history = deque(maxlen=TREND_LEN)

    print("Live drowsiness risk prediction running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        risk_prob = None
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]

            left = compute_ear(landmarks, LEFT_EYE, w, h)
            right = compute_ear(landmarks, RIGHT_EYE, w, h)
            ear = (left + right) / 2.0
            mar = compute_mar(landmarks, MOUTH, w, h)

            X = np.array([[ear, mar]])
            risk_prob = clf.predict_proba(X)[0][1]
            risk_history.append(risk_prob)

            cv2.putText(frame, f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            risk_history.append(0.0)
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if risk_prob is not None:
            risk_pct = risk_prob * 100
            risk_color = (0, 0, 255) if risk_prob > 0.6 else \
                         (0, 165, 255) if risk_prob > 0.3 else (0, 255, 0)

            cv2.putText(frame, f"DROWSINESS RISK: {risk_pct:.1f}%",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)

            bar_w = int(300 * risk_prob)
            cv2.rectangle(frame, (10, 80), (310, 100), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, 80), (10 + bar_w, 100), risk_color, -1)

        draw_trend_graph(frame, list(risk_history))
        cv2.putText(frame, "Risk trend (last ~3s)", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "ML Drowsiness Risk Predictor - Raks (trained SVM, real NTHU data)",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        cv2.imshow("Drowsiness Risk Predictor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_risk_demo()
