"""
Live robustness stress-test demo — Raks's distinct contribution.
Unlike Hafsa's detection demo or Sheethal's tracking demo, this shows
whether the system SURVIVES real-world degradation: blur, glare,
lighting shifts, partial occlusion — applied live, in real time, with
visible confidence/accuracy readout.

Controls (press during demo):
  1 = clean (no degradation)
  2 = motion blur
  3 = brightness spike (simulated glare)
  4 = dim lighting
  5 = partial occlusion (simulated glasses glare)
  Q = quit
"""
import cv2
import numpy as np
import joblib
from skimage.feature import hog

MODEL_PATH = "hog_svm_model.pkl"
IMG_SIZE = (64, 64)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ── Degradation transforms, applied live to the raw frame ──

def degrade_clean(frame):
    return frame, "CLEAN"

def degrade_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0), "MOTION BLUR"

def degrade_glare(frame):
    return np.clip(frame.astype(np.int16) + 90, 0, 255).astype(np.uint8), "SIMULATED GLARE"

def degrade_dim(frame):
    return np.clip(frame.astype(np.int16) - 90, 0, 255).astype(np.uint8), "DIM LIGHTING"

def degrade_occlusion(frame):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (w//3, h//4), (2*w//3, h//2), (255, 255, 255), -1)
    return out, "SIMULATED OCCLUSION / GLASSES GLARE"

DEGRADATIONS = {
    ord('1'): degrade_clean,
    ord('2'): degrade_blur,
    ord('3'): degrade_glare,
    ord('4'): degrade_dim,
    ord('5'): degrade_occlusion,
}

def extract_hog(img_gray):
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    return hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")

def run_stress_test_demo():
    print("Loading trained HOG+SVM model...")
    clf = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    print("Live robustness stress-test demo running.")
    print("Press 1-5 to toggle degradation, Q to quit.")

    current_mode = ord('1')
    confidence_history = []
    HISTORY_LEN = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in DEGRADATIONS:
            current_mode = key
            confidence_history = []

        degrade_fn = DEGRADATIONS[current_mode]
        degraded_frame, condition_label = degrade_fn(frame)

        gray = cv2.cvtColor(degraded_frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        frame_confidences = []
        for (x, y, w, h) in eyes:
            eye_crop = gray[y:y+h, x:x+w]
            if eye_crop.size == 0:
                continue
            features = extract_hog(eye_crop).reshape(1, -1)
            pred = clf.predict(features)[0]
            prob = clf.predict_proba(features)[0].max()
            frame_confidences.append(prob)

            label = "OPEN" if pred == 1 else "CLOSED"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)
            cv2.rectangle(degraded_frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(degraded_frame, f"{label} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if frame_confidences:
            confidence_history.append(np.mean(frame_confidences))
        else:
            confidence_history.append(0.0)
        confidence_history = confidence_history[-HISTORY_LEN:]
        rolling_conf = np.mean(confidence_history)

        cv2.putText(degraded_frame, f"CONDITION: {condition_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(degraded_frame, f"Detections found: {len(eyes)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        bar_color = (0, 255, 0) if rolling_conf > 0.7 else \
                    (0, 165, 255) if rolling_conf > 0.4 else (0, 0, 255)
        cv2.putText(degraded_frame, f"Rolling confidence: {rolling_conf:.2f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)
        bar_w = int(300 * rolling_conf)
        cv2.rectangle(degraded_frame, (10, 100), (310, 120), (60, 60, 60), -1)
        cv2.rectangle(degraded_frame, (10, 100), (10 + bar_w, 120), bar_color, -1)

        cv2.putText(degraded_frame, "Robustness Stress Test - Raks (Evaluation)",
                    (10, degraded_frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        cv2.putText(degraded_frame, "Press 1-5 to change condition, Q to quit",
                    (10, degraded_frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Robustness Stress Test", degraded_frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stress_test_demo()
