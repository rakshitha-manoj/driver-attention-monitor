"""
Live demo: HOG + SVM eye-state classifier running on webcam.
This is Raks's independent classical CV pipeline, no landmark geometry,
no dependency on Hafsa's or Sheethal's modules. Run this after
hog_svm_baseline.py has produced hog_svm_model.pkl.
"""
import cv2
import numpy as np
import joblib
from skimage.feature import hog

MODEL_PATH = "hog_svm_model.pkl"
IMG_SIZE = (64, 64)

# Haar cascade for eye detection (comes bundled with opencv-python)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def extract_hog(img_gray):
    img_gray = cv2.resize(img_gray, IMG_SIZE)
    return hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")

def run_live_demo():
    print("Loading trained HOG+SVM model...")
    clf = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    print("Live HOG+SVM eye classifier running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in eyes:
            eye_crop = gray[y:y+h, x:x+w]
            if eye_crop.size == 0:
                continue

            features = extract_hog(eye_crop).reshape(1, -1)
            pred = clf.predict(features)[0]
            prob = clf.predict_proba(features)[0].max()

            label = "OPEN" if pred == 1 else "CLOSED"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, "HOG+SVM Classical CV Baseline — Raks",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        cv2.imshow("HOG+SVM Live Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_demo()
