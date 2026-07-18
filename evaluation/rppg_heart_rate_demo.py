"""
Live remote photoplethysmography (rPPG) heart-rate estimation - Raks's
novel signal, camera-only, no contact sensor.

Detects subtle color changes in the forehead skin caused by blood flow,
extracts the pulse signal from the green channel, bandpass-filters it
to the physiological heart-rate range (42-240 bpm), and estimates BPM
via FFT peak detection. Draws the forehead ROI box and live waveform
directly on the video feed, plus the current BPM reading.

This is a genuinely different signal type from anything else in the
project: not eye state, not head pose, not gaze - a physiological
signal extracted purely from skin color, relevant to fatigue since
drowsiness is associated with measurable heart-rate variability drops.
"""
import sys, os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
import mediapipe as mp

BUFFER_SECONDS = 8
FPS_ASSUMED = 30
BUFFER_LEN = BUFFER_SECONDS * FPS_ASSUMED
MIN_BPM, MAX_BPM = 42, 180

# Forehead region landmark indices (a small patch between the eyebrows, up)
FOREHEAD_IDS = [10, 108, 151, 337]

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def get_forehead_roi(landmarks, w, h, frame):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in FOREHEAD_IDS])
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    # expand slightly upward to get more forehead skin, avoid eyebrows
    y0 = max(0, y0 - 25)
    y1 = max(y0 + 10, y1 - 5)
    x0, x1 = max(0, x0 - 10), min(w, x1 + 10)
    return x0, y0, x1, y1

def bandpass_filter(signal, fps, low_bpm=MIN_BPM, high_bpm=MAX_BPM):
    nyq = 0.5 * fps
    low = (low_bpm / 60.0) / nyq
    high = (high_bpm / 60.0) / nyq
    if low <= 0 or high >= 1:
        return signal
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)

def estimate_bpm(green_signal, fps):
    if len(green_signal) < fps * 3:
        return None
    detrended = green_signal - np.mean(green_signal)
    filtered = bandpass_filter(detrended, fps)

    fft_vals = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    bpm_freqs = freqs * 60.0

    valid = (bpm_freqs >= MIN_BPM) & (bpm_freqs <= MAX_BPM)
    if not np.any(valid):
        return None

    peak_idx = np.argmax(fft_vals[valid])
    peak_bpm = bpm_freqs[valid][peak_idx]
    return peak_bpm

def draw_waveform(frame, signal, x0=10, y0=200, w=300, h=80):
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 1)
    if len(signal) < 2:
        return
    sig = np.array(signal)
    sig = (sig - sig.min()) / (sig.max() - sig.min() + 1e-6)
    pts = []
    for i, val in enumerate(sig):
        px = x0 + int(i / len(sig) * w)
        py = y0 + h - int(val * h)
        pts.append((px, py))
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0, 255, 0), 1)

def run_rppg_demo():
    cap = cv2.VideoCapture(0)
    green_buffer = deque(maxlen=BUFFER_LEN)
    bpm_display = None

    print("Live rPPG heart-rate demo running.")
    print("Sit still, keep your forehead visible and well lit. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            x0, y0, x1, y1 = get_forehead_roi(landmarks, w, h, frame)

            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                mean_green = np.mean(roi[:, :, 1])  # BGR, index 1 = green
                green_buffer.append(mean_green)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(frame, "Forehead ROI", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if len(green_buffer) >= FPS_ASSUMED * 3:
                bpm = estimate_bpm(np.array(green_buffer), FPS_ASSUMED)
                if bpm is not None:
                    bpm_display = bpm

        if bpm_display is not None:
            cv2.putText(frame, f"HEART RATE: {bpm_display:.0f} BPM",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Estimating heart rate... hold still",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        buffer_pct = len(green_buffer) / BUFFER_LEN * 100
        cv2.putText(frame, f"Signal buffer: {buffer_pct:.0f}%",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        draw_waveform(frame, list(green_buffer)[-150:])
        cv2.putText(frame, "Pulse signal (raw)", (10, 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "rPPG Heart Rate (camera-only) - Raks (Evaluation, novel signal)",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        cv2.imshow("rPPG Heart Rate Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rppg_demo()
