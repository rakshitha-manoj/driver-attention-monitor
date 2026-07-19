"""
Live rPPG heart-rate estimation demo v2.

Fix from v1: the BPM estimate was recomputed and displayed every single
frame off a noisy, still-filling buffer, so a single bad frame (motion,
lighting flicker) could swing the displayed number by dozens of BPM.
v2 fixes this with:
  1. Only recompute the FFT estimate every 15 frames (~0.5s), not every frame
  2. Detrend with a linear fit instead of just subtracting the mean
  3. Apply a Hann window before the FFT (reduces spectral leakage)
  4. Require a minimum peak prominence, reject low-confidence estimates
  5. Smooth the DISPLAYED bpm with a moving average of the last 5 estimates
  6. Use a longer buffer (10s) for a more stable frequency resolution
"""
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from collections import deque

BUFFER_SECONDS = 10
FPS_ASSUMED = 30
BUFFER_LEN = BUFFER_SECONDS * FPS_ASSUMED
MIN_BPM, MAX_BPM = 42, 180
RECOMPUTE_EVERY_N_FRAMES = 15
BPM_SMOOTHING_WINDOW = 5
MIN_PEAK_PROMINENCE_RATIO = 1.5  # peak must be this much stronger than median

FOREHEAD_IDS = [10, 108, 151, 337]

import mediapipe as mp
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def get_forehead_roi(landmarks, w, h):
    pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in FOREHEAD_IDS])
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
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
    """Returns (bpm, confidence) or (None, 0) if signal quality is too low."""
    if len(green_signal) < fps * 4:
        return None, 0

    detrended = detrend(green_signal, type="linear")
    filtered = bandpass_filter(detrended, fps)

    window = np.hanning(len(filtered))
    windowed = filtered * window

    fft_vals = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0/fps)
    bpm_freqs = freqs * 60.0

    valid = (bpm_freqs >= MIN_BPM) & (bpm_freqs <= MAX_BPM)
    if not np.any(valid):
        return None, 0

    valid_power = fft_vals[valid]
    valid_bpm = bpm_freqs[valid]

    peak_idx = np.argmax(valid_power)
    peak_power = valid_power[peak_idx]
    median_power = np.median(valid_power)

    # signal-quality check: reject if the peak isn't clearly above the noise floor
    if median_power == 0 or peak_power / median_power < MIN_PEAK_PROMINENCE_RATIO:
        return None, 0

    confidence = min(1.0, (peak_power / median_power) / 4.0)
    return valid_bpm[peak_idx], confidence

def draw_waveform(frame, signal, x0=10, y0=220, w=300, h=80):
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (30, 30, 30), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (100, 100, 100), 1)
    if len(signal) < 2:
        return
    sig = np.array(signal)
    if sig.max() - sig.min() < 1e-6:
        return
    sig = (sig - sig.min()) / (sig.max() - sig.min())
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
    bpm_estimates = deque(maxlen=BPM_SMOOTHING_WINDOW)
    frame_counter = 0
    smoothed_bpm = None
    last_confidence = 0

    print("Live rPPG heart-rate demo v2 running.")
    print("Sit still, keep your forehead visible and well lit.")
    print("Estimate stabilizes after ~10 seconds of steady signal. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            x0, y0, x1, y1 = get_forehead_roi(landmarks, w, h)

            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                mean_green = np.mean(roi[:, :, 1])
                green_buffer.append(mean_green)

                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(frame, "Forehead ROI", (x0, y0 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            frame_counter += 1
            if len(green_buffer) >= FPS_ASSUMED * 4 and frame_counter % RECOMPUTE_EVERY_N_FRAMES == 0:
                bpm, confidence = estimate_bpm(np.array(green_buffer), FPS_ASSUMED)
                if bpm is not None:
                    bpm_estimates.append(bpm)
                    last_confidence = confidence
                    if len(bpm_estimates) >= 2:
                        smoothed_bpm = np.median(bpm_estimates)  # median is more robust than mean here
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if smoothed_bpm is not None:
            quality_txt = "good" if last_confidence > 0.6 else "low" if last_confidence > 0.3 else "poor"
            quality_color = (0, 255, 0) if last_confidence > 0.6 else \
                             (0, 165, 255) if last_confidence > 0.3 else (0, 0, 255)
            cv2.putText(frame, f"HEART RATE: {smoothed_bpm:.0f} BPM",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Signal quality: {quality_txt}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 1)
        else:
            buffer_pct = min(100, len(green_buffer) / (FPS_ASSUMED * 4) * 100)
            cv2.putText(frame, f"Estimating... buffer {buffer_pct:.0f}% (hold still)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        draw_waveform(frame, list(green_buffer)[-150:])
        cv2.putText(frame, "Pulse signal (raw)", (10, 215),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "rPPG Heart Rate v2 (camera-only) - Raks",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        cv2.imshow("rPPG Heart Rate Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_rppg_demo()
