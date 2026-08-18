"""
Live webcam version of the camera-angle sensitivity experiment.
Captures EAR readings interactively at approximate 0/15/25 degree
positions and fits a linear correction with interactive frame-rate counters.
"""
import sys, os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

# Fallback index mapping if the external perception module isn't loaded
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

ANGLES = [0, 15, 25]
CAPTURE_SECONDS = 5

def local_compute_ear(landmarks, eye_indices, w, h):
    """Fallback execution module to compute EAR directly if local import path shifts."""
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in [eye_indices[12], eye_indices[4], eye_indices[14], eye_indices[2], eye_indices[0], eye_indices[8]]]
    p2_p6 = np.linalg.norm(pts[0] - pts[1])
    p3_p5 = np.linalg.norm(pts[2] - pts[3])
    p1_p4 = np.linalg.norm(pts[4] - pts[5])
    if p1_p4 == 0: return 0.0
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def capture_ear_for_angle(cap, angle_label):
    # ASYNCHRONOUS ENGINE FIX: Use non-blocking loops for the 3-second setup countdown
    countdown_start = time.time()
    while time.time() - countdown_start < 3.0:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        remaining = 3 - int(time.time() - countdown_start)
        cv2.putText(frame, f"POSITION FOR {angle_label} DEG IN: {remaining}s", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow("Angle Experiment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Capturing {angle_label} deg - hold still, eyes open...")
    ears = []
    start = time.time()
    
    while time.time() - start < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret: continue
        
        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left = local_compute_ear(landmarks, LEFT_EYE, w, h)
            right = local_compute_ear(landmarks, RIGHT_EYE, w, h)
            ears.append((left + right) / 2.0)

        cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 4)
        cv2.putText(frame, f"RECORDING {angle_label} DEG: {len(ears)} frames collected",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Angle Experiment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"--> Done: mean EAR = {np.mean(ears):.4f}")
    return ears

def run_experiment():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Angle Experiment", cv2.WINDOW_NORMAL)
    results_by_angle = {}

    print("Starting automated angle sensitivity routine...")

    for angle in ANGLES:
        ears = capture_ear_for_angle(cap, angle)
        results_by_angle[angle] = ears

    cap.release()
    cv2.destroyAllWindows()

    baseline = np.mean(results_by_angle[0])
    angles_arr, offsets_arr = [], []
    for angle, ears in results_by_angle.items():
        offset = np.mean(ears) - baseline
        angles_arr.append(angle)
        offsets_arr.append(offset)

    coeffs = np.polyfit(angles_arr, offsets_arr, deg=1)
    slope, intercept = coeffs
    print(f"\nLinear correction: EAR_correction(angle) = {slope:.6f} * angle + {intercept:.6f}")

    # Plot and save diagnostic visualization map
    plt.figure(figsize=(6, 4))
    plt.scatter(angles_arr, offsets_arr, color="darkmagenta", s=70, label="Measured Offset")
    plt.plot(angles_arr, np.polyval(coeffs, angles_arr), color="cyan", linestyle="--", label="Linear Fit")
    plt.xlabel("Camera angle (degrees)")
    plt.ylabel("EAR offset from frontal")
    plt.title("Camera-Angle Distortion Calibration Map")
    plt.legend()
    plt.tight_layout()
    plt.savefig("angle_sensitivity_live.png", dpi=150)
    print("Saved fit results plot graph to angle_sensitivity_live.png")

if __name__ == "__main__":
    run_experiment()