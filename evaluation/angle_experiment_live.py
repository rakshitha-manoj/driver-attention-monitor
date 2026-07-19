"""
Live webcam version of the camera-angle sensitivity experiment.
Captures EAR readings interactively at approximate 0/15/25 degree
positions and fits a linear correction.
"""
import sys, os, time
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import compute_ear, LEFT_EYE, RIGHT_EYE
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

ANGLES = [0, 15, 25]
CAPTURE_SECONDS = 5

def capture_ear_for_angle(cap, angle_label):
    print(f"\n>>> Position yourself at approximately {angle_label} deg. "
          f"Capturing in 3 seconds...")
    time.sleep(3)
    print(f"Capturing {angle_label} deg for {CAPTURE_SECONDS}s - hold still, eyes open...")

    ears = []
    start = time.time()
    while time.time() - start < CAPTURE_SECONDS:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = frame.shape[:2]
            left = compute_ear(landmarks, LEFT_EYE, w, h)
            right = compute_ear(landmarks, RIGHT_EYE, w, h)
            ears.append((left + right) / 2.0)

        cv2.putText(frame, f"Capturing {angle_label} deg... {len(ears)} samples",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Angle Experiment", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"  {angle_label} deg: {len(ears)} samples, mean EAR = {np.mean(ears):.4f}")
    return ears

def run_experiment():
    cap = cv2.VideoCapture(0)
    results_by_angle = {}

    print("Live angle sensitivity experiment.")
    print("Keep eyes OPEN and look at the camera throughout each capture window.")

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
        print(f"Angle {angle} deg: offset from frontal baseline = {offset:+.4f}")

    coeffs = np.polyfit(angles_arr, offsets_arr, deg=1)
    slope, intercept = coeffs
    print(f"\nLinear correction: EAR_correction(angle) = {slope:.6f} * angle + {intercept:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(angles_arr, offsets_arr, color="steelblue", s=60, label="measured offset")
    fit_line = np.polyval(coeffs, angles_arr)
    ax.plot(angles_arr, fit_line, color="red", linestyle="--", label="linear fit")
    ax.set_xlabel("Camera angle (degrees)")
    ax.set_ylabel("EAR offset from frontal")
    ax.set_title("Live camera-angle sensitivity (Raks, self-captured)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("angle_sensitivity_live.png", dpi=150)
    print("Saved plot to angle_sensitivity_live.png")

    return slope, intercept, results_by_angle

def apply_angle_correction(ear, angle, slope, intercept):
    correction = slope * angle + intercept
    return ear - correction

if __name__ == "__main__":
    run_experiment()
