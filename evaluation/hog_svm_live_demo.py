"""
Advanced Driver Attention Monitor v5 - Outer Lip Contour Engine.
Switches to outer lip geometric tracking to capture soft/partial yawns 
and integrates frame-rate derivative tracking to catch rapid mouth openings.

Press F to toggle fullscreen, Q to quit.
"""
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

# Landmark mappings
LEFT_EYE_EAR = [362, 385, 386, 263, 374, 380]
RIGHT_EYE_EAR = [33, 160, 159, 133, 153, 144]

# Robust Outer Lip Trackers (Upper lip top to lower lip bottom)
# Vertical: 0 to 17, 11 to 16, 12 to 15 / Horizontal: 61 to 291
OUTER_LIP_MAR = [61, 0, 11, 12, 291, 15, 16, 17]

WINDOW_NAME = "Advanced Distraction & Yawn Monitor"

def get_pixel_2d(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def calculate_ear(landmarks, eye_indices, w, h):
    p1 = get_pixel_2d(landmarks[eye_indices[0]], w, h)
    p2 = get_pixel_2d(landmarks[eye_indices[1]], w, h)
    p3 = get_pixel_2d(landmarks[eye_indices[2]], w, h)
    p4 = get_pixel_2d(landmarks[eye_indices[3]], w, h)
    p5 = get_pixel_2d(landmarks[eye_indices[4]], w, h)
    p6 = get_pixel_2d(landmarks[eye_indices[5]], w, h)

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0: return 0.0
    return (v1 + v2) / (2.0 * horizontal)

def calculate_outer_mar(landmarks, lip_indices, w, h):
    """Computes stable Mouth Aspect Ratio using the highly visible outer lip limits."""
    m1 = get_pixel_2d(landmarks[lip_indices[0]], w, h) # Left corner
    m2 = get_pixel_2d(landmarks[lip_indices[1]], w, h)
    m3 = get_pixel_2d(landmarks[lip_indices[2]], w, h)
    m4 = get_pixel_2d(landmarks[lip_indices[3]], w, h)
    m5 = get_pixel_2d(landmarks[lip_indices[4]], w, h) # Right corner
    m6 = get_pixel_2d(landmarks[lip_indices[5]], w, h)
    m7 = get_pixel_2d(landmarks[lip_indices[6]], w, h)
    m8 = get_pixel_2d(landmarks[lip_indices[7]], w, h)

    v1 = np.linalg.norm(m2 - m8)
    v2 = np.linalg.norm(m3 - m7)
    v3 = np.linalg.norm(m4 - m6)
    horizontal = np.linalg.norm(m1 - m5)

    if horizontal == 0: return 0.0
    return (v1 + v2 + v3) / (3.0 * horizontal)

def run_attention_monitor():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False

    calibration_frames = []
    calibrated = False
    
    ear_threshold = 0.22
    mar_threshold = 0.35  # Optimized outer lip activation threshold floor
    
    # Track frame changes to calculate yawn acceleration vectors
    mar_history = deque(maxlen=5)

    print("Initializing environment calibration...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb)

        avg_ear, current_mar = 0.0, 0.0
        eye_label, mouth_label = "OPEN", "NORMAL"
        eye_color, mouth_color = (0, 255, 0), (0, 255, 0)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            avg_ear = (calculate_ear(landmarks, LEFT_EYE_EAR, w, h) + calculate_ear(landmarks, RIGHT_EYE_EAR, w, h)) / 2.0
            current_mar = calculate_outer_mar(landmarks, OUTER_LIP_MAR, w, h)
            mar_history.append(current_mar)

            if not calibrated:
                calibration_frames.append((avg_ear, current_mar))
                cv2.putText(frame, f"CALIBRATING SYSTEM: {len(calibration_frames)}/50", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if len(calibration_frames) >= 50:
                    arr = np.array(calibration_frames)
                    ear_threshold = np.mean(arr[:, 0]) * 0.70
                    
                    # Outer lips have a higher baseline resting value; scale accordingly
                    mar_threshold = np.mean(arr[:, 1]) * 1.55
                    calibrated = True
                    print(f"[CALIBRATION SUCCESS] EAR: {ear_threshold:.3f} | MAR: {mar_threshold:.3f}")

            else:
                # Calculate real-time frame changes to catch rapid openings
                mar_velocity = current_mar - mar_history[0] if len(mar_history) == 5 else 0.0

                # 1. EVALUATE MOUTH STATE (Triggers on pure size OR opening speed)
                if current_mar > mar_threshold or mar_velocity > 0.12:
                    mouth_label = "YAWNING"
                    mouth_color = (0, 0, 255)
                else:
                    mouth_label = "NORMAL"
                    mouth_color = (0, 255, 0)

                # 2. EVALUATE EYE STATE WITH SQUINT SUPPRESSION
                if mouth_label == "YAWNING":
                    # Lower the baseline significantly to prevent yawn squints from triggering false alarms
                    if avg_ear < (ear_threshold * 0.60):
                        eye_label = "CLOSED"
                        eye_color = (0, 0, 255)
                    else:
                        eye_label = "OPEN (SQUINT MODULATED)"
                        eye_color = (0, 255, 255)
                else:
                    if avg_ear < ear_threshold:
                        eye_label = "CLOSED"
                        eye_color = (0, 0, 255)
                    else:
                        eye_label = "OPEN"
                        eye_color = (0, 255, 0)

            # Draw outer lip boundary reference dots
            for idx in LEFT_EYE_EAR + RIGHT_EYE_EAR:
                pt = get_pixel_2d(landmarks[idx], w, h).astype(int)
                cv2.circle(frame, tuple(pt), 1, eye_color, -1)
            for idx in OUTER_LIP_MAR:
                pt = get_pixel_2d(landmarks[idx], w, h).astype(int)
                cv2.circle(frame, tuple(pt), 1, mouth_color, -1)

        # Render HUD elements
        cv2.putText(frame, f"EYE STATE   : {eye_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"MOUTH STATE : {mouth_label}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        
        cv2.putText(frame, f"EAR: {avg_ear:.3f}/{ear_threshold:.3f} | MAR: {current_mar:.3f}/{mar_threshold:.3f}", 
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

        if eye_label == "CLOSED" or mouth_label == "YAWNING":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 4)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attention_monitor()