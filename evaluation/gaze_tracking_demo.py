"""
Live gaze/iris tracking demo v6 - Optimized Eye-Corner Sensitivity.
Fine-tuned vertical tracking deviation thresholds to prevent downward gaze sticking.

Look straight ahead at the screen for the first 1-2 seconds after launch.
Press F to toggle fullscreen, Q to quit.
"""
import cv2
import numpy as np
import mediapipe as mp

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_CORNERS = [263, 362]     # outer, inner
RIGHT_EYE_CORNERS = [133, 33]      # inner, outer

WINDOW_NAME = "Gaze Tracking"

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

def iris_center(landmarks, iris_ids, w, h):
    pts = np.array([get_point(landmarks, i, w, h) for i in iris_ids])
    return pts.mean(axis=0)

def estimate_horizontal_ratio(iris_pt, corner1, corner2):
    axis_vec = corner2 - corner1
    iris_vec = iris_pt - corner1
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        return 0.5
    projection = np.dot(iris_vec, axis_vec) / (axis_len ** 2)
    return np.clip(projection, 0.0, 1.0)

def calculate_vertical_deviation(iris_pt, corner1, corner2):
    """
    Calculates how far the iris sits vertically above or below the 
    line connecting the horizontal eye corners, normalized by eye width.
    """
    eye_corners_y = (corner1[1] + corner2[1]) / 2.0
    eye_width = np.linalg.norm(corner2 - corner1)
    if eye_width == 0:
        return 0.0
    
    # MediaPipe Y increases downwards. 
    # (eye_corners_y - iris_pt[1]) is POSITIVE when iris is higher up on screen (looking UP)
    return (eye_corners_y - iris_pt[1]) / eye_width

def classify_horizontal(ratio):
    if ratio < 0.35:
        return "RIGHT"
    elif ratio > 0.65:
        return "LEFT"
    return "CENTER"

def classify_vertical_calibrated(deviation, baseline):
    relative_diff = deviation - baseline
    
    # OPTIMIZED SENSITIVITY: 
    # Positive relative_diff = iris moved UP relative to baseline
    # Negative relative_diff = iris moved DOWN relative to baseline
    if relative_diff > 0.04:
        return "UP"
    elif relative_diff < -0.03:  # Increased sensitivity to catch narrow downward gaze shifts
        return "DOWN"
    return "CENTER"

def run_gaze_tracking():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False

    print("Live gaze tracking (Optimized Corner Thresholds v6) running.")
    print("Keep eyes centered on screen initially to calibrate baseline...")

    off_road_streak = 0
    calibration_frames = []
    v_baseline = 0.0
    calibrated = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1) # Mirror image for natural user interaction display
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_iris_pt = iris_center(landmarks, LEFT_IRIS, w, h)
            right_iris_pt = iris_center(landmarks, RIGHT_IRIS, w, h)

            left_h1 = get_point(landmarks, LEFT_EYE_CORNERS[0], w, h)
            left_h2 = get_point(landmarks, LEFT_EYE_CORNERS[1], w, h)
            right_h1 = get_point(landmarks, RIGHT_EYE_CORNERS[0], w, h)
            right_h2 = get_point(landmarks, RIGHT_EYE_CORNERS[1], w, h)

            # 1. HORIZONTAL TRACKING
            h_ratio = (estimate_horizontal_ratio(left_iris_pt, left_h1, left_h2) +
                       estimate_horizontal_ratio(right_iris_pt, right_h1, right_h2)) / 2.0

            # 2. VERTICAL TRACKING
            left_v_dev = calculate_vertical_deviation(left_iris_pt, left_h1, left_h2)
            right_v_dev = calculate_vertical_deviation(right_iris_pt, right_h1, right_h2)
            v_deviation = (left_v_dev + right_v_dev) / 2.0

            # 3. CALIBRATION RUN
            if not calibrated:
                calibration_frames.append(v_deviation)
                cv2.putText(frame, f"CALIBRATING BASELINE: {len(calibration_frames)}/30", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if len(calibration_frames) >= 30:
                    v_baseline = np.mean(calibration_frames)
                    calibrated = True
                    print(f"Calibration successful. Vertical baseline set to: {v_baseline:.4f}")
                h_label, v_label = "CENTER", "CENTER"
            else:
                h_label = classify_horizontal(h_ratio)
                v_label = classify_vertical_calibrated(v_deviation, v_baseline)

            if h_label == "CENTER" and v_label == "CENTER":
                gaze_label = "ON ROAD"
            else:
                gaze_label = f"{v_label if v_label != 'CENTER' else ''} {h_label if h_label != 'CENTER' else ''}".strip()
                
            gaze_color = (0, 255, 0) if gaze_label == "ON ROAD" else (0, 165, 255)

            # Draw visual tracking components
            for pt in [left_iris_pt, right_iris_pt]:
                cv2.circle(frame, tuple(pt.astype(int)), 4, (255, 0, 255), -1)
            for pt in [left_h1, left_h2, right_h1, right_h2]:
                cv2.circle(frame, tuple(pt.astype(int)), 3, (0, 255, 255), -1)

            # Arrow Projection Engine
            face_center = get_point(landmarks, 1, w, h)
            dx = (h_ratio - 0.5) * -160
            dy = (v_deviation - v_baseline) * -1200  
            arrow_end = (int(face_center[0] + dx), int(face_center[1] + dy))
            cv2.arrowedLine(frame, tuple(face_center.astype(int)), arrow_end,
                            gaze_color, 3, tipLength=0.3)

            if calibrated:
                cv2.putText(frame, f"GAZE: {gaze_label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)
                cv2.putText(frame, f"h_ratio={h_ratio:.2f}  v_dev={v_deviation:.2f} (base={v_baseline:.2f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if gaze_label != "ON ROAD" and calibrated:
                off_road_streak += 1
            else:
                off_road_streak = 0
            distraction_alert = off_road_streak > 45

            if distraction_alert:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
                cv2.putText(frame, "SUSTAINED GAZE DEVIATION - DISTRACTION",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Gaze Tracking v6 - Optimized Sensitivity | F=fullscreen Q=quit",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

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
    run_gaze_tracking()