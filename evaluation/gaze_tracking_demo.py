"""
Live gaze/iris tracking demo - Raks's distinct visual detection.

Neither Hafsa's eye-open/closed detection nor Sheethal's head-pose
tracking looks at WHERE the eyes are actually pointing. A driver's head
can face forward while their eyes are on a phone. This tracks the iris
position live (MediaPipe's refine_landmarks gives iris points that
nobody else's module is using) and estimates gaze direction, drawing
it visually on the face like a proper detection overlay, not a
score/dashboard.
"""
import sys, os
import cv2
import numpy as np
import mediapipe as mp

# Iris landmark indices (only available when refine_landmarks=True)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Eye corner landmarks, used to compute where the iris sits within the eye
LEFT_EYE_CORNERS = [263, 362]   # outer, inner
RIGHT_EYE_CORNERS = [133, 33]    # inner, outer

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

def estimate_gaze_ratio(iris_pt, corner1, corner2):
    """
    Returns a 0-1 value: where the iris sits between the two eye corners.
    ~0.5 = centered (looking straight), toward 0 or 1 = looking to a side.
    """
    eye_vec = corner2 - corner1
    iris_vec = iris_pt - corner1
    eye_len = np.linalg.norm(eye_vec)
    if eye_len == 0:
        return 0.5
    projection = np.dot(iris_vec, eye_vec) / (eye_len ** 2)
    return np.clip(projection, 0.0, 1.0)

def classify_gaze(ratio):
    if ratio < 0.35:
        return "LOOKING RIGHT", (0, 165, 255)
    elif ratio > 0.65:
        return "LOOKING LEFT", (0, 165, 255)
    else:
        return "ON ROAD", (0, 255, 0)

def run_gaze_tracking():
    cap = cv2.VideoCapture(0)
    print("Live gaze/iris tracking running. Press Q to quit.")

    off_road_streak = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ── Left eye ──
            left_iris_pt = iris_center(landmarks, LEFT_IRIS, w, h)
            left_c1 = get_point(landmarks, LEFT_EYE_CORNERS[0], w, h)
            left_c2 = get_point(landmarks, LEFT_EYE_CORNERS[1], w, h)
            left_ratio = estimate_gaze_ratio(left_iris_pt, left_c1, left_c2)

            # ── Right eye ──
            right_iris_pt = iris_center(landmarks, RIGHT_IRIS, w, h)
            right_c1 = get_point(landmarks, RIGHT_EYE_CORNERS[0], w, h)
            right_c2 = get_point(landmarks, RIGHT_EYE_CORNERS[1], w, h)
            right_ratio = estimate_gaze_ratio(right_iris_pt, right_c1, right_c2)

            avg_ratio = (left_ratio + right_ratio) / 2.0
            gaze_label, gaze_color = classify_gaze(avg_ratio)

            # ── Draw iris points ──
            cv2.circle(frame, tuple(left_iris_pt.astype(int)), 4, (255, 0, 255), -1)
            cv2.circle(frame, tuple(right_iris_pt.astype(int)), 4, (255, 0, 255), -1)

            # ── Draw eye corner reference points ──
            for pt in [left_c1, left_c2, right_c1, right_c2]:
                cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 255), -1)

            # ── Gaze direction arrow from face center ──
            face_center = get_point(landmarks, 1, w, h)  # nose tip as anchor
            direction = (avg_ratio - 0.5) * 2  # -1 (right) to +1 (left)
            arrow_end = (int(face_center[0] - direction * 80), int(face_center[1]))
            cv2.arrowedLine(frame, tuple(face_center.astype(int)), arrow_end,
                             gaze_color, 3, tipLength=0.3)

            # ── Sustained distraction tracking ──
            if gaze_label != "ON ROAD":
                off_road_streak += 1
            else:
                off_road_streak = 0

            distraction_alert = off_road_streak > 45  # ~1.5s at 30fps

            cv2.putText(frame, f"GAZE: {gaze_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, gaze_color, 2)

            if distraction_alert:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
                cv2.putText(frame, "SUSTAINED GAZE DEVIATION - DISTRACTION",
                            (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Gaze / Iris Tracking - Raks (Evaluation, new signal)",
                    (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gaze_tracking()
