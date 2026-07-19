"""
Hand-near-face / phone-use detector - manual distraction signal.
Uses MediaPipe Hands (a separate model from Face Mesh) to detect when
a hand is raised near the ear/face region, a classic phone-call
gesture. Covers "manual distraction" (hand off wheel, phone use), a
category neither Hafsa's (eye/mouth state) nor Sheethal's (head pose)
work touches - a genuinely new modality.
"""
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

LEFT_EAR_REGION = 234
RIGHT_EAR_REGION = 454
NEAR_FACE_THRESHOLD_RATIO = 0.35

WINDOW_NAME = "Hand-Near-Face Detector"

def get_point(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])

def hand_center(hand_landmarks, w, h):
    pts = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
    return pts.mean(axis=0)

def run_hand_detector():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False

    near_face_streak = 0
    print("Live hand-near-face / phone-use detector running. Press Q to quit, F for fullscreen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(rgb)
        hand_results = hands.process(rgb)

        hand_near_face = False
        face_width = None

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            left_ear = get_point(landmarks, LEFT_EAR_REGION, w, h)
            right_ear = get_point(landmarks, RIGHT_EAR_REGION, w, h)
            face_width = np.linalg.norm(right_ear - left_ear)

            cv2.circle(frame, tuple(left_ear.astype(int)), 4, (0, 200, 255), -1)
            cv2.circle(frame, tuple(right_ear.astype(int)), 4, (0, 200, 255), -1)

            if hand_results.multi_hand_landmarks and face_width:
                threshold_px = face_width * (1 + NEAR_FACE_THRESHOLD_RATIO)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hc = hand_center(hand_landmarks, w, h)
                    dist_left = np.linalg.norm(hc - left_ear)
                    dist_right = np.linalg.norm(hc - right_ear)
                    if min(dist_left, dist_right) < threshold_px:
                        hand_near_face = True
                        cv2.putText(frame, "HAND NEAR FACE", tuple(hc.astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        elif hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if hand_near_face:
            near_face_streak += 1
        else:
            near_face_streak = 0

        sustained_alert = near_face_streak > 30

        status_label = "PHONE / HAND NEAR FACE" if hand_near_face else "HANDS CLEAR"
        status_color = (0, 0, 255) if hand_near_face else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

        if sustained_alert:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(frame, "SUSTAINED MANUAL DISTRACTION",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, "Hand/Phone Detector - Raks (manual distraction) | F=fullscreen Q=quit",
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
    run_hand_detector()
