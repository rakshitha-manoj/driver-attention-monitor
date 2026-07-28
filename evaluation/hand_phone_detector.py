"""
Advanced Distraction Monitor v22 - Strict Verified Fusion Engine.
Fixes:
- Dark Flask / Unrecognized Container Drinking (Triggers via Hand-to-Mouth Landmark Proximity)
- Bottle near ear/chest triggering Phone Call (Strictly requires YOLO 'Phone' class)
- Low Phone Use vs Phone Call distinction (Split cleanly at Chin Y-boundary)
- Flat Phone on Ear Detection (Lowers phone threshold with hand-intersection requirement)

Press F to toggle fullscreen, Q to quit.
"""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

yolo_model = YOLO('yolov8n.pt') 

DISTRACTION_CLASSES = {
    67: "Phone",
    39: "Bottle",
    41: "Cup"
}

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)
hands = mp_hands.Hands(
    max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

NOSE = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
MOUTH_CENTER = 13

WINDOW_NAME = "Strict Verified Distraction Monitor"

def get_pixel_2d(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def get_closest_point_on_box(box, target_pt):
    x1, y1, x2, y2 = box
    closest_x = np.clip(target_pt[0], x1, x2)
    closest_y = np.clip(target_pt[1], y1, y2)
    return np.array([closest_x, closest_y])

def run_fusion_monitor():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False

    decision_history = deque(maxlen=8)

    print("Starting v22 Strict Engine...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1) # Natural mirroring
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. RUN MEDIAPIPE TRACKERS
        hand_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        hand_boxes = []
        hand_landmarks_2d = []

        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                pts = [get_pixel_2d(lm, w, h) for lm in hand_lm.landmark]
                pts_arr = np.array(pts)
                
                hx1, hy1 = np.min(pts_arr, axis=0).astype(int)
                hx2, hy2 = np.max(pts_arr, axis=0).astype(int)
                hand_boxes.append([hx1, hy1, hx2, hy2])
                
                # Store Wrist (0), Index MCP (5), Ring MCP (13) for tight gesture mapping
                hand_landmarks_2d.extend([pts[0], pts[5], pts[13]])

        # 2. RUN YOLO INFERENCE WITH HAND INTERSECTION
        yolo_results = yolo_model(frame, verbose=False)[0]
        valid_objects = []
        
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id in DISTRACTION_CLASSES:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label_name = DISTRACTION_CLASSES[class_id]

                # Hand intersection check
                is_held_by_hand = False
                for hbox in hand_boxes:
                    if not (xyxy[2] < hbox[0] or xyxy[0] > hbox[2] or xyxy[3] < hbox[1] or xyxy[1] > hbox[3]):
                        is_held_by_hand = True
                        break

                # Low threshold (0.18) for phones when held in hand (catches flat ear holds & sideways)
                if label_name == "Phone":
                    if confidence < 0.18 or not is_held_by_hand:
                        continue
                elif label_name in ["Bottle", "Cup"]:
                    if confidence < 0.20 or not is_held_by_hand:
                        continue

                valid_objects.append({
                    'class': label_name,
                    'box': xyxy,
                    'confidence': confidence
                })

        # 3. SPATIAL EVALUATION LOGIC
        raw_frame_alert = "CLEAR"

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            nose_2d = get_pixel_2d(landmarks[NOSE], w, h)
            chin_2d = get_pixel_2d(landmarks[CHIN], w, h)
            left_ear_2d = get_pixel_2d(landmarks[LEFT_EAR], w, h)
            right_ear_2d = get_pixel_2d(landmarks[RIGHT_EAR], w, h)
            mouth_2d = get_pixel_2d(landmarks[MOUTH_CENTER], w, h)

            face_width = np.linalg.norm(right_ear_2d - left_ear_2d)
            face_height = np.linalg.norm(chin_2d - nose_2d)

            # A. CHECK HAND-TO-MOUTH GESTURE (Guarantees Drinking even if Flask is not detected)
            hand_touching_mouth = False
            for hand_pt in hand_landmarks_2d:
                dist_to_mouth = np.linalg.norm(hand_pt - mouth_2d)
                if dist_to_mouth < (face_height * 1.25): # Tight boundary at lips
                    hand_touching_mouth = True
                    break

            # B. EVALUATE VERIFIED YOLO OBJECTS
            for obj in valid_objects:
                box = obj['box']
                obj_type = obj['class']

                near_ear_left = get_closest_point_on_box(box, left_ear_2d)
                near_ear_right = get_closest_point_on_box(box, right_ear_2d)
                near_mouth = get_closest_point_on_box(box, mouth_2d)

                dist_left_ear = np.linalg.norm(near_ear_left - left_ear_2d)
                dist_right_ear = np.linalg.norm(near_ear_right - right_ear_2d)
                dist_mouth = np.linalg.norm(near_mouth - mouth_2d)

                box_center_y = (box[1] + box[3]) / 2.0

                if obj_type == "Phone":
                    # PHONE CALL: Phone box is near left or right ear AND center is ABOVE chin level
                    if min(dist_left_ear, dist_right_ear) < (face_width * 0.85) and box_center_y < chin_2d[1]:
                        raw_frame_alert = "PHONE CALL"
                        break
                    # TEXTING / PHONE USE: Phone box center is BELOW chin level or held out front
                    else:
                        raw_frame_alert = "TEXTING / PHONE USE"
                        break

                elif obj_type in ["Bottle", "Cup"]:
                    if dist_mouth < (face_height * 2.0) or hand_touching_mouth:
                        raw_frame_alert = "DRINKING / EATING"
                        break

            # C. DRINKING FALLBACK (When Dark Flask is undetected by YOLO)
            # Triggers if a hand touches the mouth area and is NOT holding a verified phone
            if raw_frame_alert == "CLEAR" and hand_touching_mouth:
                raw_frame_alert = "DRINKING / EATING"

        # Render Active Object Boxes
        for obj in valid_objects:
            xyxy = obj['box']
            label_name = obj['class']
            box_color = (0, 255, 255) if label_name == "Phone" else (255, 165, 0)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)
            cv2.putText(frame, f"{label_name} {obj['confidence']:.2f}", (xyxy[0], xyxy[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

        # 4. TEMPORAL DECISION SMOOTHING
        decision_history.append(raw_frame_alert)
        smoothed_decision = max(set(decision_history), key=decision_history.count) if len(decision_history) > 0 else "CLEAR"

        if smoothed_decision != "CLEAR":
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(frame, f"CRITICAL: {smoothed_decision}", (10, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status_txt = "DISTRACTION DETECTED" if smoothed_decision != "CLEAR" else "SYSTEM CLEAR"
        status_color = (0, 0, 255) if smoothed_decision != "CLEAR" else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

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
    run_fusion_monitor()