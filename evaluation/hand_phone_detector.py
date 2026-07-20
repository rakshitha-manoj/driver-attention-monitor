"""
Advanced Distraction Monitor v15 - YOLO Deep Learning Fusion Edition.
Uses YOLOv8 to explicitly detect the presence of a phone object before checking
spatial regions relative to the face mesh, eliminating skeleton false positives.

Press F to toggle fullscreen, Q to quit.
"""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Initialize Deep Learning Object Detector
# Natively trained on COCO dataset; Class ID 67 corresponds exactly to 'cell phone'
yolo_model = YOLO('yolov8n.pt') 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.6, min_tracking_confidence=0.6
)

# Core Facemesh Indices
NOSE = 1
CHIN = 152
LEFT_EAR = 234
RIGHT_EAR = 454
FOREHEAD = 10

WINDOW_NAME = "YOLO Deep Learning Fusion Monitor"

def get_pixel_2d(lm, w, h):
    return np.array([lm.x * w, lm.y * h])

def run_fusion_monitor():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = False

    print("Loading YOLOv8 network and starting stream...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1) # Natural mirroring
        
        # 1. RUN YOLO OBJECT DETECTION INFERENCE
        # verbose=False keeps the console clean during real-time loops
        yolo_results = yolo_model(frame, verbose=False)[0]
        
        phone_boxes = []
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filter specifically for 'cell phone' with a reliable confidence floor
            if class_id == 67 and confidence > 0.45:
                # Extract pixel bounding coordinates [x_min, y_min, x_max, y_max]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                phone_boxes.append(xyxy)
                
                # Draw visual target locator box over the detected device
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 2)
                cv2.putText(frame, f"Phone {confidence:.2f}", (xyxy[0], xyxy[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 2. RUN FACEMESH BOUNDARY EXTRACTION
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(rgb)

        phone_call = False
        low_texting = False

        if face_results.multi_face_landmarks and len(phone_boxes) > 0:
            landmarks = face_results.multi_face_landmarks[0].landmark
            
            nose_2d = get_pixel_2d(landmarks[NOSE], w, h)
            chin_2d = get_pixel_2d(landmarks[CHIN], w, h)
            left_ear_2d = get_pixel_2d(landmarks[LEFT_EAR], w, h)
            right_ear_2d = get_pixel_2d(landmarks[RIGHT_EAR], w, h)
            forehead_2d = get_pixel_2d(landmarks[FOREHEAD], w, h)

            face_width = np.linalg.norm(right_ear_2d - left_ear_2d)
            face_height = np.linalg.norm(chin_2d - forehead_2d)

            # Draw localization vector guides
            cv2.line(frame, tuple(left_ear_2d.astype(int)), tuple(right_ear_2d.astype(int)), (255, 255, 0), 1)
            cv2.line(frame, tuple(chin_2d.astype(int)), tuple(forehead_2d.astype(int)), (255, 255, 0), 1)

            # 3. SPATIAL FUSION LAYER
            for box in phone_boxes:
                # Calculate the center coordinate of the phone object bounding box
                phone_center = np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])

                dist_to_left_ear = np.linalg.norm(phone_center - left_ear_2d)
                dist_to_right_ear = np.linalg.norm(phone_center - right_ear_2d)

                # Sector A: Phone is located in close proximity to either ear channel
                if min(dist_to_left_ear, dist_to_right_ear) < (face_width * 0.95) and phone_center[1] < chin_2d[1]:
                    phone_call = True
                    break
                
                # Sector B: Phone is detected operating below the nose line
                elif phone_center[1] > (nose_2d[1] - face_height * 0.2):
                    low_texting = True

        # 4. MONITOR UI LAYER GENERATION
        if phone_call or low_texting:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            msg = "CRITICAL: PHONE CALL DETECTED" if phone_call else "CRITICAL: LOW TEXTING DETECTED"
            cv2.putText(frame, msg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        status_txt = "DISTRACTION DETECTED" if (phone_call or low_texting) else "SYSTEM CLEAR"
        status_color = (0, 0, 255) if (phone_call or low_texting) else (0, 255, 0)
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