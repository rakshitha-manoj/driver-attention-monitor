"""
hand_detector.py — Evaluation layer (Raks)
Detects objects held in the driver's hand using MediaPipe Hands.

Key improvements:
  - Orientation-Invariant Grip Detection: Uses 3D/2D Euclidean distance
    from fingertips to palm/wrist relative to palm length. Works in
    ANY hand orientation (sideways next to ear, upside down, angled).
  - Proximity Check: Checks if hand is raised near the driver's head/face
    region (upper 65 % of frame or close to face center).
  - Reduced sustain duration to 1.0s for responsive object alert.
"""

import math
import cv2
import numpy as np
import mediapipe as mp


class HandObjectDetector:
    """
    Detects hands holding objects near the driver's face/ear area.
    Requires BOTH grip pose AND raised position to flag an object.
    """

    def __init__(self, sustain_seconds=0.8, detect_interval=2,
                 upper_frame_fraction=0.60, clear_seconds=0.5,
                 yolo_conf_threshold=0.25):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles  = mp.solutions.drawing_styles

        self.sustain_seconds      = sustain_seconds
        self.detect_interval      = detect_interval
        self.upper_frame_fraction = upper_frame_fraction
        self.clear_seconds        = clear_seconds
        self.yolo_conf_threshold  = yolo_conf_threshold

        self._grip_near_face_since = None
        self._under_since          = None
        self.object_flag   = False
        self.hand_detected = False
        self.hand_count    = 0
        self._frame_counter = 0
        self._cached_results = None

        # Initialize YOLOv8 for object detection
        from ultralytics import YOLO
        self.yolo = YOLO('yolov8n.pt')
        self.distracting_classes = {39, 41, 63, 67, 73}  # bottle, cup, laptop, cell phone, book

    # ─── Orientation-Invariant Grip Detection ───────────────────────

    @staticmethod
    def is_gripping(hand_landmarks):
        """
        Check if the hand is curled/gripping an object using rotation-invariant
        3D landmark Euclidean distances.
        """
        lm = hand_landmarks.landmark

        # Palm scale (Wrist to Middle MCP distance)
        dx = lm[9].x - lm[0].x
        dy = lm[9].y - lm[0].y
        dz = lm[9].z - lm[0].z
        palm_scale = math.sqrt(dx*dx + dy*dy + dz*dz)

        if palm_scale < 1e-4:
            return False

        finger_tips = [8, 12, 16, 20]
        curled = 0

        for tip_id in finger_tips:
            tx = lm[tip_id].x - lm[0].x
            ty = lm[tip_id].y - lm[0].y
            tz = lm[tip_id].z - lm[0].z
            tip_dist = math.sqrt(tx*tx + ty*ty + tz*tz)
            ratio = tip_dist / palm_scale

            if ratio < 1.55:
                curled += 1

        dx_thumb = lm[4].x - lm[5].x
        dy_thumb = lm[4].y - lm[5].y
        dz_thumb = lm[4].z - lm[5].z
        thumb_dist = math.sqrt(dx_thumb*dx_thumb + dy_thumb*dy_thumb + dz_thumb*dz_thumb) / palm_scale

        thumb_curled = (thumb_dist < 1.25)

        return (curled >= 3) or (curled >= 2 and thumb_curled)

    # ─── Main update ─────────────────────────────────────────────

    def update(self, frame_rgb, frame_bgr, now, face_center=None):
        """
        Process hand detection.
        Returns (object_flag, hand_detected, n_hands).
        Draws hand landmarks on frame_bgr in-place.
        """
        self._frame_counter += 1
        h, w = frame_bgr.shape[:2]

        self._cached_results = self.hands.process(frame_rgb)

        results = self._cached_results
        grip_near_face = False
        yolo_confirmed = False
        n_hands = 0

        if results and results.multi_hand_landmarks:
            n_hands = len(results.multi_hand_landmarks)

            # Run YOLOv8 on the frame to detect target distracting objects
            detected_objects = []
            try:
                yolo_results = self.yolo(frame_rgb, verbose=False)
                if yolo_results and len(yolo_results) > 0:
                    for box in yolo_results[0].boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls in self.distracting_classes and conf >= self.yolo_conf_threshold:
                            xyxy_val = box.xyxy[0]
                            if hasattr(xyxy_val, 'tolist'):
                                x1, y1, x2, y2 = xyxy_val.tolist()
                            else:
                                x1, y1, x2, y2 = xyxy_val
                            detected_objects.append({
                                "cls": cls,
                                "conf": conf,
                                "box": (x1, y1, x2, y2),
                                "active": False
                            })
            except Exception:
                pass

            for hand_lm in results.multi_hand_landmarks:
                wrist_x = hand_lm.landmark[0].x
                wrist_y = hand_lm.landmark[0].y
                raised = (wrist_y < self.upper_frame_fraction)

                near_face = False
                if face_center is not None:
                    fx, fy = face_center
                    hand_px_x = wrist_x * w
                    hand_px_y = wrist_y * h
                    dist_to_face = math.sqrt((hand_px_x - fx)**2 + (hand_px_y - fy)**2)
                    diag = math.sqrt(w*w + h*h)
                    near_face = (dist_to_face < 0.45 * diag)
                    should_flag = near_face
                else:
                    should_flag = raised

                overlaps_object = False
                gripping = False

                if should_flag:
                    self.mp_drawing.draw_landmarks(
                        frame_bgr, hand_lm,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

                    gripping = self.is_gripping(hand_lm)

                    if detected_objects:
                        for lm_pt in hand_lm.landmark:
                            lx = lm_pt.x * w
                            ly = lm_pt.y * h
                            for obj in detected_objects:
                                x1, y1, x2, y2 = obj["box"]
                                if (x1 - 25 <= lx <= x2 + 25) and (y1 - 25 <= ly <= y2 + 25):
                                    overlaps_object = True
                                    obj["active"] = True
                                    break
                            if overlaps_object:
                                break

                    if overlaps_object or gripping:
                        grip_near_face = True
                        if overlaps_object:
                            yolo_confirmed = True

                        wrist_px = (int(wrist_x * w), int(wrist_y * h))
                        label = "OBJECT IN HAND" if overlaps_object else "PHONE / HAND NEAR EAR"
                        cv2.putText(frame_bgr, label, wrist_px,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

            # Draw only active/overlapping objects to avoid focusing on background objects
            for obj in detected_objects:
                if obj["active"]:
                    x1, y1, x2, y2 = obj["box"]
                    cls = obj["cls"]
                    conf = obj["conf"]
                    cv2.rectangle(frame_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                    name = self.yolo.names[cls]
                    cv2.putText(frame_bgr, f"{name} {conf:.2f}", (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Debounce logic to prevent flickering
        required_sustain = 0.4 if yolo_confirmed else self.sustain_seconds
        if grip_near_face:
            self._under_since = None
            if self._grip_near_face_since is None:
                self._grip_near_face_since = now
            if (now - self._grip_near_face_since) >= required_sustain:
                self.object_flag = True
        else:
            self._grip_near_face_since = None
            if self.object_flag:
                if self._under_since is None:
                    self._under_since = now
                elif (now - self._under_since) >= self.clear_seconds:
                    self.object_flag = False
                    self._under_since = None
            else:
                self.object_flag = False

        self.hand_detected = (n_hands > 0)
        self.hand_count    = n_hands
        return self.object_flag, self.hand_detected, n_hands

