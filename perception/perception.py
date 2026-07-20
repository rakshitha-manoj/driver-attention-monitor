"""
perception.py
Person A — Hafsa — Perception Layer
Driver Attention Monitoring Project

Changes from previous version:
  1. ear_confidence redesigned — no longer conflates closed eyes with low confidence
  2. blink_state "unknown" now a proper third state (flagged to team)
  3. blink_dur_avg added to output dict (Sheethal needs for scoring)
  4. blink_count and yawn_count now in output dict
  5. CSV logging added for Raks
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from collections import deque


# ─────────────────────────────────────────────────────────────────────
# MediaPipe landmark IDs — fixed, same person same IDs every frame
# ─────────────────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61, 291, 39, 181, 0, 17, 269, 405]


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────

def get_point(landmarks, idx, w, h):
    """Convert normalised landmark (0–1) to pixel coordinates."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def compute_ear(landmarks, eye_ids, w, h):
    """
    Eye Aspect Ratio.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Open eye ≈ 0.25–0.35. Closed eye ≈ 0.0
    """
    p = [get_point(landmarks, i, w, h) for i in eye_ids]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks, mouth_ids, w, h):
    """
    Mouth Aspect Ratio.
    MAR > 0.6 sustained for 1.5s = yawn event.
    """
    p = [get_point(landmarks, i, w, h) for i in mouth_ids]
    vertical_1 = np.linalg.norm(p[2] - p[6])
    vertical_2 = np.linalg.norm(p[3] - p[7])
    horizontal = np.linalg.norm(p[0] - p[1])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def apply_clahe(frame):
    """
    CLAHE — fixes uneven lighting before passing frame to MediaPipe.
    Applied to L channel of LAB colourspace only.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)


def compute_landmark_confidence(landmarks, eye_ids):
    """
    ── CHANGED from previous version ──

    OLD code (wrong):
        ear_confidence = round(1.0 if ear > 0.15 else 0.5, 2)

    WHY IT WAS WRONG:
        When a driver is drowsy, EAR drops to near 0.
        Old code saw EAR < 0.15 and said "low confidence."
        But detection was working perfectly — eyes ARE closed.
        Drowsy = low EAR is correct, not a failure.

    NEW approach:
        Use MediaPipe's own .visibility score per landmark (0.0–1.0).
        Visibility goes LOW when:
          - Glasses frames physically block eye corner landmarks
          - Extreme camera angle pushes landmarks off-screen
          - Very dark lighting makes eye region undetectable
        Visibility stays HIGH even when eyes are genuinely closed.

        This correctly distinguishes:
          Eyes closed (drowsy) → low EAR, HIGH confidence → detection working
          Eyes blocked by glasses → low EAR, LOW confidence → measurement unreliable
    """
    visibility_scores = []
    for idx in eye_ids:
        lm = landmarks[idx]
        if hasattr(lm, 'visibility'):
            visibility_scores.append(lm.visibility)

    if not visibility_scores:
        return 1.0  # visibility not in this MediaPipe version, assume fine

    mean_visibility = np.mean(visibility_scores)

    if mean_visibility >= 0.6:
        return 1.0
    elif mean_visibility >= 0.3:
        return 0.5
    else:
        return 0.2


# ─────────────────────────────────────────────────────────────────────
# Main Perception class
# ─────────────────────────────────────────────────────────────────────

class PerceptionModule:

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── EAR threshold ──
        self.ear_threshold     = 0.25
        self.calibrated        = False
        self.calibration_ears  = []
        self.calibration_secs  = 10
        self.calibration_start = None
        self.calibration_variance = 0.0  # NEW: stored for diagnostics

        # ── Blink tracking ──
        self.blink_count      = 0
        self.blink_state      = "open"
        self.eye_closed_start = None
        self.BLINK_MAX_DUR    = 0.4

        # NEW: rolling window of last 20 blink durations
        self.blink_durations = deque(maxlen=20)
        self.blink_dur_avg   = 0.0   # Sheethal needs this for her scoring formula

        # ── Yawn tracking ──
        self.yawn_count       = 0
        self.mouth_open_start = None
        self.yawn_in_progress = False
        self.MAR_THRESHOLD    = 0.6
        self.YAWN_MIN_DUR     = 1.5

        self.frame_id      = 0
        self.session_start = time.time()


    def _run_calibration(self, ear):
        """
        First 10 seconds: observe open-eye EAR.
        Personalised threshold = mean - 2*std, floored at 0.15.

        NEW: also stores calibration_variance.
        High variance = person was blinking during calibration
        = threshold is less reliable = diagnostic warning printed.
        """
        now = time.time()

        if self.calibration_start is None:
            self.calibration_start = now
            print("Calibration started — keep eyes open for 10 seconds...")

        elapsed = now - self.calibration_start

        if elapsed < self.calibration_secs:
            self.calibration_ears.append(ear)
            return False

        if not self.calibrated and len(self.calibration_ears) > 10:
            mean = np.mean(self.calibration_ears)
            std  = np.std(self.calibration_ears)
            self.calibration_variance = round(float(std), 4)  # NEW
            self.ear_threshold = max(mean - 2 * std, 0.15)
            self.calibrated    = True
            print(f"Calibration done. Threshold: {self.ear_threshold:.4f}")
            if std > 0.03:
                print(f"WARNING: High variance ({std:.4f}) — were you blinking?")
        return True


    def _update_blink(self, ear, timestamp):
        """
        Blink = EAR dip + recovery within 400ms.
        Drowsy hold = EAR below threshold for > 400ms (NOT a blink).

        CHANGED: now tracks blink duration for rolling average.
        blink_dur_avg is the new output key Sheethal needs.
        Alert blinks ~150ms. Drowsy blinks ~300ms+.
        """
        if ear < self.ear_threshold:
            self.blink_state = "closed"
            if self.eye_closed_start is None:
                self.eye_closed_start = timestamp
        else:
            if self.blink_state == "closed" and self.eye_closed_start is not None:
                duration = timestamp - self.eye_closed_start
                if duration <= self.BLINK_MAX_DUR:
                    self.blink_count += 1
                    # NEW: track duration, update rolling average
                    self.blink_durations.append(duration)
                    self.blink_dur_avg = round(float(np.mean(self.blink_durations)), 4)
            self.blink_state      = "open"
            self.eye_closed_start = None


    def _update_yawn(self, mar, timestamp):
        """
        Yawn = MAR > 0.6 sustained for 1.5 seconds.
        Counted once per yawn event (not per frame).
        Unchanged from previous version.
        """
        if mar > self.MAR_THRESHOLD:
            if self.mouth_open_start is None:
                self.mouth_open_start = timestamp
            elif (timestamp - self.mouth_open_start) >= self.YAWN_MIN_DUR:
                if not self.yawn_in_progress:
                    self.yawn_count      += 1
                    self.yawn_in_progress = True
        else:
            self.mouth_open_start = None
            self.yawn_in_progress = False


    def process_frame(self, frame):
        """
        Main entry point. Call every frame.
        Returns the data contract dict.

        CHANGED output dict:
          - ear_confidence: now uses visibility score not EAR value
          - blink_count: now included (was internal only before)
          - blink_dur_avg: NEW key — Sheethal needs for scoring
          - yawn_count: now included (was internal only before)
          - blink_state: "unknown" is now official third value
        """
        self.frame_id += 1
        timestamp = round(time.time() - self.session_start, 3)

        enhanced = apply_clahe(frame)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results  = self.face_mesh.process(rgb)

        # No face detected
        if not results.multi_face_landmarks:
            return {
                "frame_id"          : self.frame_id,
                "timestamp"         : timestamp,
                "EAR"               : None,
                "MAR"               : None,
                "blink_state"       : "unknown",  # official third state
                "blink_count"       : self.blink_count,
                "blink_dur_avg"     : self.blink_dur_avg,
                "yawn_count"        : self.yawn_count,
                "ear_confidence"    : 0.0,
                "landmarks_detected": False
            }

        landmarks = results.multi_face_landmarks[0].landmark
        h, w      = frame.shape[:2]

        left_ear  = compute_ear(landmarks, LEFT_EYE,  w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)
        ear       = round((left_ear + right_ear) / 2.0, 4)
        mar       = round(compute_mar(landmarks, MOUTH, w, h), 4)

        if not self.calibrated:
            self._run_calibration(ear)

        self._update_blink(ear, timestamp)
        self._update_yawn(mar, timestamp)

        # NEW confidence logic using visibility, not EAR value
        ear_conf_left  = compute_landmark_confidence(landmarks, LEFT_EYE)
        ear_conf_right = compute_landmark_confidence(landmarks, RIGHT_EYE)
        ear_confidence = round((ear_conf_left + ear_conf_right) / 2.0, 2)

        return {
            "frame_id"          : self.frame_id,
            "timestamp"         : timestamp,
            "EAR"               : ear,
            "MAR"               : mar,
            "blink_state"       : self.blink_state,
            "blink_count"       : self.blink_count,       # now in dict
            "blink_dur_avg"     : self.blink_dur_avg,     # NEW
            "yawn_count"        : self.yawn_count,         # now in dict
            "ear_confidence"    : ear_confidence,
            "landmarks_detected": True
        }


# ─────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────

class PerceptionLogger:
    """
    NEW — was not in previous version.
    Writes every frame dict to CSV for Raks to run batch analysis.
    """
    def __init__(self, filepath="perception/output_log.csv"):
        directory = os.path.dirname(filepath)

        if directory:
             os.makedirs(directory, exist_ok=True)
        self.filepath = filepath
        self.file     = open(filepath, "w", newline="")
        self.writer   = None

    def log(self, frame_dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=frame_dict.keys())
            self.writer.writeheader()
        self.writer.writerow(frame_dict)

    def close(self):
        self.file.close()
        print(f"Log saved: {self.filepath}")


# ─────────────────────────────────────────────────────────────────────
# Live demo
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    module = PerceptionModule()
    logger = PerceptionLogger("output_log.csv")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        print("Mac: System Settings → Privacy & Security → Camera → allow Terminal")
        exit()

    print("Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = module.process_frame(frame)
        logger.log(output)

        if not module.calibrated:
            elapsed   = time.time() - (module.calibration_start or time.time())
            remaining = max(0, module.calibration_secs - elapsed)
            line1     = f"CALIBRATING — keep eyes open ({remaining:.0f}s)"
            c1        = (0, 165, 255)
        else:
            ear_s = f"{output['EAR']:.4f}" if output['EAR'] is not None else "—"
            mar_s = f"{output['MAR']:.4f}" if output['MAR'] is not None else "—"
            line1 = f"EAR:{ear_s} MAR:{mar_s} Conf:{output['ear_confidence']} Thresh:{module.ear_threshold:.3f}"
            c1    = (0, 255, 0)

        line2 = (f"Blinks:{output['blink_count']} "
                 f"AvgDur:{output['blink_dur_avg']:.3f}s "
                 f"Yawns:{output['yawn_count']} "
                 f"Eyes:{output['blink_state']}")
        line3 = "Face:YES" if output["landmarks_detected"] else "Face:NO"
        c3    = (0, 255, 0) if output["landmarks_detected"] else (0, 0, 255)

        cv2.putText(frame, line1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c1, 2)
        cv2.putText(frame, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, line3, (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c3, 2)

        if output["frame_id"] % 30 == 0:
            print(output)

        cv2.imshow("Perception Module — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.close()