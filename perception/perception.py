import cv2
import mediapipe as mp
import numpy as np
import time

# ─────────────────────────────────────────────
# MediaPipe landmark IDs — these are fixed
# Same person, same frame, same IDs every time
# ─────────────────────────────────────────────

# 6 points around each eye (p1 to p6)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 8 points around the mouth
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]


# ─────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────

def get_point(landmark, idx, w, h):
    """Convert normalised landmark (0-1) to pixel coordinates."""
    lm = landmark[idx]
    return np.array([lm.x * w, lm.y * h])


def compute_ear(landmarks, eye_ids, w, h):
    """
    Eye Aspect Ratio.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
    Open eye ≈ 0.25-0.35. Closed eye ≈ 0.0
    """
    p = [get_point(landmarks, i, w, h) for i in eye_ids]
    vertical_1 = np.linalg.norm(p[1] - p[5])
    vertical_2 = np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def compute_mar(landmarks, mouth_ids, w, h):
    """
    Mouth Aspect Ratio.
    High value = mouth open = possible yawn.
    MAR > 0.6 for 1.5 seconds = yawn event.
    """
    p = [get_point(landmarks, i, w, h) for i in mouth_ids]
    vertical_1 = np.linalg.norm(p[2] - p[6])
    vertical_2 = np.linalg.norm(p[3] - p[7])
    horizontal = np.linalg.norm(p[0] - p[1])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def apply_clahe(frame):
    """
    CLAHE — fixes uneven lighting (desk lamp, dark room, sunlight glare).
    Works on the L (lightness) channel of LAB colourspace.
    Apply this BEFORE passing frame to MediaPipe.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────
# Main Perception class
# ─────────────────────────────────────────────

class PerceptionModule:

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── EAR threshold (personalised during calibration) ──
        self.ear_threshold   = 0.25   # default, overwritten after calibration
        self.calibrated      = False
        self.calibration_ears = []
        self.calibration_secs = 10    # how long to observe open eyes
        self.calibration_start = None

        # ── Blink tracking ──
        self.blink_count      = 0
        self.blink_state      = "open"
        self.eye_closed_start = None
        self.BLINK_MAX_DUR    = 0.4   # seconds — longer than this = drowsy hold, not blink

        # ── Yawn tracking ──
        self.yawn_count        = 0
        self.mouth_open_start  = None
        self.yawn_in_progress  = False
        self.MAR_THRESHOLD     = 0.6
        self.YAWN_MIN_DUR      = 1.5  # seconds mouth must stay open

        self.frame_id      = 0
        self.session_start = time.time()


    # ── Calibration ──────────────────────────────────────────

    def _run_calibration(self, ear):
        """
        Collect EAR readings for first 10 seconds.
        After that, set personalised threshold = mean - 2*std.
        This prevents false alerts for people with naturally narrow eyes.
        Returns True when calibration is complete.
        """
        now = time.time()

        if self.calibration_start is None:
            self.calibration_start = now
            print("Calibration started — keep eyes open and look at the camera...")

        elapsed = now - self.calibration_start

        if elapsed < self.calibration_secs:
            self.calibration_ears.append(ear)
            return False   # still collecting

        # Calibration window over — compute threshold
        if not self.calibrated and len(self.calibration_ears) > 10:
            mean = np.mean(self.calibration_ears)
            std  = np.std(self.calibration_ears)
            self.ear_threshold = max(mean - 2 * std, 0.15)  # floor at 0.15 for safety
            self.calibrated = True
            print(f"Calibration done. Your EAR threshold: {self.ear_threshold:.3f}  "
                  f"(default was 0.25)")
        return True


    # ── Blink detection ──────────────────────────────────────

    def _update_blink(self, ear, timestamp):
        """
        A blink = EAR drops below threshold AND comes back up within 400ms.
        Longer closures (drowsy holds) are NOT counted as blinks.
        Updates self.blink_state and self.blink_count.
        """
        if ear < self.ear_threshold:
            # Eyes just closed or are staying closed
            self.blink_state = "closed"
            if self.eye_closed_start is None:
                self.eye_closed_start = timestamp

        else:
            # Eyes are now open
            if self.blink_state == "closed" and self.eye_closed_start is not None:
                duration = timestamp - self.eye_closed_start
                if duration <= self.BLINK_MAX_DUR:
                    self.blink_count += 1   # quick closure = blink
                # long closure = drowsy hold, Person B handles that via EAR value
            self.blink_state      = "open"
            self.eye_closed_start = None


    # ── Yawn detection ───────────────────────────────────────

    def _update_yawn(self, mar, timestamp):
        """
        A yawn = MAR > 0.6 sustained for at least 1.5 seconds.
        Counts each complete yawn event once (not every frame during the yawn).
        """
        if mar > self.MAR_THRESHOLD:
            if self.mouth_open_start is None:
                self.mouth_open_start = timestamp   # mouth just opened

            elif (timestamp - self.mouth_open_start) >= self.YAWN_MIN_DUR:
                if not self.yawn_in_progress:
                    self.yawn_count += 1
                    self.yawn_in_progress = True     # don't count again mid-yawn
        else:
            # Mouth closed — reset
            self.mouth_open_start = None
            self.yawn_in_progress = False


    # ── Main entry point ─────────────────────────────────────

    def process_frame(self, frame):
        """
        Call this every frame. Returns the data contract dict.
        This is what Sheethal (Decision) and Raks (Evaluation) consume.
        """
        self.frame_id += 1
        timestamp = round(time.time() - self.session_start, 3)

        # ── Preprocessing ──
        enhanced = apply_clahe(frame)
        rgb      = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results  = self.face_mesh.process(rgb)

        # ── No face detected ──
        if not results.multi_face_landmarks:
            return {
                "frame_id"          : self.frame_id,
                "timestamp"         : timestamp,
                "EAR"               : None,
                "MAR"               : None,
                "blink_state"       : "unknown",
                "ear_confidence"    : 0.0,
                "landmarks_detected": False
            }

        # ── Extract landmarks ──
        landmarks = results.multi_face_landmarks[0].landmark
        h, w      = frame.shape[:2]

        # ── Compute EAR (average both eyes) ──
        left_ear  = compute_ear(landmarks, LEFT_EYE,  w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE, w, h)
        ear       = round((left_ear + right_ear) / 2.0, 4)

        # ── Compute MAR ──
        mar = round(compute_mar(landmarks, MOUTH, w, h), 4)

        # ── Calibration (first 10 seconds) ──
        if not self.calibrated:
            self._run_calibration(ear)

        # ── Update blink and yawn state ──
        self._update_blink(ear, timestamp)
        self._update_yawn(mar, timestamp)

        # ── Confidence: lower when EAR is suspiciously low even when eyes look open ──
        # (can happen with heavy glasses, partial occlusion, or extreme angle)
        ear_confidence = round(1.0 if ear > 0.15 else 0.5, 2)

        return {
            "frame_id"          : self.frame_id,
            "timestamp"         : timestamp,
            "EAR"               : ear,
            "MAR"               : mar,
            "blink_state"       : self.blink_state,
            "ear_confidence"    : ear_confidence,
            "landmarks_detected": True
        }


# ─────────────────────────────────────────────
# Run this file directly to test your module
# python3 perception.py
# ─────────────────────────────────────────────

if __name__ == "__main__":
    module = PerceptionModule()
    cap    = cv2.VideoCapture(0)

    print("Perception module running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        print("ret =", ret)

        if not ret:
           print("Failed to read frame")
           break
        output = module.process_frame(frame)

        # ── On-screen display ──
        if not module.calibrated:
            top_line = "Calibrating — keep eyes open..."
        else:
            ear_val = output['EAR'] if output['EAR'] is not None else "—"
            mar_val = output['MAR'] if output['MAR'] is not None else "—"
            top_line = f"EAR: {ear_val}  |  MAR: {mar_val}  |  Threshold: {module.ear_threshold:.3f}"

        count_line = (f"Blinks: {module.blink_count}  |  "
                      f"Yawns: {module.yawn_count}  |  "
                      f"Eyes: {output['blink_state']}")

        detected = "Face: YES" if output["landmarks_detected"] else "Face: NO"

        cv2.putText(frame, top_line,   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255,   0), 2)
        cv2.putText(frame, count_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(frame, detected,   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # Print dict to terminal every second (approx 30 frames)
        if output["frame_id"] % 30 == 0:
            print(output)

        cv2.imshow("Perception Module — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()