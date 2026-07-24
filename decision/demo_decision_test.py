"""
Standalone test harness for the Decision module (items 2-8), using
SIMULATED Perception values (keypress-toggled eyes-closed, no real
yawn/blink data).

Hafsa's real Perception module is now wired in separately, in
integrated_demo.py -- use that one for anything involving actual
EAR/MAR/yawn/blink behavior. This file still has a use: it's a
lighter-weight way to test nod/distraction/PERCLOS/scoring/state-
machine logic in isolation, without needing her module running or
her FaceMesh instance's extra overhead.

Separate from demo.py so your working head-pose demo stays intact.

Controls: [c] toggle simulated eye-closed   [q] quit
"""

import time
import cv2
import mediapipe as mp

from config import LANDMARKS
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator
from nod_detector import NodDetector
from distraction_detector import DistractionDetector
from perclos import PerclosWindow
from scoring import compute_drowsiness_score
from state_machine import DrowsinessStateMachine
from alert_system import AlertSystem
from pose_sanity import is_plausible
from face_loss_detector import FaceLossDetector
from nod_rate import NodRateWindow


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_estimator = HeadPoseEstimator()
calibrator = PoseCalibrator(calibration_duration=3.0)
nod_detector = NodDetector()
distraction_detector = DistractionDetector()
perclos_window = PerclosWindow(window_seconds=60.0)
state_machine = DrowsinessStateMachine()
alert_system = AlertSystem()
face_loss_detector = FaceLossDetector(loss_threshold_seconds=2.0)
nod_rate_window = NodRateWindow(window_seconds=60.0)
NOD_ALERT_THRESHOLD = 3  # nods within the window to trigger the sticky alert
_last_nod_count = 0

# --- simulated Perception output -- intentional for this lightweight
# harness. For real EAR/MAR/yawn/blink data, use integrated_demo.py ---
simulated_eyes_closed = False
simulated_yawn_count = 0
simulated_blink_duration = 0.0
# ----------------------------------------------------------------------

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    now = time.time()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    pitch = yaw = roll = 0.0
    distraction_flag = False
    face_detected = bool(results.multi_face_landmarks)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )

            pose = pose_estimator.estimate_pose(face_landmarks.landmark, frame)

            if pose is not None:
                if not calibrator.is_calibrated:
                    calibrator.calibrate(pose)

                pose = calibrator.get_relative_pose(pose)
                pitch, yaw, roll = pose["pitch"], pose["yaw"], pose["roll"]

                if is_plausible(pitch, yaw, roll):
                    nod_detector.update(pitch, yaw=yaw, roll=roll, now=now)
                    distraction_flag = distraction_detector.update(yaw, pitch=pitch, now=now)
                # else: bad frame (likely face partially out of view) --
                # skip updating either detector rather than trust it

    perclos = perclos_window.update(simulated_eyes_closed, now)

    score, weights_used = compute_drowsiness_score(
        perclos=perclos,
        yawn_rate=simulated_yawn_count,
        blink_duration=simulated_blink_duration,
        nod_count=nod_detector.nod_count,
        ear_confidence=1.0,
    )

    face_lost_alert = face_loss_detector.update(face_detected, now)

    if nod_detector.nod_count > _last_nod_count:
        for _ in range(nod_detector.nod_count - _last_nod_count):
            nod_rate_window.add_nod(now)
        _last_nod_count = nod_detector.nod_count

    drowsy_nod_alert = nod_rate_window.count_recent(now) >= NOD_ALERT_THRESHOLD

    state = state_machine.update(score, now)
    frame, alert_fired = alert_system.update(frame, state, distraction_flag,
                                              face_lost_alert, drowsy_nod_alert, now)

    hud_lines = [
        f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
        f"Nods: {nod_detector.nod_count}  Distraction: {distraction_flag}"
        f" ({distraction_detector.distraction_axis})" if distraction_flag else
        f"Nods: {nod_detector.nod_count}  Distraction: {distraction_flag}",
        f"PERCLOS: {perclos * 100:.1f}%  Score: {score:.1f}  State: {state}",
        "[c] toggle simulated eye-closed   [q] quit",
    ]
    y = 70
    for line in hud_lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        y += 22

    cv2.imshow("Decision Module Test Harness", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("c"):
        simulated_eyes_closed = not simulated_eyes_closed

cap.release()
cv2.destroyAllWindows()
