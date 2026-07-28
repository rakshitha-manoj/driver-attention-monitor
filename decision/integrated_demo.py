"""
Integrated Decision + Perception test.

Replaces the simulated EAR/MAR/yawn values in demo_decision_test.py
with Hafsa's real PerceptionModule output. Head pose still uses its
own separate FaceMesh instance (decoupled from her module) -- worth
merging into a single shared instance at Week 6 integration, not now.

Assumes Hafsa's file lives at perception/perception.py in the repo
root (driver-attention-monitor/perception/perception.py). Adjust the
sys.path line below if her filename or folder differs.

Fix applied: yawn_rate now comes from a rolling YawnRateWindow instead
of the raw lifetime perception.yawn_count. The old version fed a
cumulative total into a rate-normalized scoring formula (max=5.0),
which permanently maxed out the yawn score component after the 5th
yawn of the whole session, even if the driver hadn't yawned in
20 minutes.
"""

import os
import sys
import time

import cv2
import mediapipe as mp

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "perception"))
from perception import PerceptionModule  # Hafsa's module

from config import LANDMARKS
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator
from nod_detector import NodDetector
from distraction_detector import DistractionDetector
from perclos import PerclosWindow
from yawn_rate import YawnRateWindow
from scoring import compute_drowsiness_score
from state_machine import DrowsinessStateMachine
from alert_system import AlertSystem
from pose_sanity import is_plausible
from face_loss_detector import FaceLossDetector
from output_contract import build_output_dict
from decision_logger import DecisionLogger
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

perception = PerceptionModule()  # Hafsa's module -- its own FaceMesh internally

pose_estimator = HeadPoseEstimator()
calibrator = PoseCalibrator(calibration_duration=3.0)
nod_detector = NodDetector()
distraction_detector = DistractionDetector()
perclos_window = PerclosWindow(window_seconds=60.0)
yawn_rate_window = YawnRateWindow(window_seconds=60.0)
state_machine = DrowsinessStateMachine()
alert_system = AlertSystem()
face_loss_detector = FaceLossDetector(loss_threshold_seconds=2.0)
decision_logger = DecisionLogger("output_log.csv")
nod_rate_window = NodRateWindow(window_seconds=60.0)
NOD_ALERT_THRESHOLD = 3
_last_nod_count = 0

_last_yawn_count = 0  # tracks perception_output["yawn_count"] between frames to detect new yawns

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    now = time.time()

    # ---- Perception: Hafsa's real output ----
    perception_output = perception.process_frame(frame.copy())

    ear_confidence = perception_output["ear_confidence"]
    is_eyes_closed = perception_output["blink_state"] == "closed"

    # blink_dur_avg is now a real contract field (rolling average of
    # her last 20 completed blinks) -- no more reaching into her
    # private eye_closed_start attribute to approximate it.
    blink_duration = perception_output["blink_dur_avg"]

    # Read from her output dict, not her internal attribute -- keeps
    # this file decoupled from her class internals.
    yawn_count = perception_output["yawn_count"]

    # Convert the lifetime cumulative yawn_count into a rolling rate.
    # Detect new yawns since last frame and feed each into the window,
    # so old yawns naturally age out after 60 seconds.
    if yawn_count > _last_yawn_count:
        for _ in range(yawn_count - _last_yawn_count):
            yawn_rate_window.add_yawn(now)
        _last_yawn_count = yawn_count

    yawn_rate = yawn_rate_window.rate_per_minute(now)

    # ---- Decision: head pose (own FaceMesh, decoupled from Perception) ----
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

    # ---- Fusion ----
    perclos = perclos_window.update(is_eyes_closed, now)

    score, weights_used = compute_drowsiness_score(
        perclos=perclos,
        yawn_rate=yawn_rate,
        blink_duration=blink_duration,
        nod_count=nod_detector.nod_count,
        ear_confidence=ear_confidence,
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

    # ---- Build and log the actual contract output ----
    # Uses Perception's frame_id so Raks can align both modules'
    # logs row-by-row.
    pose_plausible = is_plausible(pitch, yaw, roll)

    output_dict = build_output_dict(
        frame_id=perception_output["frame_id"],
        drowsiness_score=score,
        system_state=state,
        perclos=perclos,
        yawn_count=yawn_count,
        nod_count=nod_detector.nod_count,
        head_pitch=pitch,
        head_yaw=yaw,
        head_roll=roll,
        distraction_flag=distraction_flag,
        alert_fired=alert_fired,
        distraction_axis=distraction_detector.distraction_axis,
        face_lost_alert=face_lost_alert,
        pose_plausible=pose_plausible,
    )
    decision_logger.log(output_dict)

    # ---- HUD ----
    hud_lines = [
        f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
        f"Nods: {nod_detector.nod_count}  Distraction: {distraction_flag}",
        f"EAR: {perception_output['EAR']}  MAR: {perception_output['MAR']}  "
        f"Conf: {ear_confidence}",
        f"PERCLOS: {perclos * 100:.1f}%  Yawn rate: {yawn_rate:.1f}/min  "
        f"Score: {score:.1f}  State: {state}",
        "[q] quit",
    ]
    y = 70
    for line in hud_lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        y += 22

    cv2.imshow("Integrated Decision + Perception", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
decision_logger.close()
