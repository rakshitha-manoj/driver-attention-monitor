"""
Integrated Decision + Perception test.

Replaces the simulated EAR/MAR/yawn values in demo_decision_test.py
with Hafsa's real PerceptionModule output. Head pose still uses its
own separate FaceMesh instance (decoupled from her module) -- worth
merging into a single shared instance at Week 6 integration, not now.

Assumes Hafsa's file lives at perception/perception.py in the repo
root (driver-attention-monitor/perception/perception.py). Adjust the
sys.path line below if her filename or folder differs.
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
from scoring import compute_drowsiness_score
from state_machine import DrowsinessStateMachine
from alert_system import AlertSystem
from pose_sanity import is_plausible


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
state_machine = DrowsinessStateMachine()
alert_system = AlertSystem()

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

    # Approximate current blink duration from her internal state
    # (not a field in the contract dict itself)
    if is_eyes_closed and perception.eye_closed_start is not None:
        blink_duration = now - perception.eye_closed_start
    else:
        blink_duration = 0.0

    yawn_count = perception.yawn_count

    # ---- Decision: head pose (own FaceMesh, decoupled from Perception) ----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    pitch = yaw = roll = 0.0
    distraction_flag = False

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
                    distraction_flag = distraction_detector.update(yaw, now)

    # ---- Fusion ----
    perclos = perclos_window.update(is_eyes_closed, now)

    score, weights_used = compute_drowsiness_score(
        perclos=perclos,
        yawn_rate=yawn_count,
        blink_duration=blink_duration,
        nod_count=nod_detector.nod_count,
        ear_confidence=ear_confidence,
    )

    state = state_machine.update(score, now)
    frame, alert_fired = alert_system.update(frame, state, distraction_flag)

    # ---- HUD ----
    hud_lines = [
        f"Pitch: {pitch:.1f}  Yaw: {yaw:.1f}  Roll: {roll:.1f}",
        f"Nods: {nod_detector.nod_count}  Distraction: {distraction_flag}",
        f"EAR: {perception_output['EAR']}  MAR: {perception_output['MAR']}  "
        f"Conf: {ear_confidence}",
        f"PERCLOS: {perclos * 100:.1f}%  Yawns: {yawn_count}  "
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
