"""
app.py — Unified Driver Attention Monitor Demo (Integration Only)
=================================================================

This file ONLY combines the three modules. All logic lives in:
  - perception/perception.py  (EAR, MAR, blink, yawn, CLAHE, calibration)
  - decision/                 (head pose, nod, distraction, scoring, alerts, face selector)
  - evaluation/               (iris/gaze, hand/object detection, logging)

Architectural features integrated here:
  - Adaptive Sensor Scheduling: scales monitoring intensity to risk level
  - Escalating Alert Response: graduated visual/audio to prevent alarm fatigue
  - On-Device Privacy: explicit declaration that all processing is local
  - Cross-Session Calibration: learns driver baseline over multiple sessions

Run:
  python app.py
  python app.py --source 1
  python app.py --source path/to/video.mp4
  python app.py --no-hands --no-mesh
  python app.py --no-adaptive

Press Q to quit.
"""

import os
import sys
import time
import argparse
import csv

import cv2
import numpy as np
import mediapipe as mp

# ── Module paths ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "perception"))
sys.path.insert(0, os.path.join(ROOT, "decision"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

# ── Perception ───────────────────────────────────────────────────────
from perception import PerceptionModule, apply_clahe

# ── Decision ─────────────────────────────────────────────────────────
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator
from nod_detector import NodDetector
from distraction_detector import DistractionDetector
from perclos import PerclosWindow
from yawn_rate import YawnRateWindow
from nod_rate import NodRateWindow
from scoring import compute_drowsiness_score
from state_machine import DrowsinessStateMachine
from alert_system import AlertSystem
from face_loss_detector import FaceLossDetector
from output_contract import build_output_dict
from pose_sanity import is_plausible
from face_selector import DriverFaceSelector
from adaptive_scheduler import AdaptiveSensorScheduler
from privacy_guard import get_privacy_summary, verify_no_network, get_log_privacy_note
from calibration_store import CalibrationStore
from session_reporter import SessionReporter

# ── Evaluation ───────────────────────────────────────────────────────
from iris_tracker import GazeTracker
from hand_detector import HandObjectDetector


# =====================================================================
#  Head Pose EMA Smoother
# =====================================================================

class PoseSmoother:
    """Exponential moving average filter for head pose angles."""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.pitch = self.yaw = self.roll = 0.0
        self._init = False

    def update(self, pitch, yaw, roll):
        if not self._init:
            self.pitch, self.yaw, self.roll = pitch, yaw, roll
            self._init = True
        else:
            a = self.alpha
            self.pitch = a * pitch + (1 - a) * self.pitch
            self.yaw   = a * yaw   + (1 - a) * self.yaw
            self.roll  = a * roll  + (1 - a) * self.roll
        return self.pitch, self.yaw, self.roll

    def reset(self):
        self._init = False


# =====================================================================
#  CSV Logger
# =====================================================================

class EventLogger:
    """Combined CSV logger: all signals in a single row per frame."""
    def __init__(self, filepath="output_log.csv"):
        d = os.path.dirname(filepath)
        if d:
            os.makedirs(d, exist_ok=True)
        self.filepath = filepath
        self.file = open(filepath, "w", newline="")
        self.writer = None

    def log(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()
        print(f"Log saved: {self.filepath}")


# =====================================================================
#  HUD
# =====================================================================

STATE_COLORS = {"ALERT": (0,200,0), "WARNING": (0,165,255), "CRITICAL": (0,0,255)}

def draw_hud(frame, d):
    h, w = frame.shape[:2]
    state = d.get("system_state", "ALERT")
    color = STATE_COLORS.get(state, (255,255,255))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent top bar
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w, 210), (20,20,20), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # State badge
    badge = f"  {state}  "
    (tw, th), _ = cv2.getTextSize(badge, font, 0.7, 2)
    bx = w - tw - 15
    cv2.rectangle(frame, (bx-5,10), (bx+tw+5, 10+th+12), color, -1)
    cv2.putText(frame, badge, (bx, 10+th+5), font, 0.7, (0,0,0), 2)

    # Score bar
    sc = d.get("drowsiness_score", 0)
    cv2.rectangle(frame, (15,15), (235,33), (60,60,60), -1)
    cv2.rectangle(frame, (15,15), (15+int(220*min(sc,100)/100), 33), color, -1)
    cv2.putText(frame, f"Drowsiness: {sc:.0f}/100", (20,29), font, 0.45, (255,255,255), 1)

    # Info lines
    s = 0.44
    y = 52
    ear_s = f"{d['EAR']:.3f}" if d.get('EAR') is not None else "---"
    mar_s = f"{d['MAR']:.3f}" if d.get('MAR') is not None else "---"

    cv2.putText(frame, f"EAR:{ear_s}  MAR:{mar_s}  Conf:{d.get('ear_confidence',0):.1f}  Thresh:{d.get('ear_threshold',0.25):.3f}",
                (15,y), font, s, (200,230,200), 1)
    y += 20
    cv2.putText(frame, f"Blinks:{d.get('blink_count',0)}  AvgDur:{d.get('blink_dur_avg',0):.3f}s  Yawns:{d.get('yawn_count',0)}  Rate:{d.get('yawn_rate',0):.1f}/m  Eyes:{d.get('blink_state','?')}",
                (15,y), font, s, (200,230,255), 1)
    y += 20
    cv2.putText(frame, f"Pitch:{d.get('head_pitch',0):+.1f}  Yaw:{d.get('head_yaw',0):+.1f}  Roll:{d.get('head_roll',0):+.1f}  Nods:{d.get('nod_count',0)}",
                (15,y), font, s, (255,220,180), 1)
    y += 20
    gh, gv = d.get('gaze_h',0.5), d.get('gaze_v',0.5)
    gd = "CENTER" if 0.35<gh<0.65 else ("LEFT" if gh<=0.35 else "RIGHT")
    cv2.putText(frame, f"PERCLOS:{d.get('PERCLOS',0)*100:.1f}%  Gaze:{gh:.2f}({gd})  V:{gv:.2f}",
                (15,y), font, s, (180,220,255), 1)
    y += 20
    obj = d.get("object_flag", False)
    hd  = d.get("hand_detected", False)
    hs  = f"Hands:{d.get('hand_count',0)}"
    if obj: hs += "  >>> OBJECT IN HAND <<<"
    hc = (0,0,255) if obj else ((0,200,200) if hd else (120,160,120))
    cv2.putText(frame, hs, (15,y), font, s, hc, 1)

    # Flags
    y += 20
    fl = []
    if d.get("distraction_flag"): fl.append(f"DISTRACTION({d.get('distraction_axis','?')})")
    if d.get("gaze_distraction"): fl.append("GAZE DISTRACTION")
    if d.get("face_lost_alert"):  fl.append("FACE LOST")
    if d.get("drowsy_nod_alert"): fl.append("REPEATED NODS")
    if d.get("object_flag"):      fl.append("OBJECT IN HAND")
    if d.get("face_rejected"):    fl.append("BG FACE REJECTED")
    fs = "  |  ".join(fl) if fl else "No active flags"
    cv2.putText(frame, fs, (15,y), font, s, (0,0,255) if fl else (100,180,100), 1)

    # Adaptive scheduler level indicator
    y += 20
    sched = d.get("schedule_level", "---")
    sched_color = {
        "RELAXED": (100, 200, 100), "ATTENTIVE": (0, 200, 200),
        "ELEVATED": (0, 165, 255), "MAXIMUM": (0, 0, 255)
    }.get(sched, (150, 150, 150))
    cv2.putText(frame, f"Monitor: {sched}", (15,y), font, s, sched_color, 1)

    # Cross-session indicator
    if d.get("has_calibration_history"):
        cv2.putText(frame, f"[Sessions: {d.get('session_count', 0)}]",
                    (170, y), font, 0.38, (150, 180, 150), 1)

    # Calibration
    if d.get("calibrating"):
        cv2.putText(frame, f"CALIBRATING - look straight ({d.get('calibration_remaining',0):.0f}s)",
                    (15,h-20), font, 0.55, (0,165,255), 2)

    cv2.putText(frame, "[Q] Quit", (w-100,h-10), font, 0.45, (120,120,120), 1)
    return frame


# =====================================================================
#  Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Driver Attention Monitor")
    p.add_argument("--source",      default="0", help="Webcam index or video path")
    p.add_argument("--no-log",      action="store_true")
    p.add_argument("--no-mesh",     action="store_true", help="Skip face mesh drawing")
    p.add_argument("--no-hands",    action="store_true", help="Disable hand detection")
    p.add_argument("--no-adaptive", action="store_true", help="Disable adaptive scheduling (run all sensors every frame)")
    p.add_argument("--no-persist",  action="store_true", help="Disable cross-session calibration persistence")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Privacy declaration (Feature 3) ──
    print(get_privacy_summary())
    verify_no_network(verbose=True)

    # ── Video ──
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open '{args.source}'"); sys.exit(1)

    # ── MediaPipe FaceMesh (shared) ──
    mp_face_mesh = mp.solutions.face_mesh
    mp_draw      = mp.solutions.drawing_utils
    mp_styles    = mp.solutions.drawing_styles
    face_mesh    = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ── Cross-Session Calibration (Feature 4) ──
    calib_store = None
    if not args.no_persist:
        calib_store = CalibrationStore()
        calib_store.load()
        print(f"  {calib_store.get_summary()}")

    # ── Modules ──
    perception  = PerceptionModule()
    pose_est    = HeadPoseEstimator()
    calibrator  = PoseCalibrator(calibration_duration=3.0)
    smoother    = PoseSmoother(alpha=0.4)
    nod_det     = NodDetector()
    distr_det   = DistractionDetector(yaw_threshold=20.0, pitch_threshold=20.0, sustain_seconds=5.0)
    perclos_win = PerclosWindow(window_seconds=60.0)
    yawn_win    = YawnRateWindow(window_seconds=60.0)
    nod_win     = NodRateWindow(window_seconds=60.0)
    state_mach  = DrowsinessStateMachine()
    alert_sys   = AlertSystem()
    face_loss   = FaceLossDetector(loss_threshold_seconds=2.0)
    face_sel    = DriverFaceSelector(max_jump_fraction=0.20)
    gaze        = GazeTracker(h_alpha=0.3, v_alpha=0.3, h_amplify=1.5, v_amplify=1.5, sustain_seconds=5.0)
    hand_det    = None if args.no_hands else HandObjectDetector(sustain_seconds=5.0)
    session_reporter = SessionReporter(log_filepath="output_log.csv", results_dir="results")
    logger      = None if args.no_log   else EventLogger("output_log.csv")

    # ── Adaptive Scheduler (Feature 1) ──
    scheduler = AdaptiveSensorScheduler(enabled=not args.no_adaptive)

    # ── Load historical calibration data ──
    if calib_store and calib_store.has_history:
        # Pre-seed EAR threshold from cross-session average
        hist_ear = calib_store.get_personalized_ear_threshold()
        perception.import_calibration({
            "ear_threshold": hist_ear,
            "ear_variance": 0.0,
        })
        # Pre-seed pose offsets
        hist_pose = calib_store.get_personalized_pose_offsets()
        if hist_pose:
            calibrator.import_offsets(hist_pose)
        # Pre-seed gaze baseline
        hist_gaze = calib_store.get_personalized_gaze_baseline()
        if hist_gaze:
            gaze.baseline_h = hist_gaze.get("h", 0.56)
            gaze.baseline_v = hist_gaze.get("v", 0.50)
            print(f"  Loaded historical gaze baseline: h={gaze.baseline_h:.3f}, v={gaze.baseline_v:.3f}")

    _last_yawn = _last_nod = 0
    NOD_ALERT_THRESH = 3
    gaze_calib_finalized = False
    session_start_time = time.time()

    # Microsleep tracking: sustained eye closure is qualitatively different
    # from gradual PERCLOS rise. A 3-second sustained closure is a microsleep
    # that should trigger WARNING/CRITICAL immediately.
    _eye_closed_since = None
    microsleep_seconds = 0.0

    # Cached results for adaptive scheduling
    _cached_gaze_h = 0.5
    _cached_gaze_v = 0.5
    _cached_obj_flag = False
    _cached_hand_found = False
    _cached_n_hands = 0

    print("=" * 60)
    window_name = "Driver Attention Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            alert_sys.reset_alerts()
            distr_det.reset()
            gaze.reset()

    cv2.setMouseCallback(window_name, on_mouse_click)

    smoothed_score = 0.0
    last_score_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        now = time.time()
        h_f, w_f = frame.shape[:2]

        # ═══════════════════════════════════════════════════════════
        # 1. Shared FaceMesh inference
        # ═══════════════════════════════════════════════════════════
        enhanced = apply_clahe(frame)
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # ═══════════════════════════════════════════════════════════
        # 2. Face selection (reject background face jumps)
        # ═══════════════════════════════════════════════════════════
        raw_faces = results.multi_face_landmarks if results.multi_face_landmarks else []
        face_lm  = face_sel.select(raw_faces, w_f, h_f)
        face_detected = (face_lm is not None)
        face_rejected = (len(raw_faces) > 0 and face_lm is None)

        # ═══════════════════════════════════════════════════════════
        # 3. Perception via shared landmarks (single FaceMesh)
        # ═══════════════════════════════════════════════════════════
        landmarks = face_lm.landmark if face_lm else None
        p_out = perception.process_landmarks(landmarks, w_f, h_f)

        # Calibration state check
        cal_ear = not perception.calibrated
        cal_pose = not calibrator.is_calibrated
        is_calibrating = cal_ear or cal_pose

        # ═══════════════════════════════════════════════════════════
        # 4. Decision & Evaluation from shared mesh
        # ═══════════════════════════════════════════════════════════
        pitch = yaw = roll = 0.0
        gaze_h, gaze_v = _cached_gaze_h, _cached_gaze_v
        distraction_flag = False
        pose_plausible = True
        raw_pose = None
        face_center_px = None

        if face_detected:
            # Face center for hand proximity check
            nose = landmarks[1]
            face_center_px = (int(nose.x * w_f), int(nose.y * h_f))

            # Draw face mesh
            if not args.no_mesh:
                mp_draw.draw_landmarks(
                    frame, face_lm,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())

            # Head pose
            raw_pose = pose_est.estimate_pose(landmarks, frame)
            if raw_pose is not None:
                if not calibrator.is_calibrated:
                    calibrator.calibrate(raw_pose)
                rel_pose = calibrator.get_relative_pose(raw_pose)
                pitch, yaw, roll = smoother.update(
                    rel_pose["pitch"], rel_pose["yaw"], rel_pose["roll"])
                pose_plausible = is_plausible(pitch, yaw, roll)
                if pose_plausible:
                    nod_det.update(pitch, yaw=yaw, roll=roll, now=now)
                    distraction_flag = distr_det.update(yaw, pitch=pitch, now=now)

                # Draw pose axes on the face (visual indicator)
                pose_est.draw_pose_axes(frame, landmarks, raw_pose)

            # Iris / Gaze — adaptive scheduling (Feature 1)
            run_gaze = scheduler.should_run_gaze(p_out["frame_id"])
            if is_calibrating:
                gaze.calibrate_sample(landmarks, w_f, h_f)
                gaze_h, gaze_v = gaze.update(landmarks, w_f, h_f, now=now)
            elif not gaze_calib_finalized:
                gaze.finalize_calibration()
                gaze_calib_finalized = True
                gaze_h, gaze_v = gaze.update(landmarks, w_f, h_f, now=now)
            elif run_gaze:
                gaze_h, gaze_v = gaze.update(landmarks, w_f, h_f, now=now)
            else:
                gaze_h, gaze_v = _cached_gaze_h, _cached_gaze_v

            _cached_gaze_h, _cached_gaze_v = gaze_h, gaze_v

            if run_gaze or is_calibrating:
                gaze.draw(frame, landmarks, w_f, h_f, gaze_h, gaze_v)
        else:
            pose_est.reset_tracking()
            smoother.reset()
            gaze_h, gaze_v = _cached_gaze_h, _cached_gaze_v

        # ═══════════════════════════════════════════════════════════
        # 5. Hand / Object detection — adaptive scheduling (Feature 1)
        # ═══════════════════════════════════════════════════════════
        obj_flag = _cached_obj_flag
        hand_found = _cached_hand_found
        n_hands = _cached_n_hands

        run_hands = scheduler.should_run_hands(p_out["frame_id"])
        if hand_det is not None and run_hands:
            obj_flag, hand_found, n_hands = hand_det.update(
                rgb, frame, now, face_center=face_center_px)
            _cached_obj_flag = obj_flag
            _cached_hand_found = hand_found
            _cached_n_hands = n_hands

        # ═══════════════════════════════════════════════════════════
        # 6. Fusion (PERCLOS, scoring, state machine, alerts)
        # ═══════════════════════════════════════════════════════════
        is_closed = (p_out["blink_state"] == "closed")
        perclos   = perclos_win.update(is_closed, now)

        # Microsleep tracking: how long have eyes been continuously closed?
        if is_closed:
            if _eye_closed_since is None:
                _eye_closed_since = now
            microsleep_seconds = now - _eye_closed_since
        else:
            _eye_closed_since = None
            microsleep_seconds = 0.0

        yc = p_out["yawn_count"]
        if yc > _last_yawn:
            for _ in range(yc - _last_yawn): yawn_win.add_yawn(now)
            _last_yawn = yc
        yr = yawn_win.rate_per_minute(now)

        if nod_det.nod_count > _last_nod:
            for _ in range(nod_det.nod_count - _last_nod): nod_win.add_nod(now)
            _last_nod = nod_det.nod_count
        drowsy_nod = nod_win.count_recent(now) >= NOD_ALERT_THRESH

        score, _ = compute_drowsiness_score(
            perclos=perclos, yawn_rate=yr,
            blink_duration=p_out["blink_dur_avg"],
            nod_count=nod_win.count_recent(now),
            microsleep_seconds=microsleep_seconds,
            ear_confidence=p_out["ear_confidence"])

        # Immediate escalation for acute microsleep events
        # A 2.2s continuous eye closure triggers WARNING; >= 3.5s triggers CRITICAL
        emergency_crit = (microsleep_seconds >= 3.5)
        emergency_warn = (microsleep_seconds >= 2.2)

        if emergency_crit:
            score = max(score, 75)
        elif emergency_warn:
            score = max(score, 45)

        # Asymmetric smoothing: rises instantly, decays slowly (e.g. 5.0 points/sec)
        if score > smoothed_score:
            smoothed_score = score
        else:
            dt = max(0.0, now - last_score_time)
            smoothed_score = max(score, smoothed_score - 5.0 * dt)
        last_score_time = now
        score = smoothed_score

        if is_calibrating:
            # During startup calibration, keep system in calm ALERT state with no false alarms
            score = 0.0
            microsleep_seconds = 0.0
            _eye_closed_since = None
            state = "ALERT"
            face_lost_alert = False
            drowsy_nod = False
            distraction_flag = False
            gaze_distraction = False
        else:
            face_lost_alert = face_loss.update(face_detected, now)
            state = state_mach.update(score, now,
                                      emergency_critical=emergency_crit,
                                      emergency_warning=emergency_warn)
            gaze_distraction = gaze.gaze_distraction_flag

        # Update adaptive scheduler with current risk (Feature 1)
        sched_level = scheduler.update(score, perclos, state, now)

        # Escalating alert response (Feature 2)
        frame, alert_fired = alert_sys.update(
            frame, state, distraction_flag, face_lost_alert, drowsy_nod,
            gaze_distraction_flag=gaze_distraction, now=now)

        # ═══════════════════════════════════════════════════════════
        # 7. HUD
        # ═══════════════════════════════════════════════════════════
        cal_rem = 0
        if cal_ear and perception.calibration_start:
            cal_rem = max(0, perception.calibration_secs - (now - perception.calibration_start))
        elif cal_pose:
            cal_rem = 3

        frame = draw_hud(frame, {
            "system_state": state, "drowsiness_score": score,
            "EAR": p_out["EAR"], "MAR": p_out["MAR"],
            "ear_confidence": p_out["ear_confidence"],
            "ear_threshold": perception.ear_threshold,
            "blink_count": p_out["blink_count"],
            "blink_dur_avg": p_out["blink_dur_avg"],
            "blink_state": p_out["blink_state"],
            "yawn_count": p_out["yawn_count"], "yawn_rate": yr,
            "head_pitch": pitch, "head_yaw": yaw, "head_roll": roll,
            "nod_count": nod_det.nod_count, "PERCLOS": perclos,
            "gaze_h": gaze_h, "gaze_v": gaze_v,
            "distraction_flag": distraction_flag,
            "distraction_axis": distr_det.distraction_axis,
            "gaze_distraction": gaze.gaze_distraction_flag,
            "face_lost_alert": face_lost_alert,
            "drowsy_nod_alert": drowsy_nod,
            "object_flag": obj_flag, "hand_detected": hand_found,
            "hand_count": n_hands, "face_rejected": face_rejected,
            "calibrating": is_calibrating,
            "calibration_remaining": cal_rem,
            "schedule_level": sched_level,
            "has_calibration_history": calib_store.has_history if calib_store else False,
            "session_count": calib_store.session_count if calib_store else 0,
        })

        # ═══════════════════════════════════════════════════════════
        # 8. Logging
        # ═══════════════════════════════════════════════════════════
        if logger:
            logger.log({
                "timestamp": p_out["timestamp"], "frame_id": p_out["frame_id"],
                "EAR": p_out["EAR"], "MAR": p_out["MAR"],
                "blink_state": p_out["blink_state"],
                "blink_count": p_out["blink_count"],
                "blink_dur_avg": p_out["blink_dur_avg"],
                "yawn_count": p_out["yawn_count"],
                "ear_confidence": p_out["ear_confidence"],
                "landmarks_detected": face_detected,
                "drowsiness_score": score, "system_state": state,
                "PERCLOS": round(perclos, 4), "yawn_rate": round(yr, 2),
                "nod_count": nod_det.nod_count,
                "head_pitch": round(pitch, 2), "head_yaw": round(yaw, 2),
                "head_roll": round(roll, 2),
                "gaze_h": round(gaze_h, 3), "gaze_v": round(gaze_v, 3),
                "distraction_flag": distraction_flag,
                "gaze_distraction": gaze.gaze_distraction_flag,
                "hand_detected": hand_found, "hand_count": n_hands,
                "object_in_hand": obj_flag,
                "face_rejected": face_rejected,
                "alert_fired": alert_fired,
                "schedule_level": sched_level,
            })

        # ═══════════════════════════════════════════════════════════
        # 9. Display
        # ═══════════════════════════════════════════════════════════
        cv2.imshow("Driver Attention Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if p_out["frame_id"] % 60 == 0:
            gd = "C" if 0.35<gaze_h<0.65 else ("L" if gaze_h<=0.35 else "R")
            hs = f"Hand:{'OBJ' if obj_flag else ('yes' if hand_found else 'no')}"
            print(f"[{p_out['timestamp']:>7.1f}s]  Score:{score:5.1f}  "
                  f"State:{state:<8s} PERCLOS:{perclos*100:5.1f}%  "
                  f"Yawns:{p_out['yawn_count']}  Nods:{nod_det.nod_count}  "
                  f"Gaze:{gaze_h:.2f}({gd})  {hs}  "
                  f"Sched:{sched_level}"
                  f"{'  [BG-REJECT]' if face_rejected else ''}")

    session_dur = time.time() - session_start_time

    # ═══════════════════════════════════════════════════════════════
    # Shutdown: save cross-session calibration (Feature 4)
    # ═══════════════════════════════════════════════════════════════
    if calib_store and perception.calibrated:
        ear_export = perception.export_calibration()
        pose_export = calibrator.export_offsets()
        gaze_baseline = {"h": gaze.baseline_h, "v": gaze.baseline_v}
        calib_store.add_session(
            ear_threshold=ear_export["ear_threshold"] if ear_export else 0.25,
            ear_variance=ear_export.get("ear_variance", 0) if ear_export else 0,
            pose_offsets=pose_export,
            gaze_baseline=gaze_baseline,
            session_duration_sec=session_dur,
            confidence=1.0 if ear_export else 0.5,
        )

    # Print adaptive scheduling savings report
    if scheduler.enabled:
        report = scheduler.get_savings_report()
        print(f"\nAdaptive scheduling savings:")
        print(f"  Total frames: {report['total_frames']}")
        print(f"  Hand detection: ran {report['hands_run']}x "
              f"(skipped {report['hands_skipped_pct']}%)")
        print(f"  Gaze tracking: ran {report['gaze_run']}x "
              f"(skipped {report['gaze_skipped_pct']}%)")

    cap.release()
    cv2.destroyAllWindows()
    if logger:
        logger.close()

    # Generate end-of-session driver trip report & timeline visualization
    session_reporter.generate_report(duration_sec=session_dur)
    print("Session ended.")


if __name__ == "__main__":
    main()
