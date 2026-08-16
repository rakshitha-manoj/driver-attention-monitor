"""
iris_tracker.py — Evaluation layer (Raks)
Real-time iris/gaze tracking using MediaPipe refined landmarks.

Uses iris center landmarks 468 (left) and 473 (right), which are
available when FaceMesh is initialized with refine_landmarks=True.

Fixes in this version:
  1. Dynamic Baseline Calibration: Auto-calibrates neutral gaze
     during the initial calibration phase. Eliminates nasal offset
     bias that previously caused gaze to concentrate on the Right (0.7-1.0).
  2. Sustained Gaze Distraction Detection: Flags sustained looking
     away (Left/Right/Up/Down) > sustain_seconds and triggers alerts.
  3. Orientation-aware ratios with smooth EMA filtering.
"""

import time
import cv2
import numpy as np


# Iris center landmark IDs (refine_landmarks=True required)
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

# Full iris ring for drawing (center + 4 cardinal points)
LEFT_IRIS_RING  = [468, 469, 470, 471, 472]
RIGHT_IRIS_RING = [473, 474, 475, 476, 477]

# Eye corner pairs per eye — sorted by x at runtime
LEFT_EYE_CORNERS  = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)

# Eye vertical landmarks for vertical gaze
LEFT_EYE_TOP     = [385, 387]
LEFT_EYE_BOTTOM  = [373, 380]
RIGHT_EYE_TOP    = [160, 158]
RIGHT_EYE_BOTTOM = [153, 144]


class GazeTracker:
    """
    Tracks horizontal and vertical gaze direction from iris landmarks.
    Outputs smoothed ratios: 0.0–1.0 for each axis.
    """

    def __init__(self, h_alpha=0.25, v_alpha=0.25,
                 h_amplify=1.15, v_amplify=1.15,
                 sustain_seconds=2.2):
        self.h_alpha   = h_alpha
        self.v_alpha   = v_alpha
        self.h_amplify = h_amplify
        self.v_amplify = v_amplify
        self.sustain_seconds = sustain_seconds

        self._h_smooth = 0.5
        self._v_smooth = 0.5
        self._initialized = False

        # Neutral baseline calibration
        self.baseline_h = 0.50
        self.baseline_v = 0.50
        self._calib_h_samples = []
        self._calib_v_samples = []

        # Sustained gaze distraction tracking
        self.gaze_distraction_flag = False
        self.gaze_distraction_axis = None
        self._off_center_since = None
        self._last_off_center_time = None

    def calibrate_sample(self, landmarks, w, h):
        """Accumulate neutral gaze samples during initial system calibration."""
        raw_h = self._compute_raw_horizontal(landmarks, w, h)
        raw_v = self._compute_raw_vertical(landmarks, w, h)
        if raw_h is not None and raw_v is not None:
            self._calib_h_samples.append(raw_h)
            self._calib_v_samples.append(raw_v)

    def finalize_calibration(self):
        """Finalize baseline neutral gaze values."""
        if len(self._calib_h_samples) > 10:
            self.baseline_h = float(np.clip(np.mean(self._calib_h_samples), 0.35, 0.65))
            self.baseline_v = float(np.clip(np.mean(self._calib_v_samples), 0.35, 0.65))
            print(f"Gaze calibration done: neutral_h={self.baseline_h:.3f}, neutral_v={self.baseline_v:.3f}")

    def update(self, landmarks, w, h, now=None):
        """
        Compute gaze from landmarks. Returns (gaze_h, gaze_v).
        Both are 0.0–1.0.  Horizontal: 0=left, 1=right, 0.5=center.
        Vertical: 0=up, 1=down, 0.5=center.
        """
        now = now if now is not None else time.time()

        raw_h = self._compute_raw_horizontal(landmarks, w, h)
        raw_v = self._compute_raw_vertical(landmarks, w, h)

        if raw_h is None or raw_v is None:
            return (self._h_smooth, self._v_smooth)

        # Center relative to baseline neutral gaze
        centered_h = (raw_h - self.baseline_h) * self.h_amplify + 0.5
        centered_v = (raw_v - self.baseline_v) * self.v_amplify + 0.5

        norm_h = float(np.clip(centered_h, 0.0, 1.0))
        norm_v = float(np.clip(centered_v, 0.0, 1.0))

        if not self._initialized:
            self._h_smooth = norm_h
            self._v_smooth = norm_v
            self._initialized = True
        else:
            self._h_smooth = (self.h_alpha * norm_h +
                              (1 - self.h_alpha) * self._h_smooth)
            self._v_smooth = (self.v_alpha * norm_v +
                              (1 - self.v_alpha) * self._v_smooth)

        # Update gaze distraction logic
        self._update_distraction(self._h_smooth, self._v_smooth, now)

        return (round(self._h_smooth, 3),
                round(self._v_smooth, 3))

    def _update_distraction(self, gh, gv, now):
        """Check for sustained off-center gaze (> sustain_seconds)."""
        off_h = (gh < 0.18 or gh > 0.82)
        off_v = (gv < 0.20 or gv > 0.80)
        is_off = off_h or off_v

        if is_off:
            if self._off_center_since is None:
                self._off_center_since = now
            self._last_off_center_time = now
            if (now - self._off_center_since) >= self.sustain_seconds:
                self.gaze_distraction_flag = True
                self.gaze_distraction_axis = "gaze_horizontal" if off_h else "gaze_vertical"
        else:
            grace_period = 0.5
            if self._off_center_since is not None:
                if self._last_off_center_time is None or (now - self._last_off_center_time) > grace_period:
                    self._off_center_since = None
                    self.gaze_distraction_flag = False
                    self.gaze_distraction_axis = None

    def reset(self):
        self._initialized = False
        self._h_smooth = 0.5
        self._v_smooth = 0.5
        self.gaze_distraction_flag = False
        self._off_center_since = None
        self._last_off_center_time = None

    # ─── Horizontal gaze ─────────────────────────────────────────

    def _compute_raw_horizontal(self, landmarks, w, h):
        ratios = []
        for iris_id, (ca, cb) in [
            (LEFT_IRIS_CENTER,  LEFT_EYE_CORNERS),
            (RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS),
        ]:
            iris_x = landmarks[iris_id].x * w
            xa = landmarks[ca].x * w
            xb = landmarks[cb].x * w
            left_x  = min(xa, xb)
            right_x = max(xa, xb)
            span = right_x - left_x
            if span < 3.0:
                continue
            ratio = (iris_x - left_x) / span
            ratios.append(ratio)

        if not ratios:
            return None
        return float(np.mean(ratios))

    # ─── Vertical gaze ───────────────────────────────────────────

    def _compute_raw_vertical(self, landmarks, w, h):
        ratios = []
        for iris_id, top_ids, bot_ids in [
            (LEFT_IRIS_CENTER,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM),
            (RIGHT_IRIS_CENTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
        ]:
            iris_y = landmarks[iris_id].y * h
            top_y  = np.mean([landmarks[i].y * h for i in top_ids])
            bot_y  = np.mean([landmarks[i].y * h for i in bot_ids])
            span = bot_y - top_y
            if span < 2.0:
                continue
            ratio = (iris_y - top_y) / span
            ratios.append(ratio)

        if not ratios:
            return None
        return float(np.mean(ratios))

    # ─── Drawing ─────────────────────────────────────────────────

    def draw(self, frame, landmarks, w, h, gaze_h, gaze_v):
        """Draw iris landmarks and gaze direction arrows on the frame."""
        for iris_ids, color in [
            (LEFT_IRIS_RING,  (0, 255, 128)),
            (RIGHT_IRIS_RING, (128, 255, 0)),
        ]:
            pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                   for i in iris_ids]
            center = pts[0]

            for pt in pts[1:]:
                cv2.circle(frame, pt, 2, color, -1)
            cv2.circle(frame, center, 4, color, -1)

            dx = int((gaze_h - 0.5) * 25)
            dy = int((gaze_v - 0.5) * 18)
            end_pt = (center[0] + dx, center[1] + dy)
            arrow_color = (0, 0, 255) if self.gaze_distraction_flag else (0, 200, 255)
            cv2.arrowedLine(frame, center, end_pt,
                            arrow_color, 2, tipLength=0.4)
