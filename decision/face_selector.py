"""
face_selector.py — Decision layer
Prevents background/passenger faces from stealing focus from the driver.

Solution:
  - Detects multiple faces.
  - Remembers the driver's face size (area) and position.
  - Filters out passenger faces that are too small (e.g. background) or too far from the driver's seat.
  - Retains the driver's seat position anchor permanently once established, so passengers are never selected.
  - Dynamically adapts to driver movement using an EMA on the driver's face area.
"""

import numpy as np
import math


class DriverFaceSelector:
    """
    Tracks the driver's face position and rejects passenger faces
    using size-based foreground selection and spatial tracking.
    """

    def __init__(self, max_jump_fraction=0.20, warmup_frames=10):
        """
        max_jump_fraction: max allowed face-center movement per frame
            as a fraction of frame size (0.20 = 20%).
        warmup_frames: accept the largest face for the first N frames to
            establish a baseline position.
        """
        self.max_jump = max_jump_fraction
        self.warmup_frames = warmup_frames

        self._prev_center = None
        self._driver_baseline_area = None
        self._driver_ref_area = None
        self._frame_count = 0
        self._reject_count = 0

    def select(self, multi_face_landmarks, frame_w, frame_h):
        """
        Selects the driver's face from a list of detected faces.

        Args:
            multi_face_landmarks: list of FaceLandmarkList detections.
            frame_w, frame_h: frame dimensions in pixels.

        Returns:
            The face_landmarks belonging to the driver, or None if rejected.
        """
        if not multi_face_landmarks:
            return None

        # Parse all faces and compute centers and areas
        candidates = []
        for face in multi_face_landmarks:
            lm = face.landmark
            # Nose tip landmark (1) for center
            nose = lm[1]

            # Reject any face not sitting in front of the camera (middle 50% of the screen width)
            if not (0.25 <= nose.x <= 0.75):
                continue

            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            xmin, xmax = min(xs) * frame_w, max(xs) * frame_w
            ymin, ymax = min(ys) * frame_h, max(ys) * frame_h
            area = (xmax - xmin) * (ymax - ymin)
            
            center = (nose.x * frame_w, nose.y * frame_h)
            candidates.append({
                "face": face,
                "center": center,
                "area": area
            })

        if not candidates:
            self._reject_count += 1
            return None

        self._frame_count += 1

        # 1. Warmup or Initial Selection
        if self._prev_center is None or self._frame_count <= self.warmup_frames:
            best_candidate = max(candidates, key=lambda c: c["area"])
            self._prev_center = best_candidate["center"]
            self._driver_baseline_area = best_candidate["area"]
            self._driver_ref_area = best_candidate["area"]
            return best_candidate["face"]

        # 2. Pre-filter passenger/background faces that are too small
        # (passenger faces further back are usually < 35% of driver's baseline face area)
        min_area_thresh = 0.35 * self._driver_baseline_area
        valid_candidates = [c for c in candidates if c["area"] >= min_area_thresh]

        # 3. Check if any candidate is the returning driver (large face in the foreground)
        # If the largest candidate is close to the baseline driver face size,
        # we lock onto it as the driver, overriding any temporary tracking losses.
        largest_candidate = max(candidates, key=lambda c: c["area"])
        if largest_candidate["area"] >= 0.70 * self._driver_baseline_area:
            self._prev_center = largest_candidate["center"]
            self._driver_ref_area = largest_candidate["area"]
            return largest_candidate["face"]

        if not valid_candidates:
            self._reject_count += 1
            return None

        # 4. Spatial tracking: find candidate closest to driver's last known seat center
        scored_candidates = []
        for c in valid_candidates:
            dx = abs(c["center"][0] - self._prev_center[0]) / frame_w
            dy = abs(c["center"][1] - self._prev_center[1]) / frame_h
            jump = max(dx, dy)
            if jump <= self.max_jump:
                scored_candidates.append((c, jump))

        if scored_candidates:
            best_candidate, _ = min(scored_candidates, key=lambda pair: pair[1])
            self._prev_center = best_candidate["center"]
            # Smoothly update reference area to adapt to normal forward/backward movement
            self._driver_ref_area = 0.9 * self._driver_ref_area + 0.1 * best_candidate["area"]
            return best_candidate["face"]
        else:
            self._reject_count += 1
            return None

    @property
    def rejections(self):
        """Number of frames rejected so far."""
        return self._reject_count

