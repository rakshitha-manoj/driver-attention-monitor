"""
face_selector.py — Decision layer
Prevents background faces from stealing focus from the driver.

Problem: with max_num_faces=1, MediaPipe returns the single most
confident face.  When a passenger or passer-by appears behind the
driver, MediaPipe may switch to their face — and suddenly all
detectors (EAR, head pose, etc.) are reading the wrong person.

Solution: track the driver's face center across frames.  If the
detected face's center jumps more than `max_jump_fraction` of the
frame size in a single frame, reject that detection entirely (it's
a different person, not the driver moving).  The driver can move
normally because head turns rarely shift the face center by more
than 15-20 % of the frame in a single frame (~33 ms at 30 fps).
"""

import numpy as np


class DriverFaceSelector:
    """
    Tracks the driver's face position and rejects sudden jumps
    caused by background faces stealing MediaPipe's detection.
    """

    def __init__(self, max_jump_fraction=0.20, warmup_frames=10):
        """
        max_jump_fraction: max allowed face-center movement per frame
            as a fraction of frame size (0.20 = 20%).
        warmup_frames: accept any face for the first N frames to
            establish a baseline position.
        """
        self.max_jump = max_jump_fraction
        self.warmup_frames = warmup_frames

        self._prev_center = None
        self._frame_count = 0
        self._reject_count = 0

    def select(self, face_landmarks, frame_w, frame_h):
        """
        Accepts or rejects a face detection based on position tracking.

        Args:
            face_landmarks: a single FaceLandmarkList (not a list of faces).
                Pass None if no face was detected this frame.
            frame_w, frame_h: frame dimensions in pixels.

        Returns:
            The same face_landmarks if accepted, or None if rejected
            (background face jump detected).
        """
        if face_landmarks is None:
            return None

        self._frame_count += 1
        center = self._get_center(face_landmarks, frame_w, frame_h)

        # During warmup, accept anything to establish baseline
        if self._frame_count <= self.warmup_frames:
            self._prev_center = center
            return face_landmarks

        # Check if the face jumped too far
        if self._prev_center is not None:
            dx = abs(center[0] - self._prev_center[0]) / frame_w
            dy = abs(center[1] - self._prev_center[1]) / frame_h
            jump = max(dx, dy)

            if jump > self.max_jump:
                self._reject_count += 1
                # Don't update _prev_center — keep tracking where
                # the driver's face WAS, not where the interloper is
                return None

        self._prev_center = center
        return face_landmarks

    @staticmethod
    def _get_center(face_landmarks, w, h):
        """Compute face center from nose tip (landmark 1)."""
        nose = face_landmarks.landmark[1]
        return (nose.x * w, nose.y * h)

    @property
    def rejections(self):
        """Number of frames rejected so far."""
        return self._reject_count
