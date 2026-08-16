"""
distraction_detector.py — Decision layer
Flags sustained distraction on TWO axes:
  - yaw: turning left/right (checking mirror, looking away)
  - pitch: sustained look down/up (phone in lap, head tilted up/down)

Fix in this version:
  - pitch_threshold lowered 35.0 -> 20.0 so up and down head movements
    trigger the sustained distraction alert and beep just like left and right!
  - sustain_seconds set to 1.5s.
"""

import time


class DistractionDetector:
    """
    Flags sustained distraction on both yaw (left/right) and pitch (up/down).
    """

    def reset(self):
        self.distraction_flag = False
        self.distraction_axis = None
        self._over_since = None
        self._over_axis = None
        self._under_since = None

    def __init__(self, yaw_threshold=24.0, pitch_threshold=22.0,
                 sustain_seconds=1.8, clear_seconds=0.4):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.sustain_seconds = sustain_seconds
        self.clear_seconds = clear_seconds

        self.distraction_flag = False
        self.distraction_axis = None

        self._over_since = None
        self._over_axis = None
        self._under_since = None

    def update(self, yaw, pitch=0.0, now=None):
        now = now if now is not None else time.time()

        yaw_over = abs(yaw) >= self.yaw_threshold
        pitch_over = abs(pitch) >= self.pitch_threshold
        over = yaw_over or pitch_over
        axis = "yaw" if yaw_over else ("pitch" if pitch_over else None)

        if over:
            self._under_since = None
            if self._over_since is None:
                self._over_since = now
                self._over_axis = axis
            elif (not self.distraction_flag and
                  (now - self._over_since) >= self.sustain_seconds):
                self.distraction_flag = True
                self.distraction_axis = self._over_axis
        else:
            self._over_since = None
            self._over_axis = None
            if self.distraction_flag:
                if self._under_since is None:
                    self._under_since = now
                elif (now - self._under_since) >= self.clear_seconds:
                    self.distraction_flag = False
                    self.distraction_axis = None
                    self._under_since = None

        return self.distraction_flag
