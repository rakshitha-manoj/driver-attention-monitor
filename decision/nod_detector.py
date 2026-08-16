"""
nod_detector.py — Decision layer
Detects drowsy head nods: pitch dips past threshold and recovers.

Changes from previous version:
  1. Vibration guard — in a bumpy ride ALL angles (pitch, yaw, roll)
     oscillate together.  A real drowsy nod is pitch-dominant with
     stable yaw/roll.  The detector now tracks recent variance on all
     three axes and rejects pitch dips that coincide with high
     yaw+roll variance.
  2. Default drop_threshold lowered 20 → 15 degrees.
  3. Default max_concurrent_yaw widened 20 → 35 degrees.
"""

import time
import numpy as np
from collections import deque


class NodDetector:
    """
    Detects a 'nod' event: calibrated pitch dropping past
    `drop_threshold` and recovering to near-neutral within
    `recovery_window` seconds -- but only when the head is roughly
    forward-facing, not vibrating, and not more often than
    `cooldown_seconds`.

    Guards:
    - max_concurrent_yaw/roll: a genuine drowsy nod happens with the
      head facing forward.  Large yaw/roll alongside a pitch dip
      means the dip came from turning/tilting, not nodding.
    - cooldown_seconds: minimum time between two counted nods, so
      landmark jitter oscillating across the threshold during a
      single motion can't register as several nods.
    - vibration guard (NEW): if the standard deviation of recent
      yaw AND roll readings both exceed `vibration_std_threshold`,
      the current motion is classified as whole-body vibration
      (bumpy road) rather than a deliberate head nod.
    """

    def __init__(self, drop_threshold=15.0, recover_threshold=5.0,
                 recovery_window=2.0, direction=-1, max_plausible_pitch=70.0,
                 max_concurrent_yaw=35.0, max_concurrent_roll=30.0,
                 cooldown_seconds=1.5, vibration_std_threshold=4.0,
                 vibration_window=15):
        self.drop_threshold = drop_threshold
        self.recover_threshold = recover_threshold
        self.recovery_window = recovery_window
        self.direction = direction
        self.max_plausible_pitch = max_plausible_pitch
        self.max_concurrent_yaw = max_concurrent_yaw
        self.max_concurrent_roll = max_concurrent_roll
        self.cooldown_seconds = cooldown_seconds

        # Vibration guard parameters
        self.vibration_std_threshold = vibration_std_threshold
        self._recent_yaw  = deque(maxlen=vibration_window)
        self._recent_roll = deque(maxlen=vibration_window)

        self.nod_count = 0
        self._dip_start_time = None
        self._in_dip = False
        self._last_nod_time = None

    def _is_vibration(self):
        """
        Check if recent motion looks like whole-body vibration rather
        than a deliberate head nod.

        In a bumpy ride, yaw AND roll both oscillate with high variance.
        In a real drowsy nod, pitch changes while yaw/roll stay stable.
        Returns True if both yaw and roll variance exceed the threshold.
        """
        if len(self._recent_yaw) < 5:
            return False

        yaw_std  = float(np.std(list(self._recent_yaw)))
        roll_std = float(np.std(list(self._recent_roll)))

        return (yaw_std > self.vibration_std_threshold and
                roll_std > self.vibration_std_threshold)

    def update(self, pitch, yaw=0.0, roll=0.0, now=None):
        now = now if now is not None else time.time()

        # Track recent yaw/roll for vibration detection
        self._recent_yaw.append(yaw)
        self._recent_roll.append(roll)

        if abs(pitch) > self.max_plausible_pitch:
            return self.nod_count

        forward_facing = (abs(yaw) <= self.max_concurrent_yaw and
                           abs(roll) <= self.max_concurrent_roll)

        signed_pitch = pitch * self.direction

        if not self._in_dip:
            if (signed_pitch <= -self.drop_threshold and
                    forward_facing and
                    not self._is_vibration()):
                self._in_dip = True
                self._dip_start_time = now
        else:
            elapsed = now - self._dip_start_time

            if signed_pitch >= -self.recover_threshold:
                in_cooldown = (
                    self._last_nod_time is not None and
                    (now - self._last_nod_time) < self.cooldown_seconds
                )

                if (elapsed <= self.recovery_window and
                        forward_facing and
                        not in_cooldown and
                        not self._is_vibration()):
                    self.nod_count += 1
                    self._last_nod_time = now

                self._in_dip = False
                self._dip_start_time = None

            elif elapsed > self.recovery_window:
                # Head stayed down too long -- not a nod, don't count it.
                self._in_dip = False
                self._dip_start_time = None

        return self.nod_count

    def reset(self):
        self.nod_count = 0
        self._in_dip = False
        self._dip_start_time = None
        self._last_nod_time = None
        self._recent_yaw.clear()
        self._recent_roll.clear()
