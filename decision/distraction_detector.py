import time


class DistractionDetector:
    """
    Flags sustained distraction on TWO axes now, not just yaw:
    - yaw: turning left/right (checking mirror, looking away)
    - pitch: sustained look down/up (phone in lap, head tilted back)

    Both use the same "held past threshold for sustain_seconds"
    logic. Pitch distraction is intentionally a SEPARATE concept
    from nod_detector's quick dip-and-recover -- a driver looking
    down at their phone for 4 seconds is not a nod, and shouldn't
    be caught or missed by the nod logic.

    pitch_threshold defaults higher (35 vs 30 for yaw) because a
    normal driving posture already involves some downward pitch
    looking at the dashboard/mirrors -- tune both against your own
    footage, these are starting points, not measured values.
    """

    def __init__(self, yaw_threshold=20.0, pitch_threshold=35.0,
                 sustain_seconds=2.0, clear_seconds=0.5):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.sustain_seconds = sustain_seconds
        self.clear_seconds = clear_seconds

        self.distraction_flag = False
        self.distraction_axis = None  # "yaw" or "pitch", for HUD/debug

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
