import time


class DistractionDetector:
    """
    Flags sustained distraction: |yaw| beyond `yaw_threshold` for
    a continuous `sustain_seconds`. Clears after the yaw has been
    back under threshold for `clear_seconds` (small debounce so it
    doesn't flicker right at the boundary).
    """

    def __init__(self, yaw_threshold=30.0, sustain_seconds=2.0,
                 clear_seconds=0.5):
        self.yaw_threshold = yaw_threshold
        self.sustain_seconds = sustain_seconds
        self.clear_seconds = clear_seconds

        self.distraction_flag = False
        self._over_since = None
        self._under_since = None

    def update(self, yaw, now=None):
        now = now if now is not None else time.time()
        over = abs(yaw) >= self.yaw_threshold

        if over:
            self._under_since = None
            if self._over_since is None:
                self._over_since = now
            elif (not self.distraction_flag and
                  (now - self._over_since) >= self.sustain_seconds):
                self.distraction_flag = True
        else:
            self._over_since = None
            if self.distraction_flag:
                if self._under_since is None:
                    self._under_since = now
                elif (now - self._under_since) >= self.clear_seconds:
                    self.distraction_flag = False
                    self._under_since = None

        return self.distraction_flag
