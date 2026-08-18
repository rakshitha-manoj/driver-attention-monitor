from collections import deque

class YawnRateWindow:
    """
    Rolling yawn rate (yawns per minute) over a time window, same
    pattern as PerclosWindow. Fixes feeding a lifetime cumulative
    yawn count into compute_drowsiness_score, which saturates the
    yawn component permanently once 5 yawns have ever occurred.
    """
    def __init__(self, window_seconds=60.0):
        self.window_seconds = window_seconds
        self._timestamps = deque()

    def add_yawn(self, now):
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def rate_per_minute(self, now):
        self._prune(now)
        return len(self._timestamps) * (60.0 / self.window_seconds)