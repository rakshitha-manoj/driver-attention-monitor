from collections import deque


class NodRateWindow:
    """
    Rolling window of recent nod timestamps -- same fix as
    yawn_rate.py, applied to nods. nod_detector.nod_count is
    lifetime-cumulative, which is fine for scoring's saturating
    normalization, but wrong for "is this driver nodding off right
    now" -- a burst of nods in a short window matters, a lifetime
    total doesn't.
    """

    def __init__(self, window_seconds=60.0):
        self.window_seconds = window_seconds
        self._timestamps = deque()

    def add_nod(self, now):
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def count_recent(self, now):
        self._prune(now)
        return len(self._timestamps)
