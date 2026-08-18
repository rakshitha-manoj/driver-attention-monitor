from collections import deque


class PerclosWindow:
    """
    Rolling, time-based PERCLOS window: the fraction of the window
    duration where eyes were classified closed (from Hafsa's
    blink_state / EAR output).

    Time-weighted rather than frame-counted, so it's robust to
    frame-rate variation. Add a second instance with
    window_seconds=5.0 later (Week 5 plan) to catch microsleeps
    the 60s window would dilute -- the class already supports it.
    """

    def __init__(self, window_seconds=60.0):
        self.window_seconds = window_seconds
        self._samples = deque()  # (timestamp, is_closed)

    def update(self, is_closed, now):
        self._samples.append((now, is_closed))
        self._prune(now)
        return self.perclos()

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def perclos(self):
        if len(self._samples) < 2:
            return 0.0

        closed_duration = 0.0
        total_duration = self._samples[-1][0] - self._samples[0][0]

        for i in range(1, len(self._samples)):
            t_prev, closed_prev = self._samples[i - 1]
            t_curr, _ = self._samples[i]
            if closed_prev:
                closed_duration += (t_curr - t_prev)

        if total_duration <= 0:
            return 0.0

        return closed_duration / total_duration
