import time


class DrowsinessStateMachine:
    """
    ALERT / WARNING / CRITICAL state machine with hysteresis.

    Transitions require the score to hold past a threshold for a
    continuous duration (not a single-frame crossing), so small
    fluctuations don't flicker the state. Defaults match the spec:
    WARNING needs score>=30 held 5s, CRITICAL needs score>=60 held
    3s (can jump straight there from ALERT), recovery to ALERT
    needs score<20 held 10s.
    """

    ALERT = "ALERT"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    def __init__(self,
                 warning_threshold=30, warning_hold=5.0,
                 critical_threshold=60, critical_hold=3.0,
                 recovery_threshold=20, recovery_hold=10.0):
        self.warning_threshold = warning_threshold
        self.warning_hold = warning_hold
        self.critical_threshold = critical_threshold
        self.critical_hold = critical_hold
        self.recovery_threshold = recovery_threshold
        self.recovery_hold = recovery_hold

        self.state = self.ALERT

        self._above_warning_since = None
        self._above_critical_since = None
        self._below_recovery_since = None

    def update(self, score, now=None):
        now = now if now is not None else time.time()

        if score >= self.warning_threshold:
            self._above_warning_since = self._above_warning_since or now
        else:
            self._above_warning_since = None

        if score >= self.critical_threshold:
            self._above_critical_since = self._above_critical_since or now
        else:
            self._above_critical_since = None

        if score < self.recovery_threshold:
            self._below_recovery_since = self._below_recovery_since or now
        else:
            self._below_recovery_since = None

        if (self._above_critical_since is not None and
                (now - self._above_critical_since) >= self.critical_hold):
            self.state = self.CRITICAL

        elif (self.state == self.ALERT and
              self._above_warning_since is not None and
              (now - self._above_warning_since) >= self.warning_hold):
            self.state = self.WARNING

        elif (self.state in (self.WARNING, self.CRITICAL) and
              self._below_recovery_since is not None and
              (now - self._below_recovery_since) >= self.recovery_hold):
            self.state = self.ALERT

        return self.state
