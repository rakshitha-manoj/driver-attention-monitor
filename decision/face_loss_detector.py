import time


class FaceLossDetector:
    """
    Tracks how long the driver's face has gone completely
    undetected. Extreme downward droop, full micro-sleep collapse,
    or the camera being blocked all take the face out of MediaPipe's
    detectable range entirely -- which the rest of this pipeline
    currently ignores rather than flags.

    Prolonged face loss while driving is a danger signal on its own,
    separate from pitch/yaw/PERCLOS (which all require a detected
    face to compute anything). This fires independently of the
    scoring/state-machine path, so it isn't diluted by score
    hysteresis -- if the face has been gone this long, that alone
    is worth an alert, regardless of what the score was doing
    right before it disappeared.
    """

    def __init__(self, loss_threshold_seconds=3.0):
        self.loss_threshold_seconds = loss_threshold_seconds
        self._lost_since = None
        self.face_lost_alert = False

    def update(self, face_detected, now=None):
        now = now if now is not None else time.time()

        if face_detected:
            self._lost_since = None
            self.face_lost_alert = False
        else:
            if self._lost_since is None:
                self._lost_since = now
            elif (now - self._lost_since) >= self.loss_threshold_seconds:
                self.face_lost_alert = True

        return self.face_lost_alert
