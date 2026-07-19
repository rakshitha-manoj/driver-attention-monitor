import time


class NodDetector:
    """
    Detects a 'nod' event: calibrated pitch dropping past
    `drop_threshold` and recovering to near-neutral within
    `recovery_window` seconds -- but only when the head is roughly
    forward-facing, and not more often than `cooldown_seconds`.

    Two guards added after real-world testing showed turn/tilt
    motion could otherwise inflate the count:
    - max_concurrent_yaw/roll: a genuine drowsy nod happens with the
      head facing forward. Large yaw/roll alongside a pitch dip
      means the dip came from turning/tilting, not nodding.
    - cooldown_seconds: minimum time between two counted nods, so
      landmark jitter oscillating across the threshold during a
      single motion can't register as several nods.
    """

    def __init__(self, drop_threshold=20.0, recover_threshold=5.0,
                 recovery_window=2.0, direction=1, max_plausible_pitch=70.0,
                 max_concurrent_yaw=20.0, max_concurrent_roll=20.0,
                 cooldown_seconds=1.0):
        self.drop_threshold = drop_threshold
        self.recover_threshold = recover_threshold
        self.recovery_window = recovery_window
        self.direction = direction
        self.max_plausible_pitch = max_plausible_pitch
        self.max_concurrent_yaw = max_concurrent_yaw
        self.max_concurrent_roll = max_concurrent_roll
        self.cooldown_seconds = cooldown_seconds

        self.nod_count = 0
        self._dip_start_time = None
        self._in_dip = False
        self._last_nod_time = None

    def update(self, pitch, yaw=0.0, roll=0.0, now=None):
        now = now if now is not None else time.time()

        if abs(pitch) > self.max_plausible_pitch:
            return self.nod_count

        forward_facing = (abs(yaw) <= self.max_concurrent_yaw and
                           abs(roll) <= self.max_concurrent_roll)

        signed_pitch = pitch * self.direction

        if not self._in_dip:
            if signed_pitch <= -self.drop_threshold and forward_facing:
                self._in_dip = True
                self._dip_start_time = now
        else:
            elapsed = now - self._dip_start_time

            if signed_pitch >= -self.recover_threshold:
                in_cooldown = (
                    self._last_nod_time is not None and
                    (now - self._last_nod_time) < self.cooldown_seconds
                )

                if elapsed <= self.recovery_window and forward_facing and not in_cooldown:
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
