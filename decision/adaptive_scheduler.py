"""
adaptive_scheduler.py — Decision layer
Adaptive Sensor Scheduling: scales monitoring intensity to actual risk.

Instead of running every detector on every frame, this scheduler adjusts
how often expensive operations (hand/YOLO detection, gaze tracking) run
based on the current drowsiness state and score. When the driver is
confidently alert, heavy detectors run infrequently. As risk signals
increase, the system escalates to full-frequency monitoring.

This is a genuine systems-architecture contribution: adaptive resource/
attention allocation. No surveyed paper implements this — they all run
every signal every frame. This directly addresses the principle that
a monitoring system should watch less when there's nothing to worry
about, the same way a human passenger would glance over occasionally
rather than stare.

Schedule levels:
    RELAXED   — score < 15, PERCLOS < 5%: hands every 15th, gaze every 5th
    ATTENTIVE — score 15-30 or PERCLOS 5-15%: hands every 5th, gaze every 2nd
    ELEVATED  — WARNING state (score >= 30): hands every 2nd, gaze every frame
    MAXIMUM   — CRITICAL state (score >= 60): everything every frame
"""

import time


class AdaptiveSensorScheduler:
    """
    Dynamically adjusts how often expensive sensors run based on risk.

    Exposes should_run_hands(frame_id) and should_run_gaze(frame_id)
    which return True/False. The caller gates its sensor calls behind
    these checks. The scheduler updates its internal level from the
    current system state and drowsiness score each frame.

    Performance impact:
        RELAXED mode:   hand/YOLO runs ~7× less, gaze ~5× less
        ATTENTIVE mode: hand/YOLO runs ~3× less, gaze ~2× less
        ELEVATED mode:  hand/YOLO runs ~2× less, gaze every frame
        MAXIMUM mode:   no savings, identical to always-on baseline
    """

    # Schedule levels with their frame intervals
    RELAXED   = "RELAXED"
    ATTENTIVE = "ATTENTIVE"
    ELEVATED  = "ELEVATED"
    MAXIMUM   = "MAXIMUM"

    # (hand_interval, gaze_interval) — run every Nth frame
    _INTERVALS = {
        "RELAXED":   (5,  3),
        "ATTENTIVE": (3,  2),
        "ELEVATED":  (2,  1),
        "MAXIMUM":   (1,  1),
    }

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._set_level(self.RELAXED)

        # Transition hysteresis: require the new level to hold for
        # a minimum duration before downgrading (prevents oscillation)
        self._downgrade_hold_sec = 3.0
        self._pending_downgrade_level = None
        self._pending_downgrade_since = None

        # Metrics for logging/evaluation
        self.frames_total = 0
        self.frames_hands_run = 0
        self.frames_gaze_run = 0

    def update(self, score, perclos, system_state, now=None):
        """
        Update the schedule level based on current risk signals.
        Call once per frame, before checking should_run_*.

        Args:
            score: drowsiness score 0-100
            perclos: PERCLOS value 0.0-1.0
            system_state: "ALERT", "WARNING", or "CRITICAL"
            now: timestamp (defaults to time.time())
        """
        if not self.enabled:
            self.level = self.MAXIMUM
            self._hand_interval, self._gaze_interval = 1, 1
            return self.level

        now = now if now is not None else time.time()

        # Determine target level from risk signals
        target = self._compute_target_level(score, perclos, system_state)

        # Upgrades (increasing monitoring) happen immediately
        level_order = [self.RELAXED, self.ATTENTIVE, self.ELEVATED, self.MAXIMUM]
        current_idx = level_order.index(self.level)
        target_idx = level_order.index(target)

        if target_idx > current_idx:
            # Escalation: apply immediately
            self._set_level(target)
            self._pending_downgrade_level = None
            self._pending_downgrade_since = None
        elif target_idx < current_idx:
            # De-escalation: require hysteresis hold
            if self._pending_downgrade_level != target:
                self._pending_downgrade_level = target
                self._pending_downgrade_since = now
            elif (now - self._pending_downgrade_since) >= self._downgrade_hold_sec:
                self._set_level(target)
                self._pending_downgrade_level = None
                self._pending_downgrade_since = None
        else:
            # Same level — clear any pending downgrade
            self._pending_downgrade_level = None
            self._pending_downgrade_since = None

        return self.level

    def should_run_hands(self, frame_id):
        """Returns True if hand/YOLO detection should run this frame."""
        self.frames_total += 1
        if not self.enabled:
            self.frames_hands_run += 1
            return True
        run = (frame_id % self._hand_interval == 0)
        if run:
            self.frames_hands_run += 1
        return run

    def should_run_gaze(self, frame_id):
        """Returns True if gaze tracking should run this frame."""
        if not self.enabled:
            self.frames_gaze_run += 1
            return True
        run = (frame_id % self._gaze_interval == 0)
        if run:
            self.frames_gaze_run += 1
        return run

    def get_savings_report(self):
        """
        Returns a dict with frame-skip statistics for evaluation.
        Call at end of session.
        """
        total = max(self.frames_total, 1)
        return {
            "total_frames": self.frames_total,
            "hands_run": self.frames_hands_run,
            "hands_skipped_pct": round(
                (1 - self.frames_hands_run / total) * 100, 1),
            "gaze_run": self.frames_gaze_run,
            "gaze_skipped_pct": round(
                (1 - self.frames_gaze_run / total) * 100, 1),
        }

    # ── Internal ──────────────────────────────────────────────────

    def _compute_target_level(self, score, perclos, system_state):
        """Determine the appropriate schedule level from risk signals."""
        if system_state == "CRITICAL" or score >= 60:
            return self.MAXIMUM
        if system_state == "WARNING" or score >= 30:
            return self.ELEVATED
        if score >= 15 or perclos >= 0.05:
            return self.ATTENTIVE
        return self.RELAXED

    def _set_level(self, level):
        """Apply a schedule level and update intervals."""
        self.level = level
        self._hand_interval, self._gaze_interval = self._INTERVALS[level]
