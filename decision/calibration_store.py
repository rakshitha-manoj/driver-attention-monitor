"""
calibration_store.py — Decision layer
Cross-Session Calibration Persistence: learns a driver's baseline over time.

Every calibration in existing drowsiness-detection systems resets every
session. A real product would learn a driver's baseline over days/weeks,
the way a fitness tracker learns your resting heart rate. This module
provides that capability.

How it works:
    1. On first launch: no history exists, system calibrates normally
       (10s EAR observation + 3s pose hold).
    2. On calibration complete: saves threshold/offsets to local JSON.
    3. On subsequent launches: loads history, computes a weighted average
       (recent sessions weighted higher via exponential decay), and uses
       this as the initial threshold — calibration still runs but starts
       from a better baseline and can be shorter.
    4. Rolling window: keeps last 10 sessions to prevent stale data from
       dominating.

Storage: ~/.driver_attention_monitor/calibration_history.json
    - Local-only, never uploaded
    - Contains only numerical calibration parameters, no images/video
    - Human-readable JSON for transparency
"""

import json
import os
import time
import math
from datetime import datetime


DEFAULT_STORE_DIR = os.path.join(os.path.expanduser("~"),
                                 ".driver_attention_monitor")
DEFAULT_STORE_FILE = "calibration_history.json"
MAX_SESSIONS = 10
SCHEMA_VERSION = 1


class CalibrationStore:
    """
    Persistent calibration storage across driving sessions.

    Maintains a rolling history of calibration results and provides
    cross-session averaged thresholds that improve over time.
    """

    def __init__(self, store_dir=None, max_sessions=MAX_SESSIONS):
        self.store_dir = store_dir or DEFAULT_STORE_DIR
        self.store_path = os.path.join(self.store_dir, DEFAULT_STORE_FILE)
        self.max_sessions = max_sessions
        self.history = []
        self._loaded = False

    def load(self):
        """Load calibration history from disk. Safe to call if file doesn't exist."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r") as f:
                    data = json.load(f)
                if data.get("version") == SCHEMA_VERSION:
                    self.history = data.get("sessions", [])
                    # Keep only last max_sessions
                    self.history = self.history[-self.max_sessions:]
                    self._loaded = True
                    print(f"Calibration history loaded: {len(self.history)} "
                          f"previous session(s)")
                else:
                    print("Calibration history version mismatch — starting fresh")
                    self.history = []
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Calibration history corrupt, starting fresh: {e}")
                self.history = []
        else:
            print("No calibration history found — first session")
        self._loaded = True

    def save(self):
        """Persist current history to disk."""
        os.makedirs(self.store_dir, exist_ok=True)
        data = {
            "version": SCHEMA_VERSION,
            "driver_id": "default",
            "last_updated": datetime.now().isoformat(),
            "sessions": self.history[-self.max_sessions:],
        }
        try:
            with open(self.store_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Calibration saved: {self.store_path}")
        except IOError as e:
            print(f"WARNING: Could not save calibration: {e}")

    def add_session(self, ear_threshold, ear_variance=0.0,
                    pose_offsets=None, gaze_baseline=None,
                    session_duration_sec=0, confidence=1.0):
        """
        Record the current session's calibration results.

        Args:
            ear_threshold: calibrated EAR blink threshold
            ear_variance: standard deviation during EAR calibration
            pose_offsets: dict with pitch, yaw, roll offsets (or None)
            gaze_baseline: dict with h, v neutral gaze values (or None)
            session_duration_sec: total session length in seconds
            confidence: overall calibration confidence (0-1)
        """
        session = {
            "timestamp": datetime.now().isoformat(),
            "ear_threshold": round(float(ear_threshold), 5),
            "ear_variance": round(float(ear_variance), 5),
            "pose_offsets": pose_offsets or {"pitch": 0, "yaw": 0, "roll": 0},
            "gaze_baseline": gaze_baseline or {"h": 0.5, "v": 0.5},
            "session_duration_sec": round(session_duration_sec, 1),
            "confidence": round(float(confidence), 3),
        }
        self.history.append(session)
        # Trim to max
        self.history = self.history[-self.max_sessions:]
        self.save()

    def get_personalized_ear_threshold(self, fallback=0.25):
        """
        Compute a cross-session weighted average EAR threshold.

        Uses exponential decay weighting: most recent sessions have
        higher influence. Sessions with high variance (noisy calibration)
        are down-weighted.

        Returns:
            float: averaged EAR threshold, or fallback if no history.
        """
        if not self.history:
            return fallback

        # Exponential decay: weight = exp(-decay * age_index)
        # age_index 0 = most recent, 1 = second most recent, etc.
        decay = 0.3
        n = len(self.history)

        weighted_sum = 0.0
        weight_total = 0.0

        for i, session in enumerate(reversed(self.history)):
            age_weight = math.exp(-decay * i)
            # Down-weight high-variance sessions
            variance = session.get("ear_variance", 0.01)
            variance_weight = 1.0 / (1.0 + variance * 10)
            # Confidence weight
            conf_weight = session.get("confidence", 1.0)

            w = age_weight * variance_weight * conf_weight
            weighted_sum += w * session["ear_threshold"]
            weight_total += w

        if weight_total <= 0:
            return fallback

        result = weighted_sum / weight_total
        return round(result, 5)

    def get_personalized_pose_offsets(self):
        """
        Compute cross-session averaged pose offsets.

        Returns:
            dict: {"pitch": float, "yaw": float, "roll": float} or None
        """
        if not self.history:
            return None

        decay = 0.3
        weighted = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        weight_total = 0.0

        for i, session in enumerate(reversed(self.history)):
            offsets = session.get("pose_offsets")
            if offsets is None:
                continue
            w = math.exp(-decay * i) * session.get("confidence", 1.0)
            for key in ("pitch", "yaw", "roll"):
                weighted[key] += w * offsets.get(key, 0.0)
            weight_total += w

        if weight_total <= 0:
            return None

        return {k: round(v / weight_total, 3) for k, v in weighted.items()}

    def get_personalized_gaze_baseline(self):
        """
        Compute cross-session averaged gaze neutral baseline.

        Returns:
            dict: {"h": float, "v": float} or None
        """
        if not self.history:
            return None

        decay = 0.3
        weighted = {"h": 0.0, "v": 0.0}
        weight_total = 0.0

        for i, session in enumerate(reversed(self.history)):
            baseline = session.get("gaze_baseline")
            if baseline is None:
                continue
            w = math.exp(-decay * i) * session.get("confidence", 1.0)
            for key in ("h", "v"):
                weighted[key] += w * baseline.get(key, 0.5)
            weight_total += w

        if weight_total <= 0:
            return None

        return {k: round(v / weight_total, 4) for k, v in weighted.items()}

    @property
    def has_history(self):
        """True if at least one previous session exists."""
        return len(self.history) > 0

    @property
    def session_count(self):
        """Number of stored sessions."""
        return len(self.history)

    def get_summary(self):
        """Human-readable summary of calibration history."""
        if not self.history:
            return "No calibration history — first session"

        n = len(self.history)
        avg_ear = self.get_personalized_ear_threshold()
        latest = self.history[-1]
        return (
            f"Calibration history: {n} session(s) | "
            f"Cross-session EAR threshold: {avg_ear:.4f} | "
            f"Latest: {latest['timestamp'][:10]}"
        )
