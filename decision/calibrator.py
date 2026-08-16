import time


class PoseCalibrator:
    """
    Calibrates the driver's neutral head pose.

    Instead of taking the offset from a single frame (which
    inherits any jitter/blink/twitch in that exact instant), this
    collects samples over `calibration_duration` seconds and sets
    the offset from their mean -- same approach as Hafsa's EAR
    calibration.

    Public interface is unchanged: calibrate(pose), is_calibrated,
    get_relative_pose(pose). demo.py does not need to change.
    """

    def __init__(self, calibration_duration=3.0):

        self.pitch_offset = None
        self.yaw_offset = None
        self.roll_offset = None

        self.is_calibrated = False

        self.calibration_duration = calibration_duration
        self._start_time = None
        self._samples = {"pitch": [], "yaw": [], "roll": []}

    def calibrate(self, pose):

        if self.is_calibrated:
            return

        if self._start_time is None:
            self._start_time = time.time()
            print(
                f"Calibrating... hold still and look straight ahead "
                f"for {self.calibration_duration:.0f}s"
            )

        self._samples["pitch"].append(pose["pitch"])
        self._samples["yaw"].append(pose["yaw"])
        self._samples["roll"].append(pose["roll"])

        elapsed = time.time() - self._start_time

        if elapsed >= self.calibration_duration:

            n = len(self._samples["pitch"])

            self.pitch_offset = sum(self._samples["pitch"]) / n
            self.yaw_offset = sum(self._samples["yaw"]) / n
            self.roll_offset = sum(self._samples["roll"]) / n

            self.is_calibrated = True

            print(
                "Calibration Complete "
                f"(pitch={self.pitch_offset:.2f}, "
                f"yaw={self.yaw_offset:.2f}, "
                f"roll={self.roll_offset:.2f}, n={n} samples)"
            )

    def get_relative_pose(self, pose):

        if not self.is_calibrated:
            return pose

        return {
            "pitch": pose["pitch"] - self.pitch_offset,
            "yaw": pose["yaw"] - self.yaw_offset,
            "roll": pose["roll"] - self.roll_offset,
        }

    def export_offsets(self):
        """
        Export calibration offsets for cross-session persistence.
        Returns dict with pitch/yaw/roll offsets, or None if not calibrated.
        """
        if not self.is_calibrated:
            return None
        return {
            "pitch": round(float(self.pitch_offset), 3),
            "yaw": round(float(self.yaw_offset), 3),
            "roll": round(float(self.roll_offset), 3),
        }

    def import_offsets(self, data):
        """
        Import calibration offsets from a previous session.
        The imported offsets serve as a warm start; live calibration
        will refine them.

        Args:
            data: dict with pitch, yaw, roll offset keys
        """
        if data is None:
            return
        if all(k in data for k in ("pitch", "yaw", "roll")):
            self.pitch_offset = float(data["pitch"])
            self.yaw_offset = float(data["yaw"])
            self.roll_offset = float(data["roll"])
            print(f"  Loaded historical pose offsets: "
                  f"pitch={self.pitch_offset:.2f}, "
                  f"yaw={self.yaw_offset:.2f}, "
                  f"roll={self.roll_offset:.2f}")

