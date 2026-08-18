"""
Camera-angle offset correction.

Separate from PoseCalibrator: calibration removes the driver's own
resting head orientation. This removes the systematic bias the
camera itself introduces when it's not mounted frontally.

Workflow (needs your 0/15/25 degree recordings, which don't exist
yet -- this is the scaffold to run once you have them):

1. Record short clips at each known camera angle with the driver
   looking straight ahead at the road (not at the camera).
2. For each clip, average the RAW pitch/yaw (before calibration)
   over the clip -- that average is the offset that angle alone
   introduces.
3. Fit a linear correction across the measured angles so unseen
   angles (e.g. your own rig's actual mount angle) can be corrected
   too, not just the three you recorded.
"""

import numpy as np


def measure_offset_from_samples(pitch_samples, yaw_samples):
    """Call once per recorded clip to get that angle's offset."""
    return {
        "pitch_offset": float(np.mean(pitch_samples)),
        "yaw_offset": float(np.mean(yaw_samples)),
    }


class CameraAngleCorrector:
    def __init__(self, angle_offsets=None):
        # angle_offsets: {camera_angle_degrees: {"pitch_offset": .., "yaw_offset": ..}}
        self.angle_offsets = angle_offsets or {}
        self._pitch_fit = None
        self._yaw_fit = None

    def add_measurement(self, camera_angle_degrees, pitch_samples, yaw_samples):
        self.angle_offsets[camera_angle_degrees] = measure_offset_from_samples(
            pitch_samples, yaw_samples
        )

    def fit_linear(self):
        if len(self.angle_offsets) < 2:
            raise ValueError("Need at least 2 recorded angles to fit a line.")

        angles = sorted(self.angle_offsets.keys())
        pitch_vals = [self.angle_offsets[a]["pitch_offset"] for a in angles]
        yaw_vals = [self.angle_offsets[a]["yaw_offset"] for a in angles]

        self._pitch_fit = np.polyfit(angles, pitch_vals, 1)
        self._yaw_fit = np.polyfit(angles, yaw_vals, 1)

    def correct(self, pitch, yaw, camera_angle):
        if self._pitch_fit is None:
            raise RuntimeError("Call fit_linear() after adding measurements first.")

        pitch_offset = np.polyval(self._pitch_fit, camera_angle)
        yaw_offset = np.polyval(self._yaw_fit, camera_angle)
        return pitch - pitch_offset, yaw - yaw_offset
