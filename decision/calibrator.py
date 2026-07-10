class PoseCalibrator:
    """
    Calibrates the driver's neutral head pose.

    During the first few frames, the driver should
    look straight ahead.

    Later, all angles become relative to that pose.
    """

    def __init__(self):

        self.pitch_offset = None
        self.yaw_offset = None
        self.roll_offset = None

        self.is_calibrated = False

    def calibrate(self, pose):

        if self.is_calibrated:
            return

        self.pitch_offset = pose["pitch"]
        self.yaw_offset = pose["yaw"]
        self.roll_offset = pose["roll"]

        self.is_calibrated = True

        print("Calibration Complete")

    def get_relative_pose(self, pose):

        if not self.is_calibrated:
            return pose

        return {

            "pitch": pose["pitch"] - self.pitch_offset,

            "yaw": pose["yaw"] - self.yaw_offset,

            "roll": pose["roll"] - self.roll_offset

        }