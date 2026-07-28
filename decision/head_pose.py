import cv2
import numpy as np

from config import (
    MODEL_POINTS,
    NOSE,
    CHIN,
    LEFT_EYE,
    RIGHT_EYE,
    LEFT_MOUTH,
    RIGHT_MOUTH
)


class HeadPoseEstimator:
    """
    Estimates head pose (Pitch, Yaw, Roll)
    using MediaPipe facial landmarks and OpenCV solvePnP.
    """

    def __init__(self):
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self.last_rvec = None
        self.last_tvec = None

    def reset_tracking(self):
        self.last_rvec = None
        self.last_tvec = None

    def get_image_points(self, landmarks, width, height):
        image_points = np.array([
            (landmarks[NOSE].x * width, landmarks[NOSE].y * height),
            (landmarks[CHIN].x * width, landmarks[CHIN].y * height),
            (landmarks[LEFT_EYE].x * width, landmarks[LEFT_EYE].y * height),
            (landmarks[RIGHT_EYE].x * width, landmarks[RIGHT_EYE].y * height),
            (landmarks[LEFT_MOUTH].x * width, landmarks[LEFT_MOUTH].y * height),
            (landmarks[RIGHT_MOUTH].x * width, landmarks[RIGHT_MOUTH].y * height),
        ], dtype=np.float64)
        return image_points



    def get_camera_matrix(self, width, height):
        focal_length = width
        camera_matrix = np.array([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        return camera_matrix

    @staticmethod
    def _wrap_180(angle):
        """Wrap any angle into (-180, 180]. Previously only pitch was
        wrapped -- yaw/roll left unwrapped is what produced the -300+
        degree readings."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def rotation_matrix_to_euler(self, rotation_matrix):
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = self._wrap_180(float(angles[0]))
        yaw = self._wrap_180(float(angles[1]))
        roll = self._wrap_180(float(angles[2]))

        return pitch, yaw, roll

    def estimate_pose(self, landmarks, frame):
        height, width = frame.shape[:2]

        image_points = self.get_image_points(landmarks, width, height)
        camera_matrix = self.get_camera_matrix(width, height)

        # Use previous pose as initial guess to prevent 180-degree flip local minima
        use_guess = (self.last_rvec is not None and self.last_tvec is not None)
        rvec_guess = self.last_rvec.copy() if use_guess else np.zeros((3, 1), dtype=np.float64)
        tvec_guess = self.last_tvec.copy() if use_guess else np.zeros((3, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            camera_matrix,
            self.dist_coeffs,
            rvec=rvec_guess,
            tvec=tvec_guess,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            self.last_rvec = None
            self.last_tvec = None
            return None

        self.last_rvec = rotation_vector
        self.last_tvec = translation_vector

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self.rotation_matrix_to_euler(rotation_matrix)

        return {
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2),
            "roll": round(roll, 2),
            "rotation_vector": rotation_vector,
            "translation_vector": translation_vector,
            "rotation_matrix": rotation_matrix,
        }
