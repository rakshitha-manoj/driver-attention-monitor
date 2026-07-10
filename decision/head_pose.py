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

        # Camera distortion (assume no distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # ----------------------------------------------------
    # Convert MediaPipe landmarks into 2D image points
    # ----------------------------------------------------

    def get_image_points(self, landmarks, width, height):

        image_points = np.array([

            (
                landmarks[NOSE].x * width,
                landmarks[NOSE].y * height
            ),

            (
                landmarks[CHIN].x * width,
                landmarks[CHIN].y * height
            ),

            (
                landmarks[LEFT_EYE].x * width,
                landmarks[LEFT_EYE].y * height
            ),

            (
                landmarks[RIGHT_EYE].x * width,
                landmarks[RIGHT_EYE].y * height
            ),

            (
                landmarks[LEFT_MOUTH].x * width,
                landmarks[LEFT_MOUTH].y * height
            ),

            (
                landmarks[RIGHT_MOUTH].x * width,
                landmarks[RIGHT_MOUTH].y * height
            )

        ], dtype=np.float64)

        return image_points

    # ----------------------------------------------------
    # Build Camera Matrix
    # ----------------------------------------------------

    def get_camera_matrix(self, width, height):

        focal_length = width

        camera_matrix = np.array([

            [focal_length, 0, width / 2],

            [0, focal_length, height / 2],

            [0, 0, 1]

        ], dtype=np.float64)

        return camera_matrix

    # ----------------------------------------------------
    # Convert Rotation Matrix -> Euler Angles
    # ----------------------------------------------------

    def rotation_matrix_to_euler(self, rotation_matrix):

        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Normalize pitch for easier interpretation
        if pitch > 180:
            pitch -= 360

        if pitch < -180:
            pitch += 360

        return pitch, yaw, roll

    # ----------------------------------------------------
    # Estimate Head Pose
    # ----------------------------------------------------

    def estimate_pose(self, landmarks, frame):

        height, width = frame.shape[:2]

        image_points = self.get_image_points(
            landmarks,
            width,
            height
        )

        camera_matrix = self.get_camera_matrix(
            width,
            height
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(

            MODEL_POINTS,

            image_points,

            camera_matrix,

            self.dist_coeffs,

            flags=cv2.SOLVEPNP_ITERATIVE

        )

        if not success:

            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        pitch, yaw, roll = self.rotation_matrix_to_euler(
            rotation_matrix
        )

        return {

            "pitch": round(pitch, 2),

            "yaw": round(yaw, 2),

            "roll": round(roll, 2),

            "rotation_vector": rotation_vector,

            "translation_vector": translation_vector,

            "rotation_matrix": rotation_matrix

        }