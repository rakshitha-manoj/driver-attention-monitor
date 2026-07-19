import cv2
import mediapipe as mp

from config import LANDMARKS
from head_pose import HeadPoseEstimator
from calibrator import PoseCalibrator


# =====================================
# Initialize MediaPipe Face Mesh
# =====================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================================
# Initialize Head Pose Estimator
# =====================================

pose_estimator = HeadPoseEstimator()

calibrator = PoseCalibrator()

# =====================================
# Open Webcam
# =====================================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror the image
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run MediaPipe
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ---------------------------------
            # Draw complete face mesh
            # ---------------------------------

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
            )

            # ---------------------------------
            # Draw the six important landmarks
            # ---------------------------------

            h, w, _ = frame.shape

            for name, index in LANDMARKS.items():

                landmark = face_landmarks.landmark[index]

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Red circle
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1
                )

                # Landmark label
                cv2.putText(
                    frame,
                    name,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )

            # ---------------------------------
            # Estimate Head Pose
            # ---------------------------------

            # ---------------------------------
# Estimate Head Pose
# ---------------------------------

            pose = pose_estimator.estimate_pose(
                face_landmarks.landmark,
                frame
            )

            # -----------------------------
            # Calibrate once
            # -----------------------------
            if pose is not None:

                if not calibrator.is_calibrated:
                    calibrator.calibrate(pose)

                pose = calibrator.get_relative_pose(pose)

                # Display angles
                angles = [
                    ("Pitch", pose["pitch"]),
                    ("Yaw", pose["yaw"]),
                    ("Roll", pose["roll"])
                ]

                y_pos = 40

                for label, value in angles:

                    cv2.putText(
                        frame,
                        f"{label}: {value:.2f}",
                        (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                    y_pos += 30
                

    # ---------------------------------
    # Display Webcam
    # ---------------------------------

    cv2.imshow("Head Pose Estimation", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()