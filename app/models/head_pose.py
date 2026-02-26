# app/models/distraction.py

import cv2
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (0.0, -63.6, -12.5),    # Chin
            (-43.3, 32.7, -26.0),   # Left eye
            (43.3, 32.7, -26.0),    # Right eye
            (-28.9, -28.9, -24.1),  # Left mouth
            (28.9, -28.9, -24.1)    # Right mouth
        ])

    def estimate(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        image_points = np.array([
            landmarks[1],    # Nose
            landmarks[199],  # Chin
            landmarks[33],   # Left eye
            landmarks[263],  # Right eye
            landmarks[61],   # Left mouth
            landmarks[291]   # Right mouth
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        rotation_matrix = rotation_matrix.T
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
            yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
            yaw = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
            roll = 0

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return pitch, yaw, roll