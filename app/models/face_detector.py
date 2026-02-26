# app/models/face_detector.py

import cv2
import mediapipe as mp
import numpy as np


class FaceDetector():
    def __init__(self, static_mode = False, max_faces=1, min_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode = static_mode,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
    
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None, None

        face_landmarks = results.multi_face_landmarks[0]

        h, w, _ = frame.shape
        landmarks = []

        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))

        landmarks = np.array(landmarks)

        x_min = np.min(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        x_max = np.max(landmarks[:, 0])
        y_max = np.max(landmarks[:, 1])

        bbox = (x_min, y_min, x_max, y_max)

        return landmarks, bbox

    