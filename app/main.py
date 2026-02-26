import cv2
import numpy as np
from app.models.face_detector import FaceDetector
from app.utils.geometry import calculate_ear, calculate_mar
from app.services.adaptive_ear import AdaptiveEAR
from app.services.perclos import PERCLOSDetector
from app.services.fatigue import FatigueScore
from app.models.emotion import EmotionInference
from app.models.head_pose import HeadPoseEstimator
from app.utils.distractions import DistractionDetector
from app.models.risk_model import BehaviorRiskModel
from app.utils.alert import AlertManager

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 291, 308, 14]

detector = FaceDetector()
adaptive_ear = AdaptiveEAR(calibration_frames=150)
perclos_detector = PERCLOSDetector(window_seconds=20, fps=30)
fatigue_scorer = FatigueScore()
emotion_model = EmotionInference("app/models/emotion_detector_best.pth")
head_pose = HeadPoseEstimator()
distraction = DistractionDetector()
risk_model = BehaviorRiskModel()
alert = AlertManager()


mar_threshold = 0.60
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    landmarks, bbox = detector.detect(frame)

    if landmarks is not None and bbox is not None:

        x_min, y_min, x_max, y_max = bbox
        h, w, _ = frame.shape
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        if x_max - x_min > 20 and y_max - y_min > 20:
            face_crop = frame[y_min:y_max, x_min:x_max]
            emotion,conf,emotion_risk = emotion_model.predict(face_crop)


            cv2.putText(frame,
                        f"{emotion} : {conf} : {emotion_risk}",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2)
    if landmarks is not None:

        left_eye = landmarks[LEFT_EYE]
        right_eye = landmarks[RIGHT_EYE]
        mouth = landmarks[MOUTH]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        ear = (left_ear + right_ear) / 2.0

        baseline_ear = adaptive_ear.update(ear)
        threshold = adaptive_ear.get_threshold()

        if baseline_ear is not None and baseline_ear > 0:
            ear_fatigue = max(min(1 - (ear / baseline_ear), 1.0), 0.0)
        else:
            ear_fatigue = 0.0

        mar = calculate_mar(mouth)
        mar_fatigue = min(mar / mar_threshold, 1.0)

        perclos_value = perclos_detector.update(ear, threshold)

        f_score = fatigue_scorer.compute(
            ear_fatigue,
            mar_fatigue,
            perclos_value
        )
        pitch,yaw,roll = head_pose.estimate(landmarks, frame.shape)
        if pitch is not None:
            distraction_score = distraction.check(pitch, yaw)
            if distraction_score>0.50:
                cv2.putText(frame, "DISTRACTED" , (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,0,255), 2)

        risk_level, risk_score = risk_model.compute(f_score,distraction_score,emotion_risk)

        if risk_level == "High":
             alert.send_alert("High Driver Risk Detected")

        cv2.putText(frame, f"Risk_level : {risk_level} , Score:{risk_score}",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)


        if threshold is None:
            cv2.putText(frame, "Calibrating...",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)
        else:
            cv2.putText(frame,
                        f"Fatigue_Score: {f_score:.2f}",
                        (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            if f_score > 0.7:
                cv2.putText(frame,
                            "HIGH FATIGUE!",
                            (200, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()