from fastapi import FastAPI
import cv2
import time
import threading
import subprocess

from app.models.face_detector import FaceDetector
from app.utils.geometry import calculate_ear, calculate_mar
from app.services.adaptive_ear import AdaptiveEAR
from app.services.perclos import PERCLOSDetector
from app.services.fatigue import FatigueScore
from app.models.emotion import EmotionInference
from app.models.head_pose import HeadPoseEstimator
from app.utils.distractions import DistractionDetector
from app.models.risk_model import BehaviorRiskModel

app = FastAPI()

detector = FaceDetector()
adaptive_ear = AdaptiveEAR(calibration_frames=150)
perclos = PERCLOSDetector(window_seconds=20, fps=30)
fatigue = FatigueScore()
emotion_model = EmotionInference("app/models/emotion_detector_best.pth")
head_pose = HeadPoseEstimator()
distraction = DistractionDetector()
risk_model = BehaviorRiskModel()

camera_running = False
camera_thread = None
alert_active = False
latest_data = {
    "fatigue": 0.0,
    "distraction": 0.0,
    "emotion": 0.0,
    "risk_score": 0.0,
    "risk_level": "Low"
}

state_lock = threading.Lock()


def play_alert():
    try:
        subprocess.Popen(["afplay", "data/assets/alert.mp3"])
    except Exception as e:
        print("Alert sound error:", e)


def camera_loop():
    global camera_running, alert_active, latest_data

    cap = cv2.VideoCapture(0)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 81, 13, 291, 308, 14]
    mar_threshold = 0.60

    while camera_running:

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        landmarks, bbox = detector.detect(frame)

        emotion_risk = 0.0
        distraction_score = 0.0
        f_score = 0.0
        risk_score = 0.0
        risk_level = "Low"

        if landmarks is not None and bbox is not None:

            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = bbox

            # SAFE CLIP
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # SAFE FACE CROP
            if x_max - x_min > 20 and y_max - y_min > 20:
                face_crop = frame[y_min:y_max, x_min:x_max]
                if face_crop.size > 0:
                    try:
                        _, _, emotion_risk = emotion_model.predict(face_crop)
                    except Exception as e:
                        print("Emotion error:", e)

            # ---- Fatigue ----
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            mouth = landmarks[MOUTH]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            baseline = adaptive_ear.update(ear)
            threshold = adaptive_ear.get_threshold()

            ear_fatigue = (
                max(min(1 - (ear / baseline), 1.0), 0.0)
                if baseline else 0.0
            )

            mar = calculate_mar(mouth)
            mar_fatigue = min(mar / mar_threshold, 1.0)

            perclos_value = perclos.update(ear, threshold)

            f_score = fatigue.compute(
                ear_fatigue,
                mar_fatigue,
                perclos_value
            )

            # ---- Distraction ----
            pitch, yaw, roll = head_pose.estimate(landmarks, frame.shape)

            if pitch is not None:
                distraction_score = distraction.check(pitch, yaw)

            # ---- Risk Model ----
            risk_level, risk_score = risk_model.compute(
                f_score,
                distraction_score,
                emotion_risk
            )

            # ---- Alert Logic ----
            if risk_level == "High":
                if not alert_active:
                    play_alert()
                    alert_active = True
            else:
                alert_active = False

            # ---- Store Latest Data ----
            with state_lock:
                latest_data["fatigue"] = float(f_score)
                latest_data["distraction"] = float(distraction_score)
                latest_data["emotion"] = float(emotion_risk)
                latest_data["risk_score"] = float(risk_score)
                latest_data["risk_level"] = risk_level

        time.sleep(0.03)

    cap.release()


@app.post("/start")
def start_camera():
    global camera_running, camera_thread

    if not camera_running:
        camera_running = True
        camera_thread = threading.Thread(
            target=camera_loop,
            daemon=True
        )
        camera_thread.start()

    return {"status": "camera_started"}


@app.post("/stop")
def stop_camera():
    global camera_running
    camera_running = False
    return {"status": "camera_stopped"}


@app.get("/latest")
def latest_metrics():
    with state_lock:
        return [[
            latest_data["fatigue"],
            latest_data["distraction"],
            latest_data["emotion"],
            latest_data["risk_score"],
            latest_data["risk_level"]
        ]]