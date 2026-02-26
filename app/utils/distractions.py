import time
import numpy as np

class DistractionDetector:
    def __init__(self,
                 yaw_threshold=25,
                 pitch_threshold=20,
                 distraction_time=3.0,
                 alpha=0.6,
                 max_angle=60):

        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distraction_time = distraction_time
        self.max_angle = max_angle

        self.alpha = alpha
        self.smooth_yaw = 0
        self.smooth_pitch = 0

        self.start_time = None
        self.current_score = 0.0

    def check(self, pitch, yaw):
        # --- Smooth ---
        self.smooth_yaw = self.alpha * yaw + (1 - self.alpha) * self.smooth_yaw
        self.smooth_pitch = self.alpha * pitch + (1 - self.alpha) * self.smooth_pitch

        abs_yaw = abs(self.smooth_yaw)
        abs_pitch = abs(self.smooth_pitch)

        current_time = time.time()
        yaw_severity = 0
        pitch_severity = 0

        if abs_yaw > self.yaw_threshold:
            yaw_severity = min(
                1.0,
                (abs_yaw - self.yaw_threshold) / (self.max_angle - self.yaw_threshold)
            )

        if abs_pitch > self.pitch_threshold:
            pitch_severity = min(
                1.0,
                (abs_pitch - self.pitch_threshold) / (self.max_angle - self.pitch_threshold)
            )

        angle_severity = 0.7 * yaw_severity + 0.3 * pitch_severity
        if angle_severity > 0:
            if self.start_time is None:
                self.start_time = current_time

            elapsed = current_time - self.start_time
            duration_ratio = min(1.0, elapsed / self.distraction_time)

            self.current_score = 0.5 * duration_ratio + 0.5 * angle_severity

        else:
            self.start_time = None
            self.current_score *= 0.85

        self.current_score = np.clip(self.current_score, 0, 1)

        return self.current_score