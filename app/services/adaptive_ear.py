import numpy as np


class AdaptiveEAR:
    def __init__(self, calibration_frames=150):
        self.calibration_frames = calibration_frames
        self.ear_values = []
        self.baseline = None

    def update(self, ear):
        if self.baseline is None:
            self.ear_values.append(ear)

            if len(self.ear_values) >= self.calibration_frames:
                self.baseline = np.mean(self.ear_values)

        return self.baseline

    def get_threshold(self):
        if self.baseline is None:
            return None
            
        return self.baseline * 0.75