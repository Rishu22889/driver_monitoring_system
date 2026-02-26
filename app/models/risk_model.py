import numpy as np

class BehaviorRiskModel:
    def __init__(self, alpha=0.2):
        self.w_drowsy = 0.5
        self.w_distraction = 0.3
        self.w_emotion = 0.2

        self.alpha = alpha
        self.smoothed_risk = 0.0

    def nonlinear_scale(self, x):
        return 1 / (1 + np.exp(-5 * (x - 0.5)))

    def interaction_boost(self, d, dis):
        return 0.2 * d * dis

    def compute(self, drowsy, distraction, emotion):

        d = self.nonlinear_scale(drowsy)
        dis = self.nonlinear_scale(distraction)
        emo = self.nonlinear_scale(emotion)

        base_risk = (
            self.w_drowsy * d +
            self.w_distraction * dis +
            self.w_emotion * emo
        )

        boosted_risk = base_risk + self.interaction_boost(d, dis)

        boosted_risk = np.clip(boosted_risk, 0, 1)

        self.smoothed_risk = (
            self.alpha * boosted_risk +
            (1 - self.alpha) * self.smoothed_risk
        )

        if self.smoothed_risk < 0.30:
            level = "Low"
        elif self.smoothed_risk < 0.45:
            level = "Medium"
        else:
            level = "High"

        return level, self.smoothed_risk