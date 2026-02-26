class FatigueScore:
    def __init__(self):
        pass

    def compute(self,ear_fatigue,mar_fatigue,perclos):
        score = (
            0.5 * perclos +
            0.3 * ear_fatigue +
            0.2 * mar_fatigue
        )

        return min(score, 1.0)