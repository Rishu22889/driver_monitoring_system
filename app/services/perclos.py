from collections import deque


class PERCLOSDetector:
    def __init__(self, window_seconds=20, fps=30):
        self.window_size = int(window_seconds * fps)
        self.buffer = deque(maxlen=self.window_size)

    def update(self, ear, threshold):
        if threshold is None:
            return 0.0

        eye_closed = 1 if ear < threshold else 0
        self.buffer.append(eye_closed)

        if len(self.buffer) == 0:
            return 0.0

        return sum(self.buffer) / len(self.buffer)