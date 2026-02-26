import subprocess
import time

class AlertManager:
    def __init__(self, cooldown=10):
        self.cooldown = cooldown
        self.last_alert_time = 0

    def send_alert(self, message):

        current_time = time.time()

        # Prevent spamming alerts
        if current_time - self.last_alert_time < self.cooldown:
            return

        self.last_alert_time = current_time

        subprocess.run([
            "osascript",
            "-e",
            f'display notification "{message}" with title "Driver Monitoring System"'
        ])