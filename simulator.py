import math
import random
from .models import Telemetry, utc_now

class SensorSimulator:
    def __init__(self):
        self.tick = 0
        self.mode = "normal"
        self.fan_on = False

    def set_mode(self, mode: str):
        self.mode = mode

    def next(self):
        self.tick += 1
        if self.mode == "sensor_dropout" and self.tick % 4 != 0:
            return None
        temp = 27 + math.sin(self.tick / 8) * 1.1 + random.uniform(-0.35, 0.35)
        vibration = 0.55 + random.uniform(-0.12, 0.12)
        humidity = 50 + math.sin(self.tick / 9) * 4 + random.uniform(-1.5, 1.5)
        if self.mode == "temperature_fault":
            temp = 39.5 + math.sin(self.tick / 3) + random.uniform(-0.4, 0.4)
        elif self.mode == "vibration_fault":
            vibration = 3.0 + random.uniform(-0.3, 0.3)
        elif self.mode == "sensor_dropout":
            temp = 26 + random.uniform(-0.5, 0.5)
        if self.fan_on and self.mode == "normal":
            temp -= 1.5
        return Telemetry(timestamp=utc_now(), temperature_c=round(temp, 2),
                         humidity_pct=round(max(0, min(100, humidity)), 2),
                         vibration_g=round(max(0, vibration), 2),
                         fan_on=self.fan_on, source="virtual-sensor")
