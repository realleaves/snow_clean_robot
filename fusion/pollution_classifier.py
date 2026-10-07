from enum import Enum


class PollutionLevel(Enum):
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


class PollutionClassifier:
    def __init__(self, light_max: float, medium_max: float):
        if not 0 < light_max < medium_max < 1:
            raise ValueError("level thresholds must satisfy 0 < light < medium < 1")
        self.light_max, self.medium_max = light_max, medium_max

    def classify(self, score: float) -> PollutionLevel:
        if not 0 <= score <= 1:
            raise ValueError("pollution score outside 0..1")
        if score < self.light_max:
            return PollutionLevel.LIGHT
        if score < self.medium_max:
            return PollutionLevel.MEDIUM
        return PollutionLevel.HEAVY
