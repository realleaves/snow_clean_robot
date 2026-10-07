"""Three-level pollution classification.

V1.0 rules::

    S <  light_max          -> LIGHT
    light_max <= S < medium_max -> MEDIUM
    S >= medium_max         -> HEAVY
"""
from enum import Enum

from utils.errors import ConfigError, PerceptionError


class PollutionLevel(Enum):
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


class PollutionClassifier:
    def __init__(self, light_max: float = 0.35, medium_max: float = 0.70):
        if not 0 < light_max < medium_max <= 1:
            raise ConfigError('level thresholds must satisfy 0 < light_max < medium_max <= 1')
        self.light_max, self.medium_max = float(light_max), float(medium_max)

    def classify(self, score: float) -> PollutionLevel:
        if score is None or not isinstance(score, (int, float)) or isinstance(score, bool):
            raise PerceptionError('pollution score must be a number')
        if not 0 <= score <= 1:
            raise PerceptionError(f'pollution score {score} outside 0..1')
        if score < self.light_max:
            return PollutionLevel.LIGHT
        if score < self.medium_max:
            return PollutionLevel.MEDIUM
        return PollutionLevel.HEAVY

    def classify_name(self, score: float) -> str:
        return self.classify(score).name

    def classify_region(self, region) -> str:
        """Classify a region's ``pollution_score`` in place and return the name."""
        if region.pollution_score is None:
            raise PerceptionError('region has no pollution_score to classify')
        name = self.classify_name(region.pollution_score)
        region.pollution_level = name
        return name
