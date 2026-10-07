"""Multimodal pollution score ``S = wv*V + wh*H + wa*A``."""
import math

from utils.errors import ConfigError, PerceptionError

KEYS = ('visual', 'humidity', 'area')


class PollutionScorer:
    def __init__(self, cfg: dict):
        missing = [f'{key}_weight' for key in KEYS if f'{key}_weight' not in cfg]
        missing += [key for key in ('max_area_ratio',) if key not in cfg]
        if missing:
            raise ConfigError(f'fusion config is missing keys: {", ".join(missing)}')
        self.cfg = dict(cfg)
        weights = [float(cfg[f'{key}_weight']) for key in KEYS]
        if any(weight < 0 for weight in weights):
            raise ConfigError('fusion weights must be nonnegative')
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ConfigError(
                f'fusion weights must sum to one, got {sum(weights):.6f}')
        if cfg['max_area_ratio'] <= 0:
            raise ConfigError('max_area_ratio must be positive')

    @property
    def weights(self) -> dict[str, float]:
        return {key: float(self.cfg[f'{key}_weight']) for key in KEYS}

    def score(self, visual: float, humidity: float, area: float) -> float:
        for name, value in (('visual', visual), ('humidity', humidity), ('area', area)):
            if value is None or not isinstance(value, (int, float)) or isinstance(value, bool):
                raise PerceptionError(f'fusion input {name} must be a number')
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise PerceptionError(f'fusion input {name}={value} must be within 0..1')
        return sum(self.cfg[f'{key}_weight'] * value for key, value
                   in (('visual', visual), ('humidity', humidity), ('area', area)))

    def score_region(self, region, humidity: float) -> float:
        """Score a :class:`~perception.pollution_region.PollutionRegion` in place."""
        value = self.score(region.visual_score, humidity, region.area_score)
        region.pollution_score = value
        return value
