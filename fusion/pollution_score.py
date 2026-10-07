import math


class PollutionScorer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        weights = [cfg[f'{key}_weight'] for key in ('visual', 'humidity', 'area')]
        if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
            raise ValueError("fusion weights must be nonnegative and sum to one")
        if cfg['max_area_ratio'] <= 0:
            raise ValueError("max_area_ratio must be positive")

    def score(self, visual: float, humidity: float, area: float) -> float:
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (visual, humidity, area)):
            raise ValueError("fusion inputs must be finite values in 0..1")
        return sum(self.cfg[f'{k}_weight'] * v for k, v in
                   (('visual', visual), ('humidity', humidity), ('area', area)))
