from dataclasses import dataclass


@dataclass
class Evaluation:
    efficiency: float | None
    passed: bool


class CleaningEvaluator:
    def __init__(self, success_threshold: float):
        if not 0 <= success_threshold <= 1:
            raise ValueError('success threshold outside 0..1')
        self.success_threshold = success_threshold

    def evaluate(self, before_area: float, after_area: float) -> Evaluation:
        if before_area <= 0 or after_area < 0:
            return Evaluation(None, False)
        efficiency = max(0.0, min(1.0, (before_area - after_area) / before_area))
        return Evaluation(efficiency, efficiency >= self.success_threshold)
