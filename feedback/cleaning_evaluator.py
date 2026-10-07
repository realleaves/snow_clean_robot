"""Post-cleaning evaluation: ``eta = (A_before - A_after) / A_before``."""
from dataclasses import dataclass

from utils.errors import ConfigError, TaskError


@dataclass
class Evaluation:
    efficiency: float | None
    passed: bool
    reason: str = ''


class CleaningEvaluator:
    def __init__(self, success_threshold: float):
        if not isinstance(success_threshold, (int, float)) or isinstance(success_threshold, bool):
            raise ConfigError('success threshold must be a number')
        if not 0 <= success_threshold <= 1:
            raise ConfigError('success threshold outside 0..1')
        self.success_threshold = float(success_threshold)

    def efficiency(self, before_area: float, after_area: float) -> float | None:
        """Relative area reduction, or ``None`` when the measurement is invalid."""
        if before_area is None or after_area is None:
            return None
        if before_area <= 0 or after_area < 0:
            return None
        return max(0.0, min(1.0, (before_area - after_area) / before_area))

    def evaluate(self, before_area: float, after_area: float) -> Evaluation:
        if before_area is None or before_area <= 0:
            return Evaluation(None, False, 'INVALID: before_area must be positive')
        if after_area is None or after_area < 0:
            return Evaluation(None, False, 'INVALID: after_area must be non-negative')
        eta = self.efficiency(before_area, after_area)
        passed = eta is not None and eta >= self.success_threshold
        return Evaluation(eta, passed, 'PASS' if passed else 'FAIL')

    def evaluate_task(self, task) -> Evaluation:
        """Evaluate a task and write the result back onto it."""
        if task.before_area is None:
            raise TaskError(f'task {task.task_id} has no before_area measurement')
        result = self.evaluate(task.before_area, task.after_area)
        task.cleaning_efficiency = result.efficiency
        return result
