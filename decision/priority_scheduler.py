import math
import time


class PriorityScheduler:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        if any(cfg[f'{k}_weight'] < 0 for k in ('pollution', 'distance', 'waiting')):
            raise ValueError('priority weights must be nonnegative')
        if cfg['max_distance_m'] <= 0 or cfg['starvation_seconds'] <= 0:
            raise ValueError('priority scales must be positive')

    def calculate(self, task, pose, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        distance = math.hypot(task.target_x - pose.x, task.target_y - pose.y)
        task.distance_score = min(distance / self.cfg['max_distance_m'], 1.0)
        elapsed = max(0.0, now - task.created_at)
        task.wait_score = min(elapsed / self.cfg['starvation_seconds'], 1.0)
        # Unbounded aging guarantees a waiting task eventually outranks newer ones.
        task.priority = (self.cfg['pollution_weight'] * task.pollution_score
                         - self.cfg['distance_weight'] * task.distance_score
                         + self.cfg['waiting_weight'] * task.wait_score
                         + elapsed / self.cfg['starvation_seconds'])
        return task.priority
