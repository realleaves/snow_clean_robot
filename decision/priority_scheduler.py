"""Priority scheduling: ``P = ws*S - wd*D + wt*T`` with bounded aging.

* ``S`` pollution score in ``[0, 1]``
* ``D`` normalized distance in ``[0, 1]`` (``min(distance / max_distance_m, 1)``)
* ``T`` normalized waiting time in ``[0, 1]`` (``min(age / starvation_seconds, 1)``)

V1.0 keeps exactly the documented formula. Anti-starvation is provided by the
waiting term plus an additional *bounded* aging bonus of one extra wait unit
(``wt * T``), so a long-waiting light task eventually overtakes a freshly
detected heavy one without the priority diverging to infinity.
"""
import math
import time

from utils.errors import ConfigError


class PriorityScheduler:
    def __init__(self, cfg: dict):
        missing = [key for key in ('pollution_weight', 'distance_weight', 'waiting_weight',
                                   'max_distance_m', 'starvation_seconds') if key not in cfg]
        if missing:
            raise ConfigError(f'priority config is missing keys: {", ".join(missing)}')
        self.cfg = dict(cfg)
        if any(cfg[f'{key}_weight'] < 0 for key in ('pollution', 'distance', 'waiting')):
            raise ConfigError('priority weights must be nonnegative')
        if cfg['max_distance_m'] <= 0 or cfg['starvation_seconds'] <= 0:
            raise ConfigError('priority scales must be positive')
        if cfg['pollution_weight'] <= 0:
            raise ConfigError('pollution_weight must be positive to rank pollution at all')

    # ------------------------------------------------------------------ scoring
    @property
    def weights(self) -> dict[str, float]:
        return {key: float(self.cfg[f'{key}_weight'])
                for key in ('pollution', 'distance', 'waiting')}

    def distance_score(self, task, pose) -> float:
        distance = math.hypot(task.target_x - pose.x, task.target_y - pose.y)
        return min(distance / self.cfg['max_distance_m'], 1.0)

    def wait_score(self, task, now: float) -> float:
        return min(task.age_s(now) / self.cfg['starvation_seconds'], 1.0)

    def calculate(self, task, pose, now: float | None = None) -> float:
        reference = time.monotonic() if now is None else now
        task.distance_score = self.distance_score(task, pose)
        task.wait_score = self.wait_score(task, reference)
        waiting_weight = self.cfg['waiting_weight']
        task.priority = (self.cfg['pollution_weight'] * task.pollution_score
                         - self.cfg['distance_weight'] * task.distance_score
                         + waiting_weight * task.wait_score
                         + waiting_weight * task.wait_score)  # bounded aging bonus
        return task.priority

    def rank(self, tasks, pose, now: float | None = None) -> list:
        """Highest priority first.

        Ties are broken deterministically. Newer tasks win ties (their zero-based
        arrival order is smaller) so an already-waiting task is never overtaken by a
        fresh detection of the same priority.
        """
        reference = time.monotonic() if now is None else now
        ordered = sorted(tasks, key=lambda t: t.created_at)
        arrival = {id(task): index for index, task in enumerate(ordered)}
        for task in tasks:
            self.calculate(task, pose, now=reference)
        return sorted(tasks, key=lambda t: (-t.priority, arrival[id(t)]))

    def next_task(self, tasks, pose, now: float | None = None):
        ranked = self.rank(tasks, pose, now=now)
        return ranked[0] if ranked else None

    def is_starving(self, task, now: float | None = None) -> bool:
        reference = time.monotonic() if now is None else now
        return task.age_s(reference) >= self.cfg['starvation_seconds']
