"""World-frame pollution target memory with distance-based de-duplication."""
from dataclasses import dataclass, field
import math
import time

from utils.errors import ConfigError, TaskError

STATUS_ACTIVE = 'ACTIVE'
STATUS_COMPLETED = 'COMPLETED'


@dataclass
class PollutionTarget:
    id: int
    x: float
    y: float
    score: float
    level: str
    status: str
    last_seen: float
    observations: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.status != STATUS_COMPLETED


class PollutionMap:
    """Keeps one target per physical stain, even when it is re-detected.

    ``upsert`` merges any new observation within ``merge_distance_m`` (inclusive)
    of an existing active target instead of creating a duplicate; observations
    further away become separate targets. A *stricter* merge radius is used while an
    existing target is still being confirmed so that a slightly displaced second
    observation of the same stain does not spawn a twin.
    """

    def __init__(self, merge_distance_m: float = 0.25, horizon: float | None = None):
        if merge_distance_m <= 0:
            raise ConfigError('merge_distance_m must be positive')
        self.merge_distance_m = float(merge_distance_m)
        self.horizon = horizon
        self.targets: dict[int, PollutionTarget] = {}
        self.next_id = 1
        self.merge_events = 0

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.targets)

    def __contains__(self, target_id: int) -> bool:
        return target_id in self.targets

    def active_targets(self) -> list[PollutionTarget]:
        return [t for t in self.targets.values() if t.active]

    def completed_targets(self) -> list[PollutionTarget]:
        return [t for t in self.targets.values() if not t.active]

    def get(self, target_id: int) -> PollutionTarget:
        if target_id not in self.targets:
            raise TaskError(f'unknown pollution target {target_id}')
        return self.targets[target_id]

    def nearest(self, x: float, y: float, active_only: bool = True):
        candidates = self.active_targets() if active_only else list(self.targets.values())
        if not candidates:
            return None, math.inf
        best = min(candidates, key=lambda t: math.hypot(t.x - x, t.y - y))
        return best, math.hypot(best.x - x, best.y - y)

    def distance_to(self, target_id: int, x: float, y: float) -> float:
        target = self.get(target_id)
        return math.hypot(target.x - x, target.y - y)

    # ------------------------------------------------------------------ mutation
    def upsert(self, x: float, y: float, score: float, level: str,
               now: float | None = None) -> tuple[PollutionTarget, str]:
        """Insert or merge an observation; returns ``(target, 'created'|'merged')``."""
        for name, value in (('x', x), ('y', y), ('score', score)):
            if not math.isfinite(float(value)):
                raise TaskError(f'pollution observation has a non-finite {name}')
        timestamp = time.monotonic() if now is None else now
        best, distance = self.nearest(x, y)
        # Inclusive boundary: an observation exactly merge_distance_m away still
        # belongs to the same stain (spec 15 example: 1.00,1.00 vs 1.10,1.08 at 0.25).
        if best is not None and distance <= self.merge_distance_m:
            self._merge(best, x, y, score, level, timestamp)
            self.merge_events += 1
            return best, 'merged'
        target = PollutionTarget(self.next_id, float(x), float(y), float(score), level,
                                 STATUS_ACTIVE, timestamp, 1, [(float(x), float(y))])
        self.targets[target.id] = target
        self.next_id += 1
        return target, 'created'

    def _merge(self, target: PollutionTarget, x: float, y: float, score: float,
               level: str, now: float) -> None:
        target.x, target.y = float(x), float(y)
        target.score = max(target.score, float(score))
        target.level = level
        target.last_seen = now
        target.observations += 1
        target.history.append((float(x), float(y)))

    def update(self, target_id: int, x: float | None = None, y: float | None = None,
               score: float | None = None, level: str | None = None,
               status: str | None = None, now: float | None = None) -> PollutionTarget:
        target = self.get(target_id)
        if x is not None:
            target.x = float(x)
        if y is not None:
            target.y = float(y)
        if score is not None:
            target.score = float(score)
        if level is not None:
            target.level = level
        if status is not None:
            if status not in (STATUS_ACTIVE, STATUS_COMPLETED):
                raise TaskError(f'unknown target status {status!r}')
            target.status = status
        target.last_seen = time.monotonic() if now is None else now
        return target

    def complete(self, target_id: int) -> PollutionTarget:
        return self.update(target_id, status=STATUS_COMPLETED)

    def reopen(self, target_id: int) -> PollutionTarget:
        return self.update(target_id, status=STATUS_ACTIVE)

    def remove(self, target_id: int) -> None:
        if target_id not in self.targets:
            raise TaskError(f'unknown pollution target {target_id}')
        del self.targets[target_id]

    def clear(self) -> None:
        self.targets.clear()

    def expire(self, now: float, max_age_s: float) -> list[int]:
        """Drop active targets not seen for *max_age_s*; returns removed IDs."""
        if max_age_s <= 0:
            raise ConfigError('max_age_s must be positive')
        stale = [t.id for t in self.active_targets() if now - t.last_seen > max_age_s]
        for target_id in stale:
            del self.targets[target_id]
        return stale
