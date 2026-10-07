from dataclasses import dataclass
import math
import time


@dataclass
class PollutionTarget:
    id: int
    x: float
    y: float
    score: float
    level: str
    status: str
    last_seen: float


class PollutionMap:
    def __init__(self, merge_distance_m: float):
        self.merge_distance_m = merge_distance_m
        self.targets: dict[int, PollutionTarget] = {}
        self.next_id = 1

    def upsert(self, x: float, y: float, score: float, level: str) -> PollutionTarget:
        for target in self.targets.values():
            if target.status != 'COMPLETED' and math.hypot(target.x - x, target.y - y) < self.merge_distance_m:
                target.x, target.y, target.score, target.level = x, y, score, level
                target.last_seen = time.monotonic()
                return target
        target = PollutionTarget(self.next_id, x, y, score, level, 'ACTIVE', time.monotonic())
        self.targets[target.id] = target
        self.next_id += 1
        return target

    def complete(self, target_id: int) -> None:
        self.targets[target_id].status = 'COMPLETED'

    def remove(self, target_id: int) -> None:
        del self.targets[target_id]
