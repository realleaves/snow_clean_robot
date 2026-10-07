from dataclasses import dataclass, field
import time


@dataclass
class CleaningTask:
    task_id: int
    target_x: float
    target_y: float
    pollution_score: float
    pollution_level: str
    distance_score: float = 0.0
    wait_score: float = 0.0
    priority: float = 0.0
    retry_count: int = 0
    status: str = "WAITING"
    before_area: float | None = None
    after_area: float | None = None
    cleaning_efficiency: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    region_id: int | None = None
