"""Task lifecycle and legal status transitions."""
from dataclasses import dataclass, field
from enum import Enum
import time

from utils.errors import TaskError


class TaskStatus(str, Enum):
    WAITING = 'WAITING'
    PLANNING = 'PLANNING'
    NAVIGATING = 'NAVIGATING'
    CLEANING = 'CLEANING'
    RECHECK = 'RECHECK'
    COMPENSATE = 'COMPENSATE'
    COMPLETED = 'COMPLETED'
    FAILED = 'FAILED'
    MANUAL_CHECK = 'MANUAL_CHECK'
    CANCELLED = 'CANCELLED'


#: Legal status transitions; anything else raises :class:`TaskError`.
ALLOWED_STATUS = {
    TaskStatus.WAITING: {TaskStatus.PLANNING, TaskStatus.WAITING, TaskStatus.FAILED,
                         TaskStatus.MANUAL_CHECK, TaskStatus.CANCELLED},
    TaskStatus.PLANNING: {TaskStatus.NAVIGATING, TaskStatus.WAITING, TaskStatus.FAILED,
                          TaskStatus.CANCELLED},
    TaskStatus.NAVIGATING: {TaskStatus.CLEANING, TaskStatus.FAILED, TaskStatus.MANUAL_CHECK,
                            TaskStatus.CANCELLED},
    TaskStatus.CLEANING: {TaskStatus.RECHECK, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.RECHECK: {TaskStatus.COMPENSATE, TaskStatus.COMPLETED, TaskStatus.FAILED,
                         TaskStatus.MANUAL_CHECK, TaskStatus.CANCELLED},
    TaskStatus.COMPENSATE: {TaskStatus.CLEANING, TaskStatus.WAITING, TaskStatus.FAILED,
                            TaskStatus.MANUAL_CHECK, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: {TaskStatus.WAITING, TaskStatus.CANCELLED},
    TaskStatus.MANUAL_CHECK: {TaskStatus.WAITING, TaskStatus.CANCELLED},
    TaskStatus.CANCELLED: set(),
}

TERMINAL_STATUS = (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.MANUAL_CHECK,
                   TaskStatus.FAILED)


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
    status: str = 'WAITING'
    before_area: float | None = None
    after_area: float | None = None
    cleaning_efficiency: float | None = None
    created_at: float = field(default_factory=time.monotonic)
    region_id: int | None = None
    passes_done: int = 0
    last_error: str | None = None

    # ------------------------------------------------------------------ status
    @property
    def task_status(self) -> TaskStatus:
        return TaskStatus(self.status)

    def can_transition(self, status) -> bool:
        target = TaskStatus(status)
        return target in ALLOWED_STATUS[self.task_status]

    def set_status(self, status, force: bool = False) -> None:
        target = TaskStatus(status)
        if not force and target not in ALLOWED_STATUS[self.task_status]:
            raise TaskError(f'illegal task status transition {self.status} -> {target.value}')
        self.status = target.value

    @property
    def terminal(self) -> bool:
        return self.task_status in TERMINAL_STATUS

    @property
    def succeeded(self) -> bool:
        return self.task_status is TaskStatus.COMPLETED

    # ------------------------------------------------------------------ metrics
    def record_cleaning(self, passes: int) -> None:
        self.passes_done += int(passes)

    def record_recheck(self, before_area: float | None, after_area: float | None,
                       efficiency: float | None) -> None:
        if before_area is not None:
            self.before_area = before_area
        self.after_area = after_area
        self.cleaning_efficiency = efficiency

    def age_s(self, now: float | None = None) -> float:
        reference = time.monotonic() if now is None else now
        return max(0.0, reference - self.created_at)
