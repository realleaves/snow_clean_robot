"""Task queue bookkeeping on top of :class:`PriorityScheduler`."""
import math

from decision.cleaning_task import ALLOWED_STATUS, CleaningTask, TaskStatus
from decision.priority_scheduler import PriorityScheduler
from utils.errors import TaskError


class TaskManager:
    """Owns every task, enforces legal status changes and picks the next task."""

    def __init__(self, scheduler: PriorityScheduler):
        self.scheduler = scheduler
        self.tasks: dict[int, CleaningTask] = {}
        self.history: list[tuple[int, str, str]] = []

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self.tasks)

    def __contains__(self, task_id: int) -> bool:
        return task_id in self.tasks

    def get(self, task_id: int) -> CleaningTask:
        if task_id not in self.tasks:
            raise TaskError(f'unknown task {task_id}')
        return self.tasks[task_id]

    def by_status(self, *statuses) -> list[CleaningTask]:
        wanted = {TaskStatus(s).value for s in statuses}
        return [t for t in self.tasks.values() if t.status in wanted]

    def waiting(self) -> list[CleaningTask]:
        return self.by_status(TaskStatus.WAITING)

    def active(self) -> list[CleaningTask]:
        return [t for t in self.tasks.values() if not t.terminal]

    def open_task_ids(self) -> list[int]:
        return [t.task_id for t in self.tasks.values() if not t.terminal]

    # ------------------------------------------------------------------ mutation
    def add_task(self, task: CleaningTask, replace: bool = False) -> CleaningTask:
        if not isinstance(task, CleaningTask):
            raise TaskError('add_task expects a CleaningTask')
        if not math.isfinite(task.pollution_score) or not 0 <= task.pollution_score <= 1:
            raise TaskError(f'task {task.task_id} has an invalid pollution_score')
        if task.task_id in self.tasks and not replace:
            raise TaskError(f'duplicate task ID {task.task_id}')
        self.tasks[task.task_id] = task
        self.history.append((task.task_id, 'NEW', task.status))
        return task

    def find_by_target(self, x: float, y: float, tolerance_m: float = 1e-6):
        """Existing *open* task covering the same target, if any.

        ``FAILED`` and ``MANUAL_CHECK`` tasks are deliberately excluded so a
        retried observation creates a fresh, traceable task instead of silently
        reusing a closed one.
        """
        for task in self.tasks.values():
            if task.terminal:
                continue
            if math.hypot(task.target_x - x, task.target_y - y) <= tolerance_m:
                return task
        return None

    def set_status(self, task_id: int, status, force: bool = False) -> CleaningTask:
        task = self.get(task_id)
        previous = task.status
        task.set_status(status, force=force)
        self.history.append((task_id, previous, task.status))
        return task

    def update(self, task_id: int, **fields) -> CleaningTask:
        task = self.get(task_id)
        for key, value in fields.items():
            if not hasattr(task, key):
                raise TaskError(f'task has no field {key!r}')
            if key == 'status':
                self.set_status(task_id, value)
                continue
            setattr(task, key, value)
        return task

    def complete(self, task_id: int) -> CleaningTask:
        return self.set_status(task_id, TaskStatus.COMPLETED)

    def fail(self, task_id: int, reason: str | None = None) -> CleaningTask:
        task = self.set_status(task_id, TaskStatus.FAILED)
        if reason is not None:
            task.last_error = reason
        return task

    def cancel(self, task_id: int) -> CleaningTask:
        return self.set_status(task_id, TaskStatus.CANCELLED)

    def manual_check(self, task_id: int, reason: str | None = None) -> CleaningTask:
        task = self.set_status(task_id, TaskStatus.MANUAL_CHECK)
        if reason is not None:
            task.last_error = reason
        return task

    def retry(self, task_id: int) -> CleaningTask:
        """Return a failed/manual-check task to the waiting queue."""
        task = self.get(task_id)
        if task.terminal and task.status == TaskStatus.COMPLETED.value:
            raise TaskError('a completed task cannot be retried')
        return self.set_status(task_id, TaskStatus.WAITING)

    def canceled(self) -> list[CleaningTask]:
        return self.by_status(TaskStatus.CANCELLED)

    def failed(self) -> list[CleaningTask]:
        return self.by_status(TaskStatus.FAILED)

    def is_known_bad_target(self, x: float, y: float, tolerance_m: float = 0.3) -> bool:
        """True when an earlier task on this whole area already failed.

        Prevents the detector from endlessly re-creating a task for a stain the
        planner can never reach (V1.0 acceptance scenario G).
        """
        for task in self.tasks.values():
            if task.status not in (TaskStatus.FAILED.value, TaskStatus.MANUAL_CHECK.value):
                continue
            if math.hypot(task.target_x - x, task.target_y - y) <= tolerance_m:
                return True
        return False

    # ------------------------------------------------------------------ selection
    def get_next_task(self, pose, now: float | None = None) -> CleaningTask | None:
        waiting = self.waiting()
        if not waiting:
            return None
        return self.scheduler.next_task(waiting, pose, now=now)

    def ranked(self, pose, now: float | None = None) -> list[CleaningTask]:
        return self.scheduler.rank(self.waiting(), pose, now=now)

    def recalculate_all(self, pose, now: float | None = None) -> list[CleaningTask]:
        """Refresh priority for every waiting task (spec: priority recalculation)."""
        return self.scheduler.rank(self.waiting(), pose, now=now)

    def cancel_all(self) -> None:
        for task in list(self.tasks.values()):
            if not task.terminal:
                self.cancel(task.task_id)
