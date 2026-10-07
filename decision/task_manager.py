from decision.cleaning_task import CleaningTask
from decision.priority_scheduler import PriorityScheduler


class TaskManager:
    def __init__(self, scheduler: PriorityScheduler):
        self.scheduler = scheduler
        self.tasks: dict[int, CleaningTask] = {}

    def add_task(self, task: CleaningTask) -> None:
        if task.task_id in self.tasks:
            raise ValueError('duplicate task ID')
        self.tasks[task.task_id] = task

    def get_next_task(self, pose) -> CleaningTask | None:
        waiting = [task for task in self.tasks.values() if task.status == 'WAITING']
        for task in waiting:
            self.scheduler.calculate(task, pose)
        return max(waiting, key=lambda t: (t.priority, -t.task_id), default=None)

    def complete(self, task_id: int) -> None: self.tasks[task_id].status = 'COMPLETED'
    def fail(self, task_id: int) -> None: self.tasks[task_id].status = 'FAILED'
    def cancel(self, task_id: int) -> None: self.tasks[task_id].status = 'FAILED'
    def retry(self, task_id: int) -> None: self.tasks[task_id].status = 'WAITING'
