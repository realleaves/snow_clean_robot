"""Bounded compensation: at most ``max_retry`` extra cleaning passes per task.

Sequence for ``max_retry = 2`` (see V1.0 acceptance plan section 25)::

    FAIL #1 -> retry_count = 1 -> COMPENSATE
    FAIL #2 -> retry_count = 2 -> COMPENSATE
    FAIL #3 ->                    MANUAL_CHECK   (no further retry)
"""
from utils.errors import ConfigError, TaskError

COMPENSATE = 'COMPENSATE'
MANUAL_CHECK = 'MANUAL_CHECK'


def validate_max_retry(max_retry: int) -> int:
    if isinstance(max_retry, bool) or not isinstance(max_retry, int) or max_retry < 0:
        raise ConfigError('max_retry must be a non-negative integer')
    return max_retry


def can_compensate(task, max_retry: int) -> bool:
    return task.retry_count < validate_max_retry(max_retry)


def next_action(task, max_retry: int) -> str:
    """Decide the next action after a failed recheck and update the task."""
    validate_max_retry(max_retry)
    if task is None:
        raise TaskError('next_action requires a task')
    if not can_compensate(task, max_retry):
        task.status = MANUAL_CHECK
        return MANUAL_CHECK
    task.retry_count += 1
    task.status = 'WAITING'
    return COMPENSATE


def compensation_budget(max_retry: int) -> int:
    return validate_max_retry(max_retry)
