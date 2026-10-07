def next_action(task, max_retry: int) -> str:
    """Increment retry count only when a compensation pass can be scheduled."""
    if task.retry_count >= max_retry:
        task.status = 'MANUAL_CHECK'
        return 'MANUAL_CHECK'
    task.retry_count += 1
    task.status = 'WAITING'
    return 'COMPENSATE'
