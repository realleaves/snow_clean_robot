"""反馈层验收测试：:mod:`feedback.compensation`（有界补偿）。

验证 ``max_retry = 2`` 的三段式序列、绝不越过重试上限（不会无限循环）、
辅助函数与配置校验。
"""
import unittest

import pytest

from decision.cleaning_task import CleaningTask
from feedback.compensation import (COMPENSATE, MANUAL_CHECK, can_compensate,
                                   compensation_budget, next_action,
                                   validate_max_retry)
from tests.helpers import CLEANING_CFG
from utils.errors import ConfigError, TaskError

MAX_RETRY = 2


def make_task(task_id=1, status='RECHECK', retry_count=0):
    return CleaningTask(task_id, 0.0, 0.0, 0.5, 'MEDIUM',
                        status=status, retry_count=retry_count)


class CompensationSequenceTests(unittest.TestCase):
    def setUp(self):
        self.max_retry = int(dict(CLEANING_CFG)['max_retry'])
        self.assertEqual(self.max_retry, MAX_RETRY)
        self.task = make_task()

    def test_failure_sequence_for_max_retry_two(self):
        first = next_action(self.task, MAX_RETRY)
        self.assertEqual(first, COMPENSATE)
        self.assertEqual(self.task.retry_count, 1)
        self.assertEqual(self.task.status, 'WAITING')

        second = next_action(self.task, MAX_RETRY)
        self.assertEqual(second, COMPENSATE)
        self.assertEqual(self.task.retry_count, 2)
        self.assertEqual(self.task.status, 'WAITING')

        third = next_action(self.task, MAX_RETRY)
        self.assertEqual(third, MANUAL_CHECK)
        self.assertEqual(self.task.status, MANUAL_CHECK)
        self.assertEqual(self.task.retry_count, MAX_RETRY)

    def test_never_increments_past_max_retry(self):
        actions = [next_action(self.task, MAX_RETRY) for _ in range(10)]
        self.assertEqual(actions[:2], [COMPENSATE, COMPENSATE])
        self.assertEqual(actions[2:], [MANUAL_CHECK] * 8)
        self.assertEqual(self.task.retry_count, MAX_RETRY)
        self.assertEqual(self.task.status, MANUAL_CHECK)
        self.assertLessEqual(self.task.retry_count, MAX_RETRY)

    def test_zero_max_retry_goes_straight_to_manual_check(self):
        task = make_task(retry_count=0)
        self.assertEqual(next_action(task, 0), MANUAL_CHECK)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.status, MANUAL_CHECK)

    def test_high_max_retry_keeps_compensating(self):
        task = make_task(retry_count=0)
        for expected in (1, 2, 3):
            self.assertEqual(next_action(task, 3), COMPENSATE)
            self.assertEqual(task.retry_count, expected)
        self.assertEqual(next_action(task, 3), MANUAL_CHECK)
        self.assertEqual(task.retry_count, 3)


class CompensationHelperTests(unittest.TestCase):
    def test_can_compensate_boundaries(self):
        self.assertTrue(can_compensate(make_task(retry_count=0), MAX_RETRY))
        self.assertTrue(can_compensate(make_task(retry_count=1), MAX_RETRY))
        self.assertFalse(can_compensate(make_task(retry_count=2), MAX_RETRY))
        self.assertFalse(can_compensate(make_task(retry_count=5), MAX_RETRY))

    def test_compensation_budget_returns_max_retry(self):
        self.assertEqual(compensation_budget(0), 0)
        self.assertEqual(compensation_budget(MAX_RETRY), MAX_RETRY)
        self.assertEqual(compensation_budget(7), 7)

    def test_validate_max_retry_accepts_non_negative_ints(self):
        self.assertEqual(validate_max_retry(0), 0)
        self.assertEqual(validate_max_retry(1), 1)
        self.assertEqual(validate_max_retry(MAX_RETRY), MAX_RETRY)

    def test_invalid_max_retry_raises_config_error(self):
        for bad in (-1, -5, 1.5, '2', None, True, False):
            with pytest.raises(ConfigError):
                validate_max_retry(bad)
            with pytest.raises(ConfigError):
                compensation_budget(bad)
            with pytest.raises(ConfigError):
                can_compensate(make_task(), bad)

    def test_next_action_with_invalid_retry_budget_raises_config_error(self):
        with pytest.raises(ConfigError):
            next_action(make_task(), -1)

    def test_next_action_without_task_raises_task_error(self):
        with pytest.raises(TaskError):
            next_action(None, MAX_RETRY)

    def test_constants_match_task_status_values(self):
        self.assertEqual(COMPENSATE, 'COMPENSATE')
        self.assertEqual(MANUAL_CHECK, 'MANUAL_CHECK')


if __name__ == '__main__':
    unittest.main()
