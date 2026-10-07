"""决策层验收测试：:mod:`decision.cleaning_task`。

覆盖任务构造默认值、完整的状态迁移矩阵（合法迁移成功 / 非法迁移抛
``TaskError``）、状态辅助属性、``set_status(force=True)`` 越权开关以及
清洗/复检计数字段。
"""
import unittest

import pytest

from decision.cleaning_task import (ALLOWED_STATUS, TERMINAL_STATUS, CleaningTask,
                                    TaskStatus)
from utils.errors import TaskError

#: V1.0 验收计划要求的完整状态清单。
REQUIRED_STATUSES = ('WAITING', 'PLANNING', 'NAVIGATING', 'CLEANING', 'RECHECK',
                     'COMPENSATE', 'COMPLETED', 'FAILED', 'MANUAL_CHECK', 'CANCELLED')


def make_task(task_id=1, x=0.0, y=0.0, score=0.5, level='MEDIUM',
              status='WAITING', created_at=100.0, **kwargs):
    return CleaningTask(task_id, x, y, score, level, status=status,
                        created_at=created_at, **kwargs)


class CleaningTaskConstructionTests(unittest.TestCase):
    def test_construction_defaults(self):
        task = make_task()
        self.assertEqual(task.status, 'WAITING')
        self.assertIs(task.task_status, TaskStatus.WAITING)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(task.passes_done, 0)
        self.assertIsNone(task.before_area)
        self.assertIsNone(task.after_area)
        self.assertIsNone(task.cleaning_efficiency)
        self.assertEqual(task.created_at, 100.0)
        self.assertIsNone(task.region_id)
        self.assertEqual(task.distance_score, 0.0)
        self.assertEqual(task.wait_score, 0.0)
        self.assertEqual(task.priority, 0.0)
        self.assertIsNone(task.last_error)

    def test_created_at_defaults_to_monotonic_timestamp(self):
        task = CleaningTask(1, 0.0, 0.0, 0.5, 'MEDIUM')
        self.assertIsInstance(task.created_at, float)
        self.assertGreater(task.created_at, 0.0)
        self.assertGreaterEqual(task.age_s(), 0.0)

    def test_required_status_list_is_complete(self):
        self.assertEqual([s.value for s in TaskStatus], list(REQUIRED_STATUSES))
        self.assertEqual(set(ALLOWED_STATUS), set(TaskStatus))

    def test_terminal_status_tuple(self):
        self.assertEqual(set(TERMINAL_STATUS),
                         {TaskStatus.COMPLETED, TaskStatus.CANCELLED,
                          TaskStatus.MANUAL_CHECK, TaskStatus.FAILED})


class CleaningTaskTransitionTests(unittest.TestCase):
    def test_every_legal_transition_succeeds(self):
        checked = 0
        for source in TaskStatus:
            for target in ALLOWED_STATUS[source]:
                task = make_task(status=source.value)
                self.assertTrue(task.can_transition(target),
                                f'{source.value} -> {target.value} should be legal')
                task.set_status(target)
                self.assertEqual(task.status, target.value)
                checked += 1
        self.assertGreater(checked, 0)

    def test_every_illegal_transition_raises_and_keeps_status(self):
        checked = 0
        for source in TaskStatus:
            for target in TaskStatus:
                if target in ALLOWED_STATUS[source]:
                    continue
                task = make_task(status=source.value)
                self.assertFalse(task.can_transition(target),
                                 f'{source.value} -> {target.value} should be illegal')
                with pytest.raises(TaskError):
                    task.set_status(target)
                self.assertEqual(task.status, source.value)
                checked += 1
        self.assertGreater(checked, 0)

    def test_legal_transition_accepts_enum_and_string(self):
        task = make_task()
        task.set_status(TaskStatus.PLANNING)
        self.assertEqual(task.status, 'PLANNING')
        task.set_status('NAVIGATING')
        self.assertEqual(task.status, 'NAVIGATING')
        task.set_status(TaskStatus.CLEANING)
        task.set_status('RECHECK')
        task.set_status('COMPLETED')
        self.assertTrue(task.terminal)

    def test_terminal_tasks_have_no_outgoing_transition(self):
        for status in ('COMPLETED', 'CANCELLED'):
            self.assertEqual(ALLOWED_STATUS[TaskStatus(status)], set())
            task = make_task(status=status)
            for target in TaskStatus:
                with pytest.raises(TaskError):
                    task.set_status(target)

    def test_failed_and_manual_check_can_return_to_waiting(self):
        for status in ('FAILED', 'MANUAL_CHECK'):
            task = make_task(status=status)
            self.assertTrue(task.terminal)
            task.set_status(TaskStatus.WAITING)
            self.assertEqual(task.status, 'WAITING')
            self.assertFalse(task.terminal)


class CleaningTaskHelperTests(unittest.TestCase):
    def test_terminal_and_succeeded_helpers(self):
        waiting = make_task(status='WAITING')
        self.assertFalse(waiting.terminal)
        self.assertFalse(waiting.succeeded)
        for status in ('FAILED', 'MANUAL_CHECK', 'CANCELLED'):
            task = make_task(status=status)
            self.assertTrue(task.terminal, status)
            self.assertFalse(task.succeeded, status)
        completed = make_task(status='COMPLETED')
        self.assertTrue(completed.terminal)
        self.assertTrue(completed.succeeded)

    def test_can_transition_boundaries(self):
        task = make_task(status='WAITING')
        self.assertTrue(task.can_transition('PLANNING'))
        self.assertTrue(task.can_transition(TaskStatus.WAITING))
        self.assertFalse(task.can_transition('COMPLETED'))
        self.assertFalse(task.can_transition('CLEANING'))

    def test_unknown_status_string_raises_value_error(self):
        task = make_task()
        with pytest.raises(ValueError):
            task.set_status('NOT_A_STATUS')
        with pytest.raises(ValueError):
            task.can_transition('BOGUS')
        with pytest.raises(ValueError):
            TaskStatus('BOGUS')
        task.status = 'BOGUS'
        with pytest.raises(ValueError):
            task.task_status

    def test_force_bypasses_transition_guard(self):
        completed = make_task(status='COMPLETED')
        with pytest.raises(TaskError):
            completed.set_status('WAITING')
        completed.set_status('WAITING', force=True)
        self.assertEqual(completed.status, 'WAITING')

        canceled = make_task(status='CANCELLED')
        with pytest.raises(TaskError):
            canceled.set_status(TaskStatus.CLEANING)
        canceled.set_status(TaskStatus.CLEANING, force=True)
        self.assertEqual(canceled.status, 'CLEANING')

    def test_force_accepts_string_and_enum(self):
        task = make_task(status='COMPLETED')
        task.set_status('FAILED', force=True)
        self.assertEqual(task.status, 'FAILED')
        task.set_status(TaskStatus.PLANNING, force=True)
        self.assertEqual(task.status, 'PLANNING')


class CleaningTaskBookkeepingTests(unittest.TestCase):
    def test_record_cleaning_accumulates_passes(self):
        task = make_task()
        self.assertEqual(task.passes_done, 0)
        task.record_cleaning(1)
        self.assertEqual(task.passes_done, 1)
        task.record_cleaning(2)
        self.assertEqual(task.passes_done, 3)
        task.record_cleaning(0)
        self.assertEqual(task.passes_done, 3)

    def test_record_recheck_stores_measurements(self):
        task = make_task()
        task.record_recheck(1000.0, 200.0, 0.8)
        self.assertEqual(task.before_area, 1000.0)
        self.assertEqual(task.after_area, 200.0)
        self.assertAlmostEqual(task.cleaning_efficiency, 0.8)
        task.record_recheck(500.0, 100.0, 0.8)
        self.assertEqual(task.before_area, 500.0)
        self.assertEqual(task.after_area, 100.0)

    def test_record_recheck_keeps_before_area_when_none(self):
        task = make_task()
        task.record_recheck(1000.0, 200.0, 0.8)
        task.record_recheck(None, 100.0, 0.9)
        self.assertEqual(task.before_area, 1000.0)
        self.assertEqual(task.after_area, 100.0)
        self.assertAlmostEqual(task.cleaning_efficiency, 0.9)

    def test_record_recheck_stores_none_after_and_efficiency(self):
        task = make_task()
        task.record_recheck(None, 5.0, None)
        self.assertIsNone(task.before_area)
        self.assertEqual(task.after_area, 5.0)
        self.assertIsNone(task.cleaning_efficiency)
        task.record_recheck(10.0, None, None)
        self.assertEqual(task.before_area, 10.0)
        self.assertIsNone(task.after_area)
        self.assertIsNone(task.cleaning_efficiency)

    def test_age_s_with_explicit_now(self):
        task = make_task(created_at=100.0)
        self.assertEqual(task.age_s(now=100.0), 0.0)
        self.assertEqual(task.age_s(now=160.0), 60.0)
        self.assertEqual(task.age_s(now=1000.0), 900.0)
        # 时钟回退时钳制到 0，绝不返回负年龄。
        self.assertEqual(task.age_s(now=50.0), 0.0)

    def test_age_s_default_now_is_monotonic(self):
        task = make_task(created_at=0.0)
        self.assertGreater(task.age_s(), 0.0)


if __name__ == '__main__':
    unittest.main()
