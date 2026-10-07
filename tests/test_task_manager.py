"""决策层验收测试：:mod:`decision.task_manager`。

覆盖任务入队与校验、优先级选择、状态机封装（complete/fail/cancel/retry/
manual_check/update）、队列查询视图、坏目标判定、批量重算/取消，以及规格 18
的重复目标语义。
"""
import math
import unittest

import pytest

from decision.cleaning_task import CleaningTask, TaskStatus
from decision.priority_scheduler import PriorityScheduler
from decision.task_manager import TaskManager
from mapping.pose_provider import RobotPose
from tests.helpers import PRIORITY_CFG
from utils.errors import TaskError


def make_task(task_id=1, x=0.0, y=0.0, score=0.5, level='MEDIUM',
              created_at=0.0, status='WAITING', **kwargs):
    return CleaningTask(task_id, x, y, score, level, created_at=created_at,
                        status=status, **kwargs)


class TaskManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager(PriorityScheduler(dict(PRIORITY_CFG)))
        self.pose = RobotPose(0, 0, 0)

    # ------------------------------------------------------------------ add/get
    def test_add_task_and_lookup(self):
        task = make_task(1)
        returned = self.manager.add_task(task)
        self.assertIs(returned, task)
        self.assertEqual(len(self.manager), 1)
        self.assertIn(1, self.manager)
        self.assertIs(self.manager.get(1), task)

    def test_duplicate_id_raises(self):
        self.manager.add_task(make_task(1))
        with pytest.raises(TaskError):
            self.manager.add_task(make_task(1))
        # replace=True 允许覆盖。
        replacement = make_task(1, score=0.9, level='HEAVY')
        self.assertIs(self.manager.add_task(replacement, replace=True), replacement)
        self.assertIs(self.manager.get(1), replacement)
        self.assertEqual(len(self.manager), 1)

    def test_get_unknown_id_raises(self):
        with pytest.raises(TaskError):
            self.manager.get(42)

    def test_add_non_task_raises(self):
        for payload in ('not a task', None, object(), 7):
            with pytest.raises(TaskError):
                self.manager.add_task(payload)

    def test_invalid_pollution_score_raises(self):
        for score in (-0.1, 1.1, float('nan'), float('inf')):
            with pytest.raises(TaskError):
                self.manager.add_task(make_task(1, score=score))
        # 边界值合法。
        self.manager.add_task(make_task(1, score=0.0))
        self.manager.add_task(make_task(2, score=1.0))

    # ------------------------------------------------------------------ selection
    def test_get_next_task_on_empty_queue_returns_none(self):
        self.assertIsNone(self.manager.get_next_task(self.pose, now=0.0))

    def test_get_next_task_single_task(self):
        task = self.manager.add_task(make_task(1))
        self.assertIs(self.manager.get_next_task(self.pose, now=0.0), task)

    def test_get_next_task_highest_priority(self):
        light = self.manager.add_task(make_task(1, score=0.2, level='LIGHT'))
        heavy = self.manager.add_task(make_task(2, score=0.9, level='HEAVY'))
        self.assertIs(self.manager.get_next_task(self.pose, now=0.0), heavy)
        self.assertEqual(self.manager.ranked(self.pose, now=0.0), [heavy, light])

    def test_get_next_task_tie_is_deterministic(self):
        early = self.manager.add_task(make_task(1, created_at=0.0))
        late = self.manager.add_task(make_task(2, created_at=100.0))
        self.assertIs(self.manager.get_next_task(self.pose, now=100.0), early)

    def test_get_next_task_ignores_non_waiting_tasks(self):
        busy = self.manager.add_task(make_task(1, score=0.9, level='HEAVY'))
        self.manager.set_status(1, TaskStatus.PLANNING)
        waiting = self.manager.add_task(make_task(2, score=0.2, level='LIGHT'))
        self.assertIs(self.manager.get_next_task(self.pose, now=0.0), waiting)
        self.manager.set_status(2, TaskStatus.PLANNING)
        self.assertIsNone(self.manager.get_next_task(self.pose, now=0.0))
        self.assertIs(busy.status, 'PLANNING')

    # ------------------------------------------------------------------ status
    def test_complete_requires_a_completed_transition_path(self):
        self.manager.add_task(make_task(1))
        with pytest.raises(TaskError):
            self.manager.complete(1)  # WAITING -> COMPLETED 非法
        for status in (TaskStatus.PLANNING, TaskStatus.NAVIGATING,
                       TaskStatus.CLEANING, TaskStatus.RECHECK):
            self.manager.set_status(1, status)
        self.manager.complete(1)
        self.assertEqual(self.manager.get(1).status, 'COMPLETED')
        self.assertTrue(self.manager.get(1).succeeded)

    def test_fail_records_reason(self):
        self.manager.add_task(make_task(1))
        task = self.manager.fail(1, reason='no path')
        self.assertEqual(task.status, 'FAILED')
        self.assertEqual(task.last_error, 'no path')
        self.assertIn(task, self.manager.failed())
        self.manager.add_task(make_task(2))
        self.assertIsNone(self.manager.fail(2).last_error)

    def test_cancel_and_canceled_view(self):
        self.manager.add_task(make_task(1))
        task = self.manager.cancel(1)
        self.assertEqual(task.status, 'CANCELLED')
        self.assertEqual(self.manager.canceled(), [task])
        self.assertTrue(task.terminal)

    def test_manual_check_records_reason(self):
        self.manager.add_task(make_task(1))
        task = self.manager.manual_check(1, reason='needs operator')
        self.assertEqual(task.status, 'MANUAL_CHECK')
        self.assertEqual(task.last_error, 'needs operator')
        self.assertTrue(task.terminal)

    def test_set_status_force_bypasses_guard(self):
        self.manager.add_task(make_task(1))
        self.manager.set_status(1, TaskStatus.PLANNING)
        self.manager.set_status(1, TaskStatus.NAVIGATING)
        self.manager.set_status(1, TaskStatus.CLEANING)
        self.manager.set_status(1, TaskStatus.RECHECK)
        self.manager.complete(1)
        with pytest.raises(TaskError):
            self.manager.set_status(1, TaskStatus.WAITING)
        self.manager.set_status(1, TaskStatus.WAITING, force=True)
        self.assertEqual(self.manager.get(1).status, 'WAITING')

    def test_retry_of_completed_is_rejected(self):
        self.manager.add_task(make_task(1, status='COMPLETED'))
        with pytest.raises(TaskError):
            self.manager.retry(1)

    def test_retry_of_failed_and_manual_check_returns_to_waiting(self):
        for task_id, status in ((1, 'FAILED'), (2, 'MANUAL_CHECK')):
            self.manager.add_task(make_task(task_id, status=status))
            task = self.manager.retry(task_id)
            self.assertEqual(task.status, 'WAITING')
            self.assertFalse(task.terminal)
        self.assertEqual(len(self.manager.waiting()), 2)

    def test_retry_unknown_id_raises(self):
        with pytest.raises(TaskError):
            self.manager.retry(99)

    def test_update_fields_and_unknown_field(self):
        self.manager.add_task(make_task(1))
        task = self.manager.update(1, last_error='boom', pollution_score=0.9,
                                   retry_count=1)
        self.assertEqual(task.last_error, 'boom')
        self.assertEqual(task.pollution_score, 0.9)
        self.assertEqual(task.retry_count, 1)
        # status 字段走状态机校验。
        self.manager.update(1, status=TaskStatus.PLANNING)
        self.assertEqual(task.status, 'PLANNING')
        with pytest.raises(TaskError):
            self.manager.update(1, not_a_field=1)
        with pytest.raises(TaskError):
            self.manager.update(99, last_error='x')

    def test_update_status_still_enforces_transitions(self):
        self.manager.add_task(make_task(1))
        with pytest.raises(TaskError):
            self.manager.update(1, status=TaskStatus.NAVIGATING)

    def test_history_records_transitions(self):
        self.manager.add_task(make_task(1))
        self.assertEqual(self.manager.history[-1], (1, 'NEW', 'WAITING'))
        self.manager.set_status(1, TaskStatus.PLANNING)
        self.assertEqual(self.manager.history[-1], (1, 'WAITING', 'PLANNING'))
        self.manager.fail(1, reason='x')
        self.assertEqual(self.manager.history[-1], (1, 'PLANNING', 'FAILED'))
        self.assertEqual(len(self.manager.history), 3)

    # ------------------------------------------------------------------ queries
    def test_by_status_and_convenience_views(self):
        waiting = self.manager.add_task(make_task(1))
        busy = self.manager.add_task(make_task(2))
        self.manager.set_status(2, TaskStatus.PLANNING)
        failed = self.manager.add_task(make_task(3, status='FAILED'))
        canceled = self.manager.add_task(make_task(4, status='CANCELLED'))
        completed = self.manager.add_task(make_task(5, status='COMPLETED'))

        self.assertEqual(self.manager.waiting(), [waiting])
        self.assertEqual(self.manager.by_status(TaskStatus.WAITING), [waiting])
        self.assertEqual(self.manager.by_status('WAITING'), [waiting])
        self.assertEqual(self.manager.by_status(TaskStatus.FAILED, 'CANCELLED'),
                         [failed, canceled])
        self.assertEqual(self.manager.failed(), [failed])
        self.assertEqual(self.manager.canceled(), [canceled])
        self.assertEqual(self.manager.active(), [waiting, busy])
        self.assertEqual(self.manager.open_task_ids(), [1, 2])
        for terminal in (failed, canceled, completed):
            self.assertTrue(terminal.terminal)

    # ------------------------------------------------------------------ targets
    def test_find_by_target_same_coordinates(self):
        open_task = self.manager.add_task(make_task(1, x=1.0, y=2.0))
        self.assertIs(self.manager.find_by_target(1.0, 2.0), open_task)
        self.assertIs(self.manager.find_by_target(1.0 + 1e-9, 2.0), open_task)

    def test_find_by_target_ignores_closed_tasks(self):
        self.manager.add_task(make_task(1, x=5.0, y=5.0, status='COMPLETED'))
        self.manager.add_task(make_task(2, x=6.0, y=6.0, status='FAILED'))
        self.manager.add_task(make_task(3, x=7.0, y=7.0, status='MANUAL_CHECK'))
        self.assertIsNone(self.manager.find_by_target(5.0, 5.0))
        self.assertIsNone(self.manager.find_by_target(6.0, 6.0))
        self.assertIsNone(self.manager.find_by_target(7.0, 7.0))

    def test_find_by_target_far_away_returns_none(self):
        self.manager.add_task(make_task(1, x=1.0, y=1.0))
        self.assertIsNone(self.manager.find_by_target(9.0, 9.0))
        self.assertIsNone(self.manager.find_by_target(1.0 + 0.5, 1.0))

    def test_is_known_bad_target_within_and_outside_tolerance(self):
        self.manager.add_task(make_task(1, x=1.0, y=1.0, status='FAILED'))
        self.manager.add_task(make_task(2, x=5.0, y=5.0, status='MANUAL_CHECK'))
        self.assertTrue(self.manager.is_known_bad_target(1.0, 1.0))
        self.assertTrue(self.manager.is_known_bad_target(1.0, 1.2, tolerance_m=0.2))
        self.assertFalse(self.manager.is_known_bad_target(1.0, 1.21, tolerance_m=0.2))
        self.assertFalse(self.manager.is_known_bad_target(1.0, 3.0))
        self.assertTrue(self.manager.is_known_bad_target(5.0, 5.0, tolerance_m=0.01))
        # tolerance 为 0 时只有完全重合才命中。
        self.assertTrue(self.manager.is_known_bad_target(5.0, 5.0, tolerance_m=0.0))
        self.assertFalse(self.manager.is_known_bad_target(5.0, 5.1, tolerance_m=0.0))

    def test_is_known_bad_target_ignores_open_tasks(self):
        self.manager.add_task(make_task(1, x=2.0, y=2.0, status='WAITING'))
        self.assertFalse(self.manager.is_known_bad_target(2.0, 2.0))

    # ------------------------------------------------------------------ bulk
    def test_recalculate_all_refreshes_waiting_priorities(self):
        waiting = [self.manager.add_task(make_task(i, x=float(i), score=0.5,
                                                   created_at=0.0))
                   for i in (1, 2, 3)]
        self.manager.set_status(3, TaskStatus.PLANNING)
        ranked = self.manager.recalculate_all(self.pose, now=60.0)
        self.assertEqual(len(ranked), 2)
        self.assertEqual([id(t) for t in ranked], [id(waiting[0]), id(waiting[1])])
        for task in ranked:
            self.assertAlmostEqual(task.wait_score, 0.5)
            self.assertNotEqual(task.priority, 0.0)
        self.assertEqual(ranked[0].priority, max(t.priority for t in ranked))
        # 非 WAITING 任务不被重算。
        busy = self.manager.get(3)
        self.assertEqual(busy.priority, 0.0)

    def test_cancel_all_cancels_every_open_task(self):
        open_tasks = [self.manager.add_task(make_task(i)) for i in (1, 2)]
        done = self.manager.add_task(make_task(3, status='COMPLETED'))
        self.manager.cancel_all()
        for task in open_tasks:
            self.assertEqual(task.status, 'CANCELLED')
        self.assertEqual(done.status, 'COMPLETED')
        self.assertEqual(self.manager.open_task_ids(), [])
        self.assertEqual(len(self.manager.canceled()), 2)

    # ------------------------------------------------------- duplicate targets
    def test_duplicate_target_task_is_possible_and_open_one_is_found(self):
        first = self.manager.add_task(make_task(1, x=2.0, y=2.0))
        second = self.manager.add_task(make_task(2, x=2.0, y=2.0))
        self.assertEqual(len(self.manager), 2)
        found = self.manager.find_by_target(2.0, 2.0)
        self.assertIs(found, first)
        self.assertIsNot(found, second)
        self.assertFalse(found.terminal)


if __name__ == '__main__':
    unittest.main()
