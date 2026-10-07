"""反馈层验收测试：:mod:`feedback.cleaning_evaluator`。

覆盖 eta 清洗效率、无效测量、成功阈值边界、``evaluate_task`` 回写，以及
阈值配置校验。
"""
import unittest

import pytest

from decision.cleaning_task import CleaningTask
from feedback.cleaning_evaluator import CleaningEvaluator, Evaluation
from tests.helpers import CLEANING_FEEDBACK
from utils.errors import ConfigError, TaskError

#: V1.0 验收的配置阈值。
THRESHOLD = 0.80


def make_evaluator(threshold=THRESHOLD):
    return CleaningEvaluator(threshold)


def make_task(task_id=1, before_area=None, after_area=None):
    return CleaningTask(task_id, 0.0, 0.0, 0.5, 'MEDIUM',
                        before_area=before_area, after_area=after_area)


class CleaningEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = CleaningEvaluator(dict(CLEANING_FEEDBACK)['success_threshold'])
        self.assertEqual(self.evaluator.success_threshold, THRESHOLD)

    # ---------------------------------------------------------------- eta table
    def test_eta_table(self):
        high = self.evaluator.evaluate(1000.0, 100.0)
        self.assertAlmostEqual(high.efficiency, 0.90)
        self.assertTrue(high.passed)
        self.assertEqual(high.reason, 'PASS')

        half = self.evaluator.evaluate(1000.0, 500.0)
        self.assertAlmostEqual(half.efficiency, 0.50)
        self.assertFalse(half.passed)
        self.assertEqual(half.reason, 'FAIL')

        none = self.evaluator.evaluate(1000.0, 1000.0)
        self.assertAlmostEqual(none.efficiency, 0.0)
        self.assertFalse(none.passed)
        self.assertEqual(none.reason, 'FAIL')

    def test_evaluation_result_is_a_dataclass(self):
        result = self.evaluator.evaluate(1000.0, 100.0)
        self.assertIsInstance(result, Evaluation)
        self.assertAlmostEqual(result.efficiency, 0.9)
        self.assertTrue(result.passed)

    # ---------------------------------------------------------------- invalid
    def test_before_area_zero_is_invalid(self):
        result = self.evaluator.evaluate(0.0, 100.0)
        self.assertIsNone(result.efficiency)
        self.assertFalse(result.passed)
        self.assertIn('INVALID', result.reason)
        self.assertIsNone(self.evaluator.efficiency(0.0, 100.0))

    def test_negative_before_area_is_invalid(self):
        result = self.evaluator.evaluate(-10.0, 5.0)
        self.assertIsNone(result.efficiency)
        self.assertFalse(result.passed)
        self.assertIn('INVALID', result.reason)

    def test_negative_after_area_is_invalid(self):
        result = self.evaluator.evaluate(1000.0, -1.0)
        self.assertIsNone(result.efficiency)
        self.assertFalse(result.passed)
        self.assertIn('INVALID', result.reason)
        self.assertIsNone(self.evaluator.efficiency(1000.0, -1.0))

    def test_none_measurements_are_invalid(self):
        self.assertIsNone(self.evaluator.efficiency(None, 100.0))
        self.assertIsNone(self.evaluator.efficiency(1000.0, None))
        result = self.evaluator.evaluate(1000.0, None)
        self.assertIsNone(result.efficiency)
        self.assertFalse(result.passed)
        self.assertIn('INVALID', result.reason)

    def test_after_greater_than_before_is_clamped_to_zero(self):
        result = self.evaluator.evaluate(1000.0, 1500.0)
        self.assertEqual(result.efficiency, 0.0)
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, 'FAIL')
        self.assertEqual(self.evaluator.efficiency(1000.0, 1500.0), 0.0)

    # ---------------------------------------------------------------- thresholds
    def test_success_threshold_boundaries(self):
        self.assertFalse(self.evaluator.evaluate(1000.0, 201.0).passed)   # 0.799
        self.assertAlmostEqual(self.evaluator.efficiency(1000.0, 201.0), 0.799)
        self.assertTrue(self.evaluator.evaluate(1000.0, 200.0).passed)    # 0.800
        self.assertAlmostEqual(self.evaluator.efficiency(1000.0, 200.0), 0.800)
        self.assertTrue(self.evaluator.evaluate(1000.0, 199.0).passed)    # 0.801
        self.assertAlmostEqual(self.evaluator.efficiency(1000.0, 199.0), 0.801)

    def test_efficiency_is_clamped_into_unit_interval(self):
        self.assertEqual(self.evaluator.efficiency(100.0, 0.0), 1.0)
        self.assertEqual(self.evaluator.efficiency(100.0, 100.0), 0.0)

    def test_zero_and_one_thresholds_are_accepted(self):
        self.assertTrue(CleaningEvaluator(0.0).evaluate(1000.0, 1000.0).passed)
        self.assertFalse(CleaningEvaluator(1.0).evaluate(1000.0, 1.0).passed)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-0.01, 1.01, 2, '0.8', None, float('nan'), True):
            with pytest.raises(ConfigError):
                CleaningEvaluator(threshold)

    # ---------------------------------------------------------------- task API
    def test_evaluate_task_uses_before_area_and_writes_back(self):
        task = make_task(1, before_area=1000.0, after_area=200.0)
        result = self.evaluator.evaluate_task(task)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(task.cleaning_efficiency, 0.8)
        self.assertAlmostEqual(result.efficiency, 0.8)

    def test_evaluate_task_writes_back_failing_efficiency(self):
        task = make_task(2, before_area=1000.0, after_area=500.0)
        result = self.evaluator.evaluate_task(task)
        self.assertFalse(result.passed)
        self.assertAlmostEqual(task.cleaning_efficiency, 0.5)

    def test_evaluate_task_without_before_area_raises(self):
        with pytest.raises(TaskError):
            self.evaluator.evaluate_task(make_task(3, before_area=None, after_area=10.0))

    def test_evaluate_task_with_none_after_area_is_invalid_not_crash(self):
        task = make_task(4, before_area=1000.0, after_area=None)
        result = self.evaluator.evaluate_task(task)
        self.assertIsNone(result.efficiency)
        self.assertFalse(result.passed)
        self.assertIsNone(task.cleaning_efficiency)


if __name__ == '__main__':
    unittest.main()
