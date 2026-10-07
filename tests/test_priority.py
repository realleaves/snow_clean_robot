import unittest

import pytest

from decision.cleaning_task import CleaningTask
from decision.priority_scheduler import PriorityScheduler
from mapping.pose_provider import RobotPose
from utils.errors import ConfigError


def make_scheduler(**overrides):
    cfg = dict(pollution_weight=.6, distance_weight=.25, waiting_weight=.15,
               max_distance_m=4, starvation_seconds=120)
    cfg.update(overrides)
    return PriorityScheduler(cfg)


class PriorityTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = PriorityScheduler(dict(pollution_weight=.6, distance_weight=.25,
            waiting_weight=.15, max_distance_m=4, starvation_seconds=120))
        self.pose = RobotPose(0, 0, 0)

    def test_severity_distance_and_wait(self):
        a = CleaningTask(1, 1, 0, .9, 'HEAVY', created_at=100)
        b = CleaningTask(2, 1, 0, .3, 'LIGHT', created_at=100)
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=100),
                           self.scheduler.calculate(b, self.pose, now=100))
        b.pollution_score = .9
        b.target_x = 3
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=100),
                           self.scheduler.calculate(b, self.pose, now=100))
        self.assertGreater(self.scheduler.calculate(a, self.pose, now=500),
                           self.scheduler.calculate(a, self.pose, now=100))

    # ---------------------------------------------------------------- formula
    def test_exact_hand_computed_priority(self):
        """P = ws*S - wd*D + wt*T，外加一个有界的额外老化单位 wt*T。

        取 S=0.5、D=0.5（目标 2 m / max 4 m）、T=0.5（等待 60 s / 120 s）::

            基础项 = 0.60*0.5 - 0.25*0.5 + 0.15*0.5 = 0.250
            老化项 = 0.15*0.5                        = 0.075
            P      = 0.325
        """
        task = CleaningTask(1, 2.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        priority = self.scheduler.calculate(task, self.pose, now=160.0)

        base = 0.60 * 0.5 - 0.25 * 0.5 + 0.15 * 0.5
        aging_bonus = 0.15 * 0.5  # source: 一个有界的额外等待单位
        expected = base + aging_bonus
        self.assertAlmostEqual(base, 0.25)
        self.assertAlmostEqual(expected, 0.325)
        self.assertAlmostEqual(priority, expected)
        self.assertAlmostEqual(task.priority, expected)
        self.assertAlmostEqual(task.distance_score, 0.5)
        self.assertAlmostEqual(task.wait_score, 0.5)

    def test_calculate_writes_scores_back_onto_task(self):
        task = CleaningTask(1, 1.0, 0.0, 0.4, 'LIGHT', created_at=0.0)
        returned = self.scheduler.calculate(task, RobotPose(0, 0, 0), now=30.0)
        self.assertAlmostEqual(task.distance_score, 0.25)
        self.assertAlmostEqual(task.wait_score, 0.25)
        self.assertAlmostEqual(task.priority, returned)

    def test_weights_property(self):
        self.assertEqual(self.scheduler.weights,
                         {'pollution': 0.6, 'distance': 0.25, 'waiting': 0.15})

    # ---------------------------------------------------------------- pollution
    def test_same_distance_severity_order(self):
        heavy = CleaningTask(1, 1.0, 0.0, 0.9, 'HEAVY', created_at=100.0)
        medium = CleaningTask(2, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        light = CleaningTask(3, 1.0, 0.0, 0.2, 'LIGHT', created_at=100.0)
        heavy_p = self.scheduler.calculate(heavy, self.pose, now=100.0)
        medium_p = self.scheduler.calculate(medium, self.pose, now=100.0)
        light_p = self.scheduler.calculate(light, self.pose, now=100.0)
        self.assertGreater(heavy_p, medium_p)
        self.assertGreater(medium_p, light_p)
        self.assertEqual(
            [t.pollution_level for t in self.scheduler.rank([light, heavy, medium],
                                                            self.pose, now=100.0)],
            ['HEAVY', 'MEDIUM', 'LIGHT'])

    # ---------------------------------------------------------------- distance
    def test_same_pollution_nearer_beats_farther(self):
        near = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        far = CleaningTask(2, 3.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        self.assertGreater(self.scheduler.calculate(near, self.pose, now=100.0),
                           self.scheduler.calculate(far, self.pose, now=100.0))

    def test_distance_score_saturates_beyond_max_distance(self):
        pose = RobotPose(0, 0, 0)
        at_limit = CleaningTask(1, 4.0, 0.0, 0.5, 'MEDIUM')
        beyond = CleaningTask(2, 10.0, 0.0, 0.5, 'MEDIUM')
        self.assertAlmostEqual(self.scheduler.distance_score(at_limit, pose), 1.0)
        self.assertAlmostEqual(self.scheduler.distance_score(beyond, pose), 1.0)
        # 饱和之后优先级完全相同（等待时间一致）。
        self.assertAlmostEqual(self.scheduler.calculate(at_limit, pose, now=0.0),
                               self.scheduler.calculate(beyond, pose, now=0.0))

    def test_distance_score_zero_at_robot_position(self):
        task = CleaningTask(1, 0.0, 0.0, 0.5, 'MEDIUM')
        self.assertAlmostEqual(self.scheduler.distance_score(task, self.pose), 0.0)

    # ---------------------------------------------------------------- waiting
    def test_same_pollution_and_distance_longer_wait_wins(self):
        older = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=0.0)
        newer = CleaningTask(2, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        self.assertGreater(self.scheduler.calculate(older, self.pose, now=200.0),
                           self.scheduler.calculate(newer, self.pose, now=200.0))

    def test_wait_score_saturates_at_starvation_seconds(self):
        task = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=0.0)
        self.assertAlmostEqual(self.scheduler.wait_score(task, now=60.0), 0.5)
        self.assertAlmostEqual(self.scheduler.wait_score(task, now=120.0), 1.0)
        self.assertAlmostEqual(self.scheduler.wait_score(task, now=600.0), 1.0)

    def test_is_starving_boundary(self):
        task = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        self.assertFalse(self.scheduler.is_starving(task, now=100.0 + 119.0))
        self.assertTrue(self.scheduler.is_starving(task, now=100.0 + 120.0))
        self.assertTrue(self.scheduler.is_starving(task, now=100.0 + 500.0))

    # ------------------------------------------------------------ anti-starvation
    def test_long_waiting_light_task_eventually_outranks_fresh_heavy(self):
        """规格 17：等待足够久的 LIGHT 任务最终要压过刚发现的 HEAVY 任务。

        污染项差距为 0.60*(0.9-0.2)=0.42，而受限老化项最多贡献 2*0.15=0.30，
        因此 LIGHT 必须在距离上更近才能完成反超（这里取 D=0 对 D=1.0）。
        """
        now = 1000.0
        light = CleaningTask(1, 0.0, 0.0, 0.2, 'LIGHT', created_at=0.0)
        heavy = CleaningTask(2, 4.0, 0.0, 0.9, 'HEAVY', created_at=now)
        self.assertTrue(self.scheduler.is_starving(light, now=now))
        self.assertFalse(self.scheduler.is_starving(heavy, now=now))
        chosen = self.scheduler.next_task([heavy, light], self.pose, now=now)
        self.assertIs(chosen, light)
        self.assertGreater(chosen.priority, heavy.priority)

    def test_priority_of_waiting_task_strictly_increases_with_age(self):
        light = CleaningTask(1, 0.0, 0.0, 0.2, 'LIGHT', created_at=0.0)
        priorities = [self.scheduler.calculate(light, self.pose, now=age)
                      for age in (10.0, 30.0, 60.0, 90.0, 119.0)]
        self.assertEqual(priorities, sorted(priorities))
        for earlier, later in zip(priorities, priorities[1:]):
            self.assertLess(earlier, later)
        # 饱和之后不再增长，保持有界。
        self.assertAlmostEqual(self.scheduler.calculate(light, self.pose, now=120.0),
                               self.scheduler.calculate(light, self.pose, now=9999.0))

    # ---------------------------------------------------------------- selection
    def test_next_task_on_empty_list_returns_none(self):
        self.assertIsNone(self.scheduler.next_task([], self.pose, now=0.0))
        self.assertEqual(self.scheduler.rank([], self.pose, now=0.0), [])

    def test_next_task_returns_single_task(self):
        task = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=0.0)
        self.assertIs(self.scheduler.next_task([task], self.pose, now=0.0), task)

    def test_next_task_returns_highest_priority(self):
        heavy = CleaningTask(1, 1.0, 0.0, 0.9, 'HEAVY', created_at=0.0)
        light = CleaningTask(2, 1.0, 0.0, 0.2, 'LIGHT', created_at=0.0)
        self.assertIs(self.scheduler.next_task([light, heavy], self.pose, now=0.0), heavy)

    def test_tie_break_prefers_earlier_created_task(self):
        early = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        late = CleaningTask(2, 1.0, 0.0, 0.5, 'MEDIUM', created_at=200.0)
        self.assertIs(self.scheduler.next_task([late, early], self.pose, now=100.0), early)
        self.assertEqual(self.scheduler.rank([late, early], self.pose, now=100.0),
                         [early, late])
        self.assertEqual(early.priority, late.priority)

    def test_tie_break_is_deterministic_for_equal_created_at(self):
        first = CleaningTask(1, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        second = CleaningTask(2, 1.0, 0.0, 0.5, 'MEDIUM', created_at=100.0)
        self.assertIs(self.scheduler.next_task([first, second], self.pose, now=100.0), first)
        self.assertIs(self.scheduler.next_task([first, second], self.pose, now=100.0), first)

    def test_rank_sorts_descending_and_updates_every_task(self):
        tasks = [CleaningTask(1, 3.0, 0.0, 0.2, 'LIGHT', created_at=0.0),
                 CleaningTask(2, 1.0, 0.0, 0.9, 'HEAVY', created_at=0.0),
                 CleaningTask(3, 2.0, 0.0, 0.5, 'MEDIUM', created_at=0.0)]
        ranked = self.scheduler.rank(tasks, self.pose, now=60.0)
        priorities = [t.priority for t in ranked]
        self.assertEqual(priorities, sorted(priorities, reverse=True))
        for task in tasks:
            self.assertAlmostEqual(task.priority,
                                   self.scheduler.calculate(task, self.pose, now=60.0))
            self.assertGreaterEqual(task.distance_score, 0.0)
            self.assertLessEqual(task.distance_score, 1.0)
            self.assertGreaterEqual(task.wait_score, 0.0)
            self.assertLessEqual(task.wait_score, 1.0)


class PriorityConfigTests(unittest.TestCase):
    def test_negative_weight_is_rejected(self):
        with pytest.raises(ConfigError):
            make_scheduler(distance_weight=-0.1)
        with pytest.raises(ConfigError):
            make_scheduler(waiting_weight=-0.5)

    def test_nonpositive_pollution_weight_is_rejected(self):
        with pytest.raises(ConfigError):
            make_scheduler(pollution_weight=0.0)

    def test_missing_keys_are_rejected(self):
        for key in ('pollution_weight', 'distance_weight', 'waiting_weight',
                    'max_distance_m', 'starvation_seconds'):
            cfg = dict(pollution_weight=.6, distance_weight=.25, waiting_weight=.15,
                       max_distance_m=4, starvation_seconds=120)
            del cfg[key]
            with pytest.raises(ConfigError):
                PriorityScheduler(cfg)

    def test_nonpositive_distance_scale_is_rejected(self):
        with pytest.raises(ConfigError):
            make_scheduler(max_distance_m=0)
        with pytest.raises(ConfigError):
            make_scheduler(max_distance_m=-1.0)

    def test_nonpositive_starvation_scale_is_rejected(self):
        with pytest.raises(ConfigError):
            make_scheduler(starvation_seconds=0)
        with pytest.raises(ConfigError):
            make_scheduler(starvation_seconds=-120)


if __name__ == '__main__':
    unittest.main()
