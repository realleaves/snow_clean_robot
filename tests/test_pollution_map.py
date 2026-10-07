"""tests/test_pollution_map.py -- world-frame target memory and merge/expiry rules."""
import math
import unittest

import pytest

from mapping.pollution_map import (
    STATUS_ACTIVE, STATUS_COMPLETED, PollutionMap, PollutionTarget,
)
from utils.errors import ConfigError, TaskError


class PollutionMapTests(unittest.TestCase):
    # ------------------------------------------------------------------ normal
    def test_first_observation_creates_target_one(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        target, action = pollution_map.upsert(1.00, 1.00, 0.50, 'MEDIUM', now=100.0)
        self.assertEqual(action, 'created')
        self.assertEqual(target.id, 1)
        self.assertIsInstance(target, PollutionTarget)
        self.assertEqual((target.x, target.y), pytest.approx((1.00, 1.00)))
        self.assertEqual(target.score, pytest.approx(0.50))
        self.assertEqual(target.level, 'MEDIUM')
        self.assertEqual(target.status, STATUS_ACTIVE)
        self.assertTrue(target.active)
        self.assertEqual(target.observations, 1)
        self.assertEqual(target.last_seen, 100.0)
        self.assertEqual(target.history, [(1.00, 1.00)])
        self.assertEqual(len(pollution_map), 1)

    def test_nearby_observation_merges_into_the_same_target(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        first, _ = pollution_map.upsert(1.00, 1.00, 0.60, 'MEDIUM', now=100.0)
        second, action = pollution_map.upsert(1.10, 1.08, 0.40, 'MEDIUM', now=105.0)
        self.assertEqual(action, 'merged')
        self.assertIs(second, first)
        self.assertEqual(first.id, 1)
        self.assertEqual(len(pollution_map), 1)
        self.assertEqual(first.observations, 2)
        # The strongest observation wins, and the latest position/time are kept.
        self.assertEqual(first.score, pytest.approx(0.60))
        self.assertEqual((first.x, first.y), pytest.approx((1.10, 1.08)))
        self.assertEqual(first.last_seen, 105.0)
        self.assertEqual(first.history, [(1.00, 1.00), (1.10, 1.08)])
        self.assertEqual(pollution_map.merge_events, 1)
        # A higher score does raise the stored score.
        pollution_map.upsert(1.12, 1.10, 0.90, 'HEAVY', now=106.0)
        self.assertEqual(first.score, pytest.approx(0.90))
        self.assertEqual(first.level, 'HEAVY')

    def test_distance_beyond_merge_radius_creates_a_second_target(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        first, _ = pollution_map.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        second, action = pollution_map.upsert(2.0, 2.0, 0.7, 'HEAVY', now=1.0)
        self.assertEqual(action, 'created')
        self.assertEqual(second.id, 2)
        self.assertEqual(len(pollution_map), 2)
        nearest, distance = pollution_map.nearest(0.1, 0.0)
        self.assertIs(nearest, first)
        self.assertEqual(distance, pytest.approx(0.1))
        self.assertIs(pollution_map.nearest(3.0, 3.0)[0], second)
        self.assertEqual(pollution_map.distance_to(second.id, 0.0, 0.0),
                         pytest.approx(math.hypot(2.0, 2.0)))
        self.assertIn(first.id, pollution_map)
        self.assertIn(second.id, pollution_map)
        self.assertNotIn(99, pollution_map)

    def test_repeated_observation_still_yields_one_target(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        target, _ = pollution_map.upsert(1.0, 1.0, 0.5, 'LIGHT', now=0.0)
        for step in range(1, 20):
            merged, action = pollution_map.upsert(1.0 + 0.001 * step, 1.0, 0.5, 'LIGHT',
                                                  now=float(step))
            self.assertEqual(action, 'merged')
            self.assertIs(merged, target)
        self.assertEqual(len(pollution_map), 1)
        self.assertEqual(target.observations, 20)
        self.assertEqual(target.last_seen, 19.0)

    # ---------------------------------------------------------------- boundary
    def test_merge_radius_is_inclusive(self):
        # Just inside the radius -> merged into the original target.
        inside = PollutionMap(merge_distance_m=0.25)
        target, _ = inside.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        merged, action = inside.upsert(0.24, 0.0, 0.5, 'LIGHT', now=1.0)
        self.assertEqual(action, 'merged')
        self.assertIs(merged, target)
        self.assertEqual(len(inside), 1)
        # Exactly on the radius -> still merged (inclusive boundary, spec 15).
        edge = PollutionMap(merge_distance_m=0.25)
        edge.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        _, action = edge.upsert(0.25, 0.0, 0.5, 'LIGHT', now=1.0)
        self.assertEqual(action, 'merged')
        self.assertEqual(len(edge), 1)
        # Just beyond the radius -> a separate target.
        outside = PollutionMap(merge_distance_m=0.25)
        outside.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        _, action = outside.upsert(0.26, 0.0, 0.5, 'LIGHT', now=1.0)
        self.assertEqual(action, 'created')
        self.assertEqual(len(outside), 2)

    def test_nearest_on_an_empty_map(self):
        pollution_map = PollutionMap()
        self.assertEqual(pollution_map.nearest(0.0, 0.0), (None, math.inf))

    def test_active_and_completed_partition(self):
        pollution_map = PollutionMap()
        active, _ = pollution_map.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        done, _ = pollution_map.upsert(3.0, 3.0, 0.5, 'LIGHT', now=0.0)
        pollution_map.complete(done.id)
        self.assertEqual(pollution_map.active_targets(), [active])
        self.assertEqual(pollution_map.completed_targets(), [done])
        self.assertTrue(active.active)
        self.assertFalse(done.active)

    # --------------------------------------------------------------- mutation
    def test_update_changes_fields(self):
        pollution_map = PollutionMap()
        target, _ = pollution_map.upsert(0.0, 0.0, 0.2, 'LIGHT', now=0.0)
        returned = pollution_map.update(target.id, x=1.0, y=2.0, score=0.8, level='HEAVY',
                                        status=STATUS_ACTIVE, now=7.0)
        self.assertIs(returned, target)
        self.assertEqual(returned.x, 1.0)
        self.assertEqual(returned.y, 2.0)
        self.assertEqual(returned.score, pytest.approx(0.8))
        self.assertEqual(returned.level, 'HEAVY')
        self.assertEqual(returned.status, STATUS_ACTIVE)
        self.assertEqual(returned.last_seen, 7.0)

    def test_complete_reopen_remove_lifecycle(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        target, _ = pollution_map.upsert(1.0, 1.0, 0.5, 'LIGHT', now=0.0)
        pollution_map.complete(target.id)
        self.assertEqual(target.status, STATUS_COMPLETED)
        self.assertFalse(target.active)
        # A completed target is no longer merged into: a later sighting is new.
        fresh, action = pollution_map.upsert(1.05, 1.05, 0.5, 'LIGHT', now=1.0)
        self.assertEqual(action, 'created')
        self.assertNotEqual(fresh.id, target.id)
        self.assertEqual(len(pollution_map), 2)
        # ... unless it is reopened explicitly.
        pollution_map.reopen(target.id)
        self.assertEqual(target.status, STATUS_ACTIVE)
        merged, action = pollution_map.upsert(1.02, 1.02, 0.5, 'LIGHT', now=2.0)
        self.assertEqual(action, 'merged')
        self.assertIs(merged, target)
        pollution_map.remove(target.id)
        self.assertNotIn(target.id, pollution_map)
        self.assertEqual(len(pollution_map), 1)

    def test_unknown_id_raises_task_error(self):
        pollution_map = PollutionMap()
        target, _ = pollution_map.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        with self.assertRaises(TaskError):
            pollution_map.remove(99)
        with self.assertRaises(TaskError):
            pollution_map.update(99, x=1.0)
        with self.assertRaises(TaskError):
            pollution_map.complete(99)
        with self.assertRaises(TaskError):
            pollution_map.reopen(99)
        with self.assertRaises(TaskError):
            pollution_map.get(99)
        with self.assertRaises(TaskError):
            pollution_map.distance_to(99, 0.0, 0.0)
        with self.assertRaises(TaskError):
            pollution_map.update(target.id, status='BOGUS')

    # ------------------------------------------------------------------ expiry
    def test_expire_removes_only_stale_active_targets(self):
        pollution_map = PollutionMap(merge_distance_m=0.25)
        old, _ = pollution_map.upsert(0.0, 0.0, 0.5, 'LIGHT', now=0.0)
        fresh, _ = pollution_map.upsert(2.0, 2.0, 0.5, 'LIGHT', now=60.0)
        done, _ = pollution_map.upsert(5.0, 5.0, 0.5, 'LIGHT', now=0.0)
        pollution_map.complete(done.id)
        removed = pollution_map.expire(now=100.0, max_age_s=50.0)
        self.assertEqual(removed, [old.id])
        self.assertNotIn(old.id, pollution_map)
        self.assertIn(fresh.id, pollution_map)
        self.assertIn(done.id, pollution_map)     # completed targets are kept
        self.assertEqual(len(pollution_map), 2)

    def test_expire_boundary_and_bad_age(self):
        pollution_map = PollutionMap()
        target, _ = pollution_map.upsert(0.0, 0.0, 0.5, 'LIGHT', now=10.0)
        # Exactly max_age_s old is *not* stale (strict greater-than).
        self.assertEqual(pollution_map.expire(now=60.0, max_age_s=50.0), [])
        self.assertIn(target.id, pollution_map)
        self.assertEqual(pollution_map.expire(now=60.0 + 1e-6, max_age_s=50.0), [target.id])
        for bad in (0.0, -1.0):
            with self.assertRaises(ConfigError):
                pollution_map.expire(now=100.0, max_age_s=bad)

    def test_non_finite_observation_raises(self):
        pollution_map = PollutionMap()
        for x, y, score in ((float('nan'), 0.0, 0.5), (0.0, float('nan'), 0.5),
                            (0.0, float('inf'), 0.5), (0.0, 0.0, float('nan'))):
            with self.assertRaises(TaskError):
                pollution_map.upsert(x, y, score, 'LIGHT', now=0.0)
        self.assertEqual(len(pollution_map), 0)

    def test_non_positive_merge_distance_raises(self):
        for bad in (0.0, -0.25):
            with self.assertRaises(ConfigError):
                PollutionMap(merge_distance_m=bad)


if __name__ == '__main__':
    unittest.main()
