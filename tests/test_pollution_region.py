"""tests/test_pollution_region.py -- PollutionRegion helpers and V1.0 invariants."""
import unittest

from perception.pollution_region import PollutionRegion


def make_region(**overrides) -> PollutionRegion:
    params = dict(region_id=1, bbox=(10, 20, 30, 40), center_px=(25, 40),
                  area_px=1200.0, visual_score=0.5, area_score=0.4)
    params.update(overrides)
    return PollutionRegion(**params)


class PollutionRegionTests(unittest.TestCase):
    # ------------------------------------------------------------------ normal
    def test_valid_region_helpers(self):
        region = make_region()
        self.assertEqual(region.width, 30)
        self.assertEqual(region.height, 40)
        self.assertTrue(region.center_inside_bbox())
        region.validate()          # must not raise

    def test_optional_fields_default_to_none(self):
        region = make_region()
        for name in ('depth_m', 'relative_x', 'relative_y', 'pollution_score',
                     'pollution_level', 'valid_depth_ratio', 'world_x', 'world_y'):
            self.assertIsNone(getattr(region, name), name)

    def test_extra_defaults_to_a_fresh_dict_per_instance(self):
        first, second = make_region(), make_region()
        self.assertEqual(first.extra, {})
        self.assertEqual(second.extra, {})
        self.assertIsNot(first.extra, second.extra)
        first.extra['note'] = 'dirty'
        self.assertEqual(second.extra, {})

    # ---------------------------------------------------------------- boundary
    def test_center_on_the_bbox_edge_is_inside(self):
        for center in ((10, 20), (40, 60), (10, 60), (40, 20)):
            region = make_region(center_px=center)
            self.assertTrue(region.center_inside_bbox(), center)
            region.validate()

    def test_score_boundaries_are_inclusive(self):
        for visual, area in ((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
            make_region(visual_score=visual, area_score=area).validate()
        make_region(pollution_score=0.0).validate()
        make_region(pollution_score=1.0).validate()

    def test_none_depth_is_accepted(self):
        region = make_region(depth_m=None)
        region.validate()
        self.assertIsNone(region.depth_m)

    # --------------------------------------------------------------- exception
    def test_zero_sized_bbox_raises(self):
        for bbox in ((10, 20, 0, 40), (10, 20, 30, 0), (10, 20, -5, 40)):
            with self.assertRaises(ValueError):
                make_region(bbox=bbox, center_px=(10, 20)).validate()

    def test_non_positive_area_raises(self):
        with self.assertRaises(ValueError):
            make_region(area_px=0.0).validate()
        with self.assertRaises(ValueError):
            make_region(area_px=-12.0).validate()

    def test_center_outside_bbox_raises(self):
        region = make_region(center_px=(5, 40))
        self.assertFalse(region.center_inside_bbox())
        with self.assertRaises(ValueError):
            region.validate()

    def test_scores_outside_unit_interval_raise(self):
        with self.assertRaises(ValueError):
            make_region(visual_score=1.01).validate()
        with self.assertRaises(ValueError):
            make_region(visual_score=-0.01).validate()
        with self.assertRaises(ValueError):
            make_region(area_score=1.5).validate()
        with self.assertRaises(ValueError):
            make_region(area_score=-1.0).validate()

    def test_negative_depth_raises(self):
        with self.assertRaises(ValueError):
            make_region(depth_m=-0.1).validate()
        with self.assertRaises(ValueError):
            make_region(depth_m=0.0).validate()

    def test_pollution_score_outside_unit_interval_raises(self):
        with self.assertRaises(ValueError):
            make_region(pollution_score=1.2).validate()
        with self.assertRaises(ValueError):
            make_region(pollution_score=-0.2).validate()


if __name__ == '__main__':
    unittest.main()
