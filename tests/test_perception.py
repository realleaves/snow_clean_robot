import unittest

from camera.mock_camera import MockCamera
from camera.virtual_scene import VirtualWorld
from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector

ROI = dict(x_start=40, x_end=600, y_start=180, y_end=480)
DEPTH = dict(min_distance_m=.15, max_distance_m=2.0)
PERCEPTION = dict(min_region_area_px=500, gaussian_kernel=5, morph_open_kernel=3,
                  morph_close_kernel=5, brightness_weight=.35, saturation_weight=.20,
                  texture_weight=.25, reflection_weight=.20, brightness_scale=80,
                  saturation_scale=80, texture_scale=80, reflection_threshold=235,
                  candidate_threshold=.17)


def make_camera(**kwargs):
    world = VirtualWorld(640, 480)
    world.add_target(0.0, 0.0, 0.05, depth_m=0.8, **kwargs)
    camera = MockCamera(640, 480, 30, world=world)
    camera.start()
    return camera


class PerceptionTests(unittest.TestCase):
    def test_reference_then_dirty_scene(self):
        camera = make_camera()
        roi = GroundROI(ROI, DEPTH)
        detector = WetRegionDetector(PERCEPTION, .2)
        image, depth, offset = roi.extract(camera.get_frame())
        detector.build_reference([image.copy()] * 3)
        self.assertEqual(detector.detect(image, depth, roi.valid_depth(depth), offset), [])
        camera.clean_reference_mode = False
        image, depth, offset = roi.extract(camera.get_frame())
        regions = detector.detect(image, depth, roi.valid_depth(depth), offset)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0].area_px, 500)
        self.assertAlmostEqual(regions[0].depth_m, .8)
        regions[0].validate()

    def test_clean_scene_produces_no_candidate(self):
        camera = make_camera(pollution_fraction=0.0)
        roi = GroundROI(ROI, DEPTH)
        detector = WetRegionDetector(PERCEPTION, .2)
        image, depth, offset = roi.extract(camera.get_frame())
        detector.build_reference([image.copy()] * 3)
        camera.clean_reference_mode = False
        image, depth, offset = roi.extract(camera.get_frame())
        self.assertEqual(detector.detect(image, depth, roi.valid_depth(depth), offset), [])

    def test_invalid_depth_region_is_kept_without_depth(self):
        camera = make_camera()
        roi = GroundROI(ROI, DEPTH)
        detector = WetRegionDetector(PERCEPTION, .2)
        image, depth, offset = roi.extract(camera.get_frame())
        detector.build_reference([image.copy()] * 3)
        camera.clean_reference_mode = False
        image, depth, offset = roi.extract(camera.get_frame())
        no_valid_depth = roi.valid_depth(depth) & False
        regions = detector.detect(image, depth, no_valid_depth, offset)
        self.assertEqual(len(regions), 1)
        self.assertIsNone(regions[0].depth_m)
        self.assertEqual(regions[0].valid_depth_ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
