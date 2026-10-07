import unittest
from camera.realsense_camera import MockCamera
from perception.ground_roi import GroundROI
from perception.wet_region_detector import WetRegionDetector


class PerceptionTests(unittest.TestCase):
    def test_reference_then_dirty_scene(self):
        camera = MockCamera()
        roi = GroundROI(dict(x_start=40, x_end=600, y_start=180, y_end=480),
                        dict(min_distance_m=.15, max_distance_m=2))
        config = dict(min_region_area_px=500, gaussian_kernel=5, morph_open_kernel=3,
            morph_close_kernel=5, brightness_weight=.35, saturation_weight=.20,
            texture_weight=.25, reflection_weight=.20, brightness_scale=80,
            saturation_scale=80, texture_scale=80, reflection_threshold=235,
            candidate_threshold=.17)
        detector = WetRegionDetector(config, .2)
        frame = camera.get_frame()
        image, depth, offset = roi.extract(frame)
        detector.build_reference([image.copy()] * 3)
        self.assertEqual(detector.detect(image, depth, roi.valid_depth(depth), offset), [])
        camera.clean_reference_mode = False
        image, depth, offset = roi.extract(camera.get_frame())
        regions = detector.detect(image, depth, roi.valid_depth(depth), offset)
        self.assertEqual(len(regions), 1)
        self.assertGreater(regions[0].area_px, 500)
        self.assertAlmostEqual(regions[0].depth_m, .8)
