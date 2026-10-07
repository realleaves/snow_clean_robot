"""tests/test_coordinate_transform.py -- hand-computed pixel/camera/robot/world chain.

Every expectation below is derived by hand from the documented convention:

* camera optical frame: ``x`` right, ``y`` down, ``z`` forward;
* the mock extrinsics map camera ``(x, y, z)`` to robot ``(forward=z, left=-x)``;
* world = pose translation/rotation of the robot-frame point.
"""
import math
import unittest

import numpy as np
import pytest

from camera.virtual_scene import CAMERA_TO_ROBOT
from mapping.coordinate_transform import (
    camera_to_robot, pixel_to_camera, pixel_to_robot_ground, pixel_to_world,
    robot_to_world, world_to_grid, world_to_robot,
)
from mapping.grid_map import GridMap
from mapping.pose_provider import RobotPose
from utils.errors import ConfigError, DepthError, PerceptionError

INTRINSICS = (500.0, 500.0, 320.0, 240.0)


class PixelToCameraTests(unittest.TestCase):
    def test_principal_point_and_cardinal_offsets(self):
        assert pixel_to_camera(320, 240, 0.8, INTRINSICS) == pytest.approx((0.0, 0.0, 0.8))
        # 100 px right of centre at fx=500, z=0.8 m -> 0.16 m right
        assert pixel_to_camera(420, 240, 0.8, INTRINSICS) == pytest.approx((0.16, 0.0, 0.8))
        # 100 px below centre at fy=500, z=0.8 m -> 0.16 m down
        assert pixel_to_camera(320, 340, 0.8, INTRINSICS) == pytest.approx((0.0, 0.16, 0.8))

    def test_invalid_depth_raises_depth_error(self):
        for bad in (0.0, -0.5, float('nan'), float('inf'), None):
            with self.assertRaises(DepthError):
                pixel_to_camera(320, 240, bad, INTRINSICS)

    def test_invalid_intrinsics_raise_config_error(self):
        for bad in ((0.0, 500.0, 320.0, 240.0), (500.0, -1.0, 320.0, 240.0)):
            with self.assertRaises(ConfigError):
                pixel_to_camera(320, 240, 0.8, bad)


class CameraToRobotTests(unittest.TestCase):
    def test_camera_point_maps_to_forward_left(self):
        # robot = (z, -x, -y)
        assert camera_to_robot((0.0, 0.0, 0.8), CAMERA_TO_ROBOT) == pytest.approx((0.8, 0.0, 0.0))
        assert camera_to_robot((0.16, 0.0, 0.8), CAMERA_TO_ROBOT) == pytest.approx((0.8, -0.16, 0.0))

    def test_principal_point_ground_projection(self):
        forward, left = pixel_to_robot_ground(320, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT)
        assert (forward, left) == pytest.approx((0.8, 0.0))

    def test_left_and_right_of_centre_keep_the_sign_convention(self):
        # u < cx is physically left of the optical axis -> positive robot "left".
        forward, left = pixel_to_robot_ground(270, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT)
        expected_left = -(270 - 320) * 0.8 / 500.0        # = +0.08 m
        assert expected_left > 0
        assert (forward, left) == pytest.approx((0.8, expected_left))

        forward, left = pixel_to_robot_ground(370, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT)
        assert left == pytest.approx(-(370 - 320) * 0.8 / 500.0)
        assert left < 0

    def test_near_and_far_depth_keep_the_sign_convention(self):
        near = pixel_to_robot_ground(270, 240, 0.3, INTRINSICS, CAMERA_TO_ROBOT)
        far = pixel_to_robot_ground(270, 240, 1.8, INTRINSICS, CAMERA_TO_ROBOT)
        assert near == pytest.approx((0.3, 0.03))
        assert far == pytest.approx((1.8, 0.18))
        assert near[1] > 0 and far[1] > 0

    def test_bad_extrinsics_raise_config_error(self):
        with self.assertRaises(ConfigError):
            camera_to_robot((0.0, 0.0, 0.8), np.eye(3))
        with self.assertRaises(ConfigError):
            camera_to_robot((0.0, 0.0, 0.8), np.full((4, 4), np.nan))
        with self.assertRaises(ConfigError):
            pixel_to_robot_ground(320, 240, 0.8, INTRINSICS, np.zeros((2, 2)))
        with self.assertRaises(ConfigError):
            pixel_to_robot_ground(320, 240, 0.8, (0.0, 500.0, 320.0, 240.0), CAMERA_TO_ROBOT)


class RobotWorldTests(unittest.TestCase):
    def test_robot_to_world_hand_computed_table(self):
        for yaw, expected in ((0.0, (3.0, 3.0)), (math.pi / 2, (2.0, 4.0)),
                              (math.pi, (1.0, 3.0)), (-math.pi / 2, (2.0, 2.0))):
            pose = RobotPose(2.0, 3.0, yaw)
            assert robot_to_world(1.0, 0.0, pose) == pytest.approx(expected)

    def test_lateral_offset_rotates_with_the_heading(self):
        assert robot_to_world(0.0, 1.0, RobotPose(2.0, 3.0, 0.0)) == pytest.approx((2.0, 4.0))
        assert robot_to_world(0.0, 1.0, RobotPose(2.0, 3.0, math.pi / 2)) == pytest.approx((1.0, 3.0))

    def test_world_to_robot_is_the_inverse(self):
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2, 0.7, -2.1):
            pose = RobotPose(1.25, -0.75, yaw)
            for forward, left in ((1.0, 0.0), (0.0, 0.5), (-0.3, -0.9), (2.0, -1.5)):
                x, y = robot_to_world(forward, left, pose)
                back = world_to_robot(x, y, pose)
                assert back[0] == pytest.approx(forward, abs=1e-9)
                assert back[1] == pytest.approx(left, abs=1e-9)

    def test_pixel_to_world_full_chain(self):
        pose = RobotPose(2.0, 3.0, 0.0)
        # principal point, depth 0.8 -> (forward=0.8, left=0) -> (2.8, 3.0)
        assert pixel_to_world(320, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT, pose) == \
            pytest.approx((2.8, 3.0))
        # a pixel 100 px left of centre adds +0.16 m of "left" -> y + 0.16
        assert pixel_to_world(220, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT, pose) == \
            pytest.approx((2.8, 3.16))

    def test_pixel_to_world_with_grid_returns_the_cell(self):
        grid = GridMap.empty(40, 40, 0.1, 0.0, 0.0)
        result = pixel_to_world(320, 240, 0.8, INTRINSICS, CAMERA_TO_ROBOT,
                                RobotPose(2.0, 3.0, 0.0), grid)
        assert len(result) == 3
        assert result[:2] == pytest.approx((2.8, 3.0))
        assert result[2] == grid.world_to_grid(result[0], result[1])
        # 2.8 / 0.1 is 27.999... in IEEE doubles, so floor() lands on cell 27.
        assert result[2] == (27, 30)

    def test_missing_or_non_finite_pose_raises(self):
        with self.assertRaises(PerceptionError):
            robot_to_world(1.0, 0.0, None)
        for pose in (RobotPose(float('nan'), 0.0, 0.0), RobotPose(0.0, float('inf'), 0.0),
                     RobotPose(0.0, 0.0, float('nan'))):
            with self.assertRaises(PerceptionError):
                robot_to_world(1.0, 0.0, pose)
        with self.assertRaises(PerceptionError):
            robot_to_world(float('nan'), 0.0, RobotPose(0.0, 0.0, 0.0))

    def test_world_to_grid_wrapper_matches_grid_map(self):
        grid = GridMap.empty(40, 40, 0.1, -1.0, -1.0)
        for x, y in ((0.0, 0.0), (-1.0, -1.0), (2.95, 2.95), (-1.05, 0.15)):
            assert world_to_grid(x, y, grid) == grid.world_to_grid(x, y)


if __name__ == '__main__':
    unittest.main()
