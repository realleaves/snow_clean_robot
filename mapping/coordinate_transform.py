"""Pixel -> camera -> robot -> world -> grid coordinate chain.

The chain is intentionally explicit so every stage can be unit-tested against a
hand-computed value:

``pixel_to_camera`` -- ``camera_to_robot`` -- ``robot_to_world`` -- ``world_to_grid``

Convention (documented because it is a frequent source of sign errors):

* camera optical frame: ``x`` right, ``y`` down, ``z`` forward, metres;
* robot frame: ``forward`` along the robot's heading, ``left`` to its left, metres;
* world frame: the artificial 2D grid map frame, metres.
"""
import math

import numpy as np

from utils.errors import ConfigError, DepthError, PerceptionError

MAX_GROUND_RANGE_M = 5.0


def _unpack_intrinsics(intrinsics) -> tuple[float, float, float, float]:
    try:
        fx, fy, cx, cy = (float(v) for v in intrinsics)
    except (TypeError, ValueError) as exc:
        raise ConfigError('intrinsics must be (fx, fy, cx, cy)') from exc
    if fx <= 0 or fy <= 0:
        raise ConfigError('focal lengths fx and fy must be positive')
    return fx, fy, cx, cy


def _validate_extrinsics(camera_to_robot) -> np.ndarray:
    matrix = np.asarray(camera_to_robot, dtype=float)
    if matrix.shape != (4, 4):
        raise ConfigError('camera_to_robot must be a 4x4 matrix')
    if not np.isfinite(matrix).all():
        raise ConfigError('camera_to_robot contains non-finite values')
    return matrix


def pixel_to_camera(u: float, v: float, depth_m: float,
                    intrinsics) -> tuple[float, float, float]:
    """Back-project a pixel with a measured depth into the camera frame."""
    fx, fy, cx, cy = _unpack_intrinsics(intrinsics)
    if depth_m is None or not math.isfinite(float(depth_m)) or float(depth_m) <= 0:
        raise DepthError(f'invalid depth {depth_m!r}; a positive metric depth is required')
    depth = float(depth_m)
    return ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)


def camera_to_robot(camera_point, camera_to_robot_matrix) -> tuple[float, float, float]:
    matrix = _validate_extrinsics(camera_to_robot_matrix)
    point = np.asarray([*camera_point, 1.0], dtype=float)
    robot = matrix @ point
    return float(robot[0]), float(robot[1]), float(robot[2])


def pixel_to_robot(u: float, v: float, depth_m: float, intrinsics,
                   extrinsics) -> tuple[float, float, float]:
    """Full pixel -> camera -> robot chain."""
    return camera_to_robot(pixel_to_camera(u, v, depth_m, intrinsics), extrinsics)


def pixel_to_robot_ground(u: int, v: int, depth_m: float,
                          intrinsics: tuple[float, float, float, float],
                          camera_to_robot_matrix: np.ndarray) -> tuple[float, float]:
    """Project a depth pixel to the robot ground plane as ``(forward, left)``."""
    forward, left, _ = pixel_to_robot(u, v, depth_m, intrinsics, camera_to_robot_matrix)
    return forward, left


def robot_to_world(forward: float, left: float, pose) -> tuple[float, float]:
    """Rotate and translate a robot-frame point into the world frame."""
    if pose is None:
        raise PerceptionError('a robot pose is required for robot_to_world')
    for name, value in (('forward', forward), ('left', left), ('x', pose.x),
                        ('y', pose.y), ('yaw', pose.yaw)):
        if not math.isfinite(float(value)):
            raise PerceptionError(f'robot_to_world received a non-finite {name}')
    return (pose.x + math.cos(pose.yaw) * forward - math.sin(pose.yaw) * left,
            pose.y + math.sin(pose.yaw) * forward + math.cos(pose.yaw) * left)


def world_to_robot(x: float, y: float, pose) -> tuple[float, float]:
    """Inverse of :func:`robot_to_world`."""
    dx, dy = x - pose.x, y - pose.y
    return (math.cos(pose.yaw) * dx + math.sin(pose.yaw) * dy,
            -math.sin(pose.yaw) * dx + math.cos(pose.yaw) * dy)


def world_to_grid(x: float, y: float, grid_map) -> tuple[int, int]:
    return grid_map.world_to_grid(x, y)


def pixel_to_world(u: float, v: float, depth_m: float, intrinsics, extrinsics,
                   pose, grid_map=None):
    """Full pixel -> camera -> robot -> world (-> grid) chain."""
    forward, left = pixel_to_robot_ground(u, v, depth_m, intrinsics, extrinsics)
    x, y = robot_to_world(forward, left, pose)
    if grid_map is None:
        return x, y
    return x, y, grid_map.world_to_grid(x, y)


def ground_range_m(forward: float, left: float) -> float:
    return math.hypot(forward, left)
