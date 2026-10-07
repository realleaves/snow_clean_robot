import math
import numpy as np


def pixel_to_robot_ground(u: int, v: int, depth_m: float,
                          intrinsics: tuple[float, float, float, float],
                          camera_to_robot: np.ndarray) -> tuple[float, float]:
    """Project a depth pixel using a measured 4x4 camera-to-robot transform."""
    fx, fy, cx, cy = intrinsics
    if depth_m <= 0 or fx <= 0 or fy <= 0 or camera_to_robot.shape != (4, 4):
        raise ValueError('valid depth, intrinsics and 4x4 extrinsics required')
    camera_point = np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy,
                             depth_m, 1.0])
    robot_point = camera_to_robot @ camera_point
    return float(robot_point[0]), float(robot_point[1])


def robot_to_world(forward: float, left: float, pose) -> tuple[float, float]:
    return (pose.x + math.cos(pose.yaw) * forward - math.sin(pose.yaw) * left,
            pose.y + math.sin(pose.yaw) * forward + math.cos(pose.yaw) * left)
