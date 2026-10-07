"""tests/test_path_planner.py -- world-frame path planning on the artificial map."""
import math

import numpy as np
import pytest

from decision.cleaning_task import CleaningTask
from mapping.grid_map import GridMap
from mapping.pose_provider import RobotPose
from planning.astar import AStar
from planning.path_planner import PathPlanner
from tests.helpers import MAP_CFG
from utils.errors import NoPathError


def make_planner(cells=None, allow_diagonal=True, resolution=0.1, origin=(-1.0, -1.0)):
    grid = GridMap(np.zeros((40, 40), np.uint8) if cells is None else cells,
                   resolution, origin[0], origin[1])
    return PathPlanner(grid, AStar(allow_diagonal)), grid


def make_task(x, y):
    return CleaningTask(1, x, y, 0.5, 'MEDIUM')


def test_config_grid_is_reachable_from_origin():
    planner, _ = make_planner()
    path = planner.plan_to_target(RobotPose(0, 0, 0), 0.0, 0.8)
    assert path
    assert planner.path_length_m(path) > 0
    assert planner.last_failure is None


def test_plan_to_task_delegates_to_target():
    planner, _ = make_planner()
    task = make_task(0.0, 0.8)
    assert planner.plan_to_task(RobotPose(0, 0, 0), task) == \
        planner.plan_to_target(RobotPose(0, 0, 0), 0.0, 0.8)


def test_plan_to_pose_uses_pose_coordinates():
    planner, _ = make_planner()
    assert planner.plan_to_pose(RobotPose(0, 0, 0), RobotPose(0.0, 0.8, 1.0))


def test_start_equal_goal_returns_single_waypoint():
    planner, _ = make_planner()
    path = planner.plan_to_target(RobotPose(0, 0, 0), 0.0, 0.0)
    assert len(path) == 1


def test_obstacle_wall_blocks_and_reports_failure():
    cells = np.zeros((40, 40), np.uint8)
    for index in range(40):       # anti-diagonal wall separating the two corners
        if index + 1 < 40:
            cells[index, index + 1] = 1
        cells[index, index] = 1
    planner, _ = make_planner(cells)
    # start (0, 0) and goal (0.8, 0.8) land on opposite sides of the wall
    path = planner.plan_to_target(RobotPose(-0.95, -0.95, 0.0), 0.85, 0.85)
    assert path == []
    assert 'no path' in (planner.last_failure or '')
    with pytest.raises(NoPathError):
        planner.plan_or_raise(RobotPose(-0.95, -0.95, 0.0), make_task(0.85, 0.85))


def test_start_inside_obstacle_is_snapped_to_a_free_cell():
    cells = np.zeros((40, 40), np.uint8)
    cells[10, 10] = 1
    planner, _ = make_planner(cells)
    path = planner.plan_to_target(RobotPose(-0.95, -0.95, 0.0), 0.0, 0.8)
    assert path
    assert planner.last_start_cell != (10, 10)


def test_fully_blocked_map_has_no_substitute_start():
    cells = np.ones((40, 40), np.uint8)
    planner, _ = make_planner(cells)
    assert planner.plan_to_target(RobotPose(0, 0, 0), 0.0, 0.8) == []
    assert 'not traversable' in (planner.last_failure or '')


def test_reachability_helper():
    planner, _ = make_planner()
    assert planner.is_reachable(RobotPose(0, 0, 0), 0.0, 0.8) is True
    cells = np.ones((40, 40), np.uint8)
    blocked_planner, _ = make_planner(cells)
    assert blocked_planner.is_reachable(RobotPose(0, 0, 0), 0.0, 0.8) is False


def test_cells_for_matches_gridmap_conversion():
    planner, grid = make_planner()
    start, goal = planner.cells_for(RobotPose(0, 0, 0), 0.0, 0.8)
    assert start == grid.world_to_grid(0, 0)
    assert goal == grid.world_to_grid(0.0, 0.8)


def test_path_length_is_euclidean():
    planner, _ = make_planner()
    path = planner.plan_to_target(RobotPose(0, 0, 0), 0.0, 0.8)
    manual = sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1))
    assert planner.path_length_m(path) == pytest.approx(manual)
    assert planner.path_length_m([]) == 0.0


def test_configured_map_parameters_are_used():
    planner, grid = make_planner()
    assert grid.width == MAP_CFG['width'] and grid.height == MAP_CFG['height']
    assert grid.resolution_m == pytest.approx(MAP_CFG['resolution_m'])
    assert grid.world_bounds()[0] == pytest.approx(MAP_CFG['origin_x'])


def test_replanning_after_map_change():
    cells = np.zeros((40, 40), np.uint8)
    planner, grid = make_planner(cells)
    assert planner.plan_to_target(RobotPose(-0.95, -0.95, 0.0), 0.85, 0.85)
    # block the whole row 18 (goal y = 19.5) so the target becomes unreachable
    grid.cells[18, :] = 1
    assert planner.plan_to_target(RobotPose(-0.95, -0.95, 0.0), 0.85, 0.85) == []
