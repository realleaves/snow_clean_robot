"""tests/test_grid_map.py -- artificial 2D occupancy map: build, query, mutate, I/O."""
import unittest

import numpy as np
import pytest
import yaml

from mapping.grid_map import CELL_FREE, CELL_OBSTACLE, GridMap
from tests.helpers import MAP_CFG
from utils.errors import ConfigError, PlanningError


def config_grid() -> GridMap:
    """The configured 40x40 map with resolution 0.1 m and origin (-1, -1)."""
    return GridMap.from_config(MAP_CFG)


class GridMapBuilderTests(unittest.TestCase):
    def test_from_config_builds_the_configured_grid(self):
        grid = config_grid()
        self.assertEqual((grid.width, grid.height), (40, 40))
        self.assertEqual(grid.cells.shape, (40, 40))
        self.assertEqual(grid.resolution_m, pytest.approx(0.1))
        self.assertEqual((grid.origin_x, grid.origin_y), pytest.approx((-1.0, -1.0)))
        self.assertEqual(grid.obstacle_count, 0)
        self.assertTrue(grid.traversable((0, 0)))
        self.assertTrue(grid.traversable((39, 39)))

    def test_empty_classmethod(self):
        grid = GridMap.empty(4, 3, 0.2, 1.0, 2.0)
        self.assertEqual((grid.width, grid.height), (4, 3))
        self.assertEqual(grid.cells.shape, (3, 4))
        self.assertEqual(grid.resolution_m, pytest.approx(0.2))
        self.assertEqual((grid.origin_x, grid.origin_y), pytest.approx((1.0, 2.0)))
        self.assertEqual(grid.obstacle_count, 0)
        self.assertEqual(grid.cells.dtype, np.uint8)

    def test_invalid_config_raises_config_error(self):
        for missing in ('width', 'height', 'resolution_m'):
            cfg = dict(MAP_CFG)
            del cfg[missing]
            with self.assertRaises(ConfigError):
                GridMap.from_config(cfg)
        with self.assertRaises(ConfigError):
            GridMap.from_config(dict(MAP_CFG, resolution_m=0.0))
        with self.assertRaises(ConfigError):
            GridMap.from_config(dict(MAP_CFG, resolution_m=-0.1))

    def test_invalid_cells_raise_config_error(self):
        with self.assertRaises(ConfigError):
            GridMap(np.array([[0, 2]], np.uint8))              # value other than 0/1
        with self.assertRaises(ConfigError):
            GridMap(np.array([[0, -1]], np.int8))
        with self.assertRaises(ConfigError):
            GridMap(np.array([0, 1, 0], np.uint8))             # 1-D
        with self.assertRaises(ConfigError):
            GridMap(np.array(1, np.uint8))                     # 0-D
        with self.assertRaises(ConfigError):
            GridMap(np.zeros((0, 0), np.uint8))                # empty

    def test_invalid_resolution_raises_config_error(self):
        for bad in (0.0, -0.5):
            with self.assertRaises(ConfigError):
                GridMap(np.zeros((2, 2), np.uint8), bad)


class GridMapTransformTests(unittest.TestCase):
    def test_world_to_grid_grid_to_world_round_trip(self):
        grid = config_grid()
        self.assertEqual(grid.world_to_grid(-0.95, -0.95), (0, 0))
        for x, y in ((-0.95, -0.95), (-0.5, 0.24), (0.0, 0.0), (2.94, -0.06), (2.9, 2.9)):
            cell = grid.world_to_grid(x, y)
            back_x, back_y = grid.grid_to_world(cell)
            # A cell centre is never further than half a cell away (+ float slack).
            self.assertLessEqual(abs(back_x - x), grid.resolution_m / 2 + 1e-9)
            self.assertLessEqual(abs(back_y - y), grid.resolution_m / 2 + 1e-9)

    def test_grid_to_world_returns_cell_centres(self):
        grid = config_grid()
        self.assertEqual(grid.grid_to_world((0, 0)), pytest.approx((-0.95, -0.95)))
        self.assertEqual(grid.grid_to_world((3, 4)), pytest.approx((-0.65, -0.55)))
        self.assertEqual(grid.grid_to_world((39, 39)), pytest.approx((2.95, 2.95)))

    def test_boundary_coordinates(self):
        grid = config_grid()
        # Exactly on the origin boundary -> first cell.
        self.assertEqual(grid.world_to_grid(-1.0, -1.0), (0, 0))
        self.assertTrue(grid.in_bounds(grid.world_to_grid(-1.0, -1.0)))
        # A hair inside / outside the upper edge of the map.
        self.assertEqual(grid.world_to_grid(3.0 - 1e-9, 3.0 - 1e-9), (39, 39))
        self.assertTrue(grid.in_bounds(grid.world_to_grid(3.0 - 1e-9, 3.0 - 1e-9)))
        self.assertEqual(grid.world_to_grid(3.0, 3.0), (40, 40))
        self.assertFalse(grid.in_bounds(grid.world_to_grid(3.0, 3.0)))
        # A hair below the origin -> negative cell, out of bounds.
        self.assertEqual(grid.world_to_grid(-1.0 - 1e-9, -1.0), (-1, 0))
        self.assertFalse(grid.in_bounds(grid.world_to_grid(-1.0 - 1e-9, -1.0)))
        # world_bounds agrees with the usable extents.
        self.assertEqual(grid.world_bounds(), pytest.approx((-1.0, -1.0, 3.0, 3.0)))


class GridMapQueryTests(unittest.TestCase):
    def test_in_bounds_for_negative_and_too_large_cells(self):
        grid = GridMap.empty(4, 3)
        self.assertTrue(grid.in_bounds((0, 0)))
        self.assertTrue(grid.in_bounds((3, 2)))
        self.assertFalse(grid.in_bounds((-1, 0)))
        self.assertFalse(grid.in_bounds((0, -1)))
        self.assertFalse(grid.in_bounds((4, 0)))
        self.assertFalse(grid.in_bounds((0, 3)))
        self.assertFalse(grid.in_bounds(('a', 0)))

    def test_traversable_and_is_obstacle(self):
        grid = GridMap.empty(3, 3)
        self.assertTrue(grid.traversable((1, 1)))
        self.assertFalse(grid.is_obstacle((1, 1)))
        self.assertFalse(grid.traversable((-1, 0)))
        grid.set_obstacle((1, 1))
        self.assertFalse(grid.traversable((1, 1)))
        self.assertTrue(grid.is_obstacle((1, 1)))
        for cell in ((-1, 0), (3, 0), (0, 3)):
            with self.assertRaises(PlanningError):
                grid.is_obstacle(cell)

    def test_set_and_clear_obstacles(self):
        grid = GridMap.empty(4, 4)
        grid.set_obstacle((1, 1))
        self.assertEqual(grid.obstacle_count, 1)
        self.assertEqual(grid.cells[1, 1], CELL_OBSTACLE)
        grid.clear_obstacle((1, 1))
        self.assertEqual(grid.obstacle_count, 0)
        self.assertEqual(grid.cells[1, 1], CELL_FREE)
        grid.set_obstacles([(0, 0), (2, 3)])
        self.assertEqual(grid.obstacle_count, 2)
        grid.set_obstacles([(0, 0), (2, 3)], occupied=False)
        self.assertEqual(grid.obstacle_count, 0)

    def test_set_obstacle_out_of_bounds_raises(self):
        grid = GridMap.empty(3, 3)
        with self.assertRaises(PlanningError):
            grid.set_obstacle((3, 0))
        with self.assertRaises(PlanningError):
            grid.clear_obstacle((0, -1))

    def test_inflate_obstacles(self):
        grid = GridMap.empty(5, 5, 0.2, 0.5, -0.5)
        grid.set_obstacle((2, 2))
        inflated = grid.inflate_obstacles(1)
        self.assertIsNot(inflated, grid)
        self.assertEqual(inflated.obstacle_count, 9)
        self.assertEqual(grid.obstacle_count, 1)             # original untouched
        self.assertFalse(grid.is_obstacle((1, 1)))
        self.assertTrue(inflated.is_obstacle((1, 1)))
        self.assertEqual(inflated.resolution_m, grid.resolution_m)
        self.assertEqual((inflated.origin_x, inflated.origin_y),
                         (grid.origin_x, grid.origin_y))
        # radius 0 is a copy; a negative radius is a config error.
        self.assertEqual(grid.inflate_obstacles(0).obstacle_count, 1)
        corner = GridMap.empty(4, 4)
        corner.set_obstacle((0, 0))
        self.assertEqual(corner.inflate_obstacles(1).obstacle_count, 4)
        with self.assertRaises(ConfigError):
            grid.inflate_obstacles(-1)

    def test_clamp_cell(self):
        open_grid = GridMap.empty(5, 5)
        self.assertEqual(open_grid.clamp_cell((2, 2)), (2, 2))
        # An out-of-bounds but otherwise free request clamps to the nearest free cell.
        self.assertEqual(open_grid.clamp_cell((-1, -1)), (0, 0))
        stuck = GridMap.empty(5, 5)
        stuck.set_obstacle((0, 0))
        self.assertEqual(stuck.clamp_cell((0, 0)), (1, 0))
        blocked = GridMap(np.ones((3, 3), np.uint8))
        self.assertIsNone(blocked.clamp_cell((1, 1)))
        self.assertIsNone(GridMap(np.ones((1, 1), np.uint8)).clamp_cell((0, 0)))

    def test_as_text(self):
        grid = GridMap(np.array([[0, 1], [1, 0]], np.uint8))
        self.assertEqual(grid.as_text(), '.#\n#.')
        self.assertEqual(grid.as_text(free='o', obstacle='X'), 'oX\nXo')


# ------------------------------------------------------------------- map files
def test_load_npy_round_trip(tmp_path):
    cells = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], np.uint8)
    path = tmp_path / 'map.npy'
    np.save(path, cells)
    loaded = GridMap.load(path)
    assert np.array_equal(loaded.cells, cells)
    assert loaded.resolution_m == pytest.approx(0.1)


def test_load_yaml_with_cells_and_resolution(tmp_path):
    path = tmp_path / 'map.yaml'
    path.write_text(yaml.safe_dump({'cells': [[0, 1], [1, 0]], 'resolution_m': 0.2}),
                    encoding='utf-8')
    loaded = GridMap.load(path)
    assert np.array_equal(loaded.cells, np.array([[0, 1], [1, 0]], np.uint8))
    assert loaded.resolution_m == pytest.approx(0.2)
    # An explicit resolution argument wins when the YAML does not carry one.
    plain = tmp_path / 'plain.yml'
    plain.write_text(yaml.safe_dump({'cells': [[0, 0], [1, 1]]}), encoding='utf-8')
    assert GridMap.load(plain, resolution_m=0.5).resolution_m == pytest.approx(0.5)


def test_load_txt_loadtxt_map(tmp_path):
    path = tmp_path / 'map.txt'
    np.savetxt(path, np.array([[0, 1], [1, 0]], np.uint8), fmt='%d')
    loaded = GridMap.load(path)
    assert np.array_equal(loaded.cells, np.array([[0, 1], [1, 0]], np.uint8))


def test_load_corrupt_or_missing_map_raises(tmp_path):
    with pytest.raises(ConfigError):
        GridMap.load(tmp_path / 'missing.npy')
    bad_yaml = tmp_path / 'bad.yaml'
    bad_yaml.write_text(yaml.safe_dump({'resolution_m': 0.1}), encoding='utf-8')
    with pytest.raises(ConfigError):
        GridMap.load(bad_yaml)
    bad_cells = tmp_path / 'bad.npy'
    np.save(bad_cells, np.array([[0, 7]], np.uint8))
    with pytest.raises(ConfigError):
        GridMap.load(bad_cells)


if __name__ == '__main__':
    unittest.main()
