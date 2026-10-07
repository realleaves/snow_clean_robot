import unittest
import numpy as np
from mapping.grid_map import GridMap
from planning.astar import AStar


class AStarTests(unittest.TestCase):
    def test_empty_and_same_cell(self):
        grid = GridMap(np.zeros((3, 3), np.uint8))
        self.assertEqual(AStar().plan(grid, (0, 0), (2, 2)), [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(AStar().plan(grid, (0, 0), (0, 0)), [(0, 0)])
        self.assertEqual(AStar().plan(grid, (-1, 0), (0, 0)), [])

    def test_obstacle_and_no_corner_cutting(self):
        grid = GridMap(np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]], np.uint8))
        path = AStar().plan(grid, (0, 0), (2, 0))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertTrue(all(grid.traversable(p) for p in path))
        blocked = GridMap(np.array([[0, 1], [1, 0]], np.uint8))
        self.assertEqual(AStar().plan(blocked, (0, 0), (1, 1)), [])
        self.assertEqual(AStar().plan(grid, (0, 0), (1, 0)), [])
