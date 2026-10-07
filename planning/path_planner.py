"""World-frame path planning on top of the cell-level A* planner."""
import math

from utils.errors import NoPathError, PlanningError


class PathPlanner:
    """Converts poses and targets into world-frame waypoint lists."""

    def __init__(self, grid_map, astar):
        self.grid_map, self.astar = grid_map, astar
        self.last_start_cell = None
        self.last_goal_cell = None
        self.last_failure = None

    # ------------------------------------------------------------------ helpers
    def cells_for(self, current_pose, target_x: float, target_y: float):
        start = self.grid_map.world_to_grid(current_pose.x, current_pose.y)
        goal = self.grid_map.world_to_grid(target_x, target_y)
        return start, goal

    def plan_cells(self, current_pose, target_x: float, target_y: float) -> list[tuple[int, int]]:
        start, goal = self.cells_for(current_pose, target_x, target_y)
        self.last_start_cell, self.last_goal_cell = start, goal
        if not self.grid_map.traversable(start):
            snapped = self.grid_map.clamp_cell(start)
            if snapped is None:
                self.last_failure = f'start cell {start} is not traversable and no substitute exists'
                return []
            self.last_start_cell = start = snapped
        path = self.astar.plan(self.grid_map, start, goal)
        self.last_failure = None if path else f'no path from {start} to {goal}'
        return path

    def plan_to_task(self, current_pose, task) -> list[tuple[float, float]]:
        """World-frame waypoints from the current pose to the task target."""
        return self.plan_to_target(current_pose, task.target_x, task.target_y)

    def plan_to_target(self, current_pose, target_x: float, target_y: float):
        return [self.grid_map.grid_to_world(cell)
                for cell in self.plan_cells(current_pose, target_x, target_y)]

    def plan_to_pose(self, current_pose, target_pose) -> list[tuple[float, float]]:
        return self.plan_to_target(current_pose, target_pose.x, target_pose.y)

    def plan_or_raise(self, current_pose, task) -> list[tuple[float, float]]:
        path = self.plan_to_task(current_pose, task)
        if not path:
            raise NoPathError(self.last_failure or 'no path to task')
        return path

    def path_length_m(self, path: list[tuple[float, float]]) -> float:
        return float(sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                         for i in range(len(path) - 1)))

    def is_reachable(self, current_pose, target_x: float, target_y: float) -> bool:
        return bool(self.plan_cells(current_pose, target_x, target_y))
