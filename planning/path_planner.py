from mapping.grid_map import GridMap
from planning.astar import AStar


class PathPlanner:
    def __init__(self, grid_map: GridMap, astar: AStar):
        self.grid_map, self.astar = grid_map, astar

    def plan_to_task(self, current_pose, task) -> list[tuple[float, float]]:
        start = self.grid_map.world_to_grid(current_pose.x, current_pose.y)
        goal = self.grid_map.world_to_grid(task.target_x, task.target_y)
        return [self.grid_map.grid_to_world(cell) for cell in self.astar.plan(self.grid_map, start, goal)]
