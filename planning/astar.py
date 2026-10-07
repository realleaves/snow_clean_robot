"""Eight-connected A* over an artificial occupancy grid."""
import heapq
import math

from utils.errors import NoPathError, PlanningError

ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


class AStar:
    """Grid A* that refuses to start or end in an obstacle and never cuts corners."""

    def __init__(self, allow_diagonal: bool = True):
        self.allow_diagonal = bool(allow_diagonal)
        self.expanded = 0

    # ------------------------------------------------------------------ helpers
    def steps(self) -> tuple[tuple[int, int], ...]:
        return ORTHOGONAL + DIAGONAL if self.allow_diagonal else ORTHOGONAL

    def heuristic(self, cell, goal) -> float:
        dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        if self.allow_diagonal:
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)
        return float(dx + dy)

    def _corner_blocked(self, grid_map, cell, dx, dy) -> bool:
        if not (dx and dy):
            return False
        return (not grid_map.traversable((cell[0] + dx, cell[1]))
                or not grid_map.traversable((cell[0], cell[1] + dy)))

    # ------------------------------------------------------------------ planning
    def plan(self, grid_map, start, goal) -> list[tuple[int, int]]:
        """Return the cell path from *start* to *goal*, or ``[]`` when unreachable."""
        if grid_map is None:
            raise PlanningError('a grid map is required')
        if not grid_map.in_bounds(start) or not grid_map.in_bounds(goal):
            return []
        if not grid_map.traversable(start) or not grid_map.traversable(goal):
            return []
        if tuple(start) == tuple(goal):
            return [tuple(start)]
        start, goal = tuple(start), tuple(goal)
        self.expanded = 0
        queue = [(self.heuristic(start, goal), 0.0, start)]
        best = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > best.get(cell, math.inf) + 1e-9:
                continue
            self.expanded += 1
            if cell == goal:
                return self._reconstruct(parent, start, goal)
            for dx, dy in self.steps():
                nxt = (cell[0] + dx, cell[1] + dy)
                if not grid_map.traversable(nxt):
                    continue
                if self._corner_blocked(grid_map, cell, dx, dy):
                    continue
                new_cost = cost + math.hypot(dx, dy)
                if new_cost < best.get(nxt, math.inf) - 1e-12:
                    best[nxt], parent[nxt] = new_cost, cell
                    heapq.heappush(queue, (new_cost + self.heuristic(nxt, goal), new_cost, nxt))
        return []

    @staticmethod
    def _reconstruct(parent, start, goal) -> list[tuple[int, int]]:
        result = [goal]
        while result[-1] != start:
            result.append(parent[result[-1]])
        return result[::-1]

    def plan_or_raise(self, grid_map, start, goal) -> list[tuple[int, int]]:
        path = self.plan(grid_map, start, goal)
        if not path:
            raise NoPathError(f'no path from {tuple(start)} to {tuple(goal)}')
        return path

    def path_cost(self, path) -> float:
        return float(sum(math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                         for i in range(len(path) - 1)))
