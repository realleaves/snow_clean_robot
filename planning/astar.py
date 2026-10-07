import heapq
import math
from mapping.grid_map import GridMap


class AStar:
    def __init__(self, allow_diagonal: bool = True):
        self.allow_diagonal = allow_diagonal

    def plan(self, grid_map: GridMap, start: tuple[int, int],
             goal: tuple[int, int]) -> list[tuple[int, int]]:
        if not grid_map.traversable(start) or not grid_map.traversable(goal):
            return []
        if start == goal:
            return [start]
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diagonal:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        def heuristic(cell):
            dx, dy = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy) if self.allow_diagonal else dx + dy
        queue = [(heuristic(start), 0.0, start)]
        best = {start: 0.0}
        parent = {}
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > best[cell] + 1e-9:
                continue
            if cell == goal:
                result = [goal]
                while result[-1] != start:
                    result.append(parent[result[-1]])
                return result[::-1]
            for dx, dy in steps:
                nxt = (cell[0] + dx, cell[1] + dy)
                if not grid_map.traversable(nxt):
                    continue
                if dx and dy and (not grid_map.traversable((cell[0] + dx, cell[1])) or
                                  not grid_map.traversable((cell[0], cell[1] + dy))):
                    continue  # no diagonal corner cutting
                new_cost = cost + math.hypot(dx, dy)
                if new_cost < best.get(nxt, math.inf):
                    best[nxt], parent[nxt] = new_cost, cell
                    heapq.heappush(queue, (new_cost + heuristic(nxt), new_cost, nxt))
        return []
