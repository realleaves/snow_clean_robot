"""Artificial 2D occupancy grid map (0 = traversable, 1 = obstacle)."""
from pathlib import Path

import numpy as np
import yaml

from utils.errors import ConfigError, PlanningError

CELL_FREE, CELL_OBSTACLE = 0, 1


class GridMap:
    def __init__(self, cells, resolution_m: float = 0.1,
                 origin_x: float = 0, origin_y: float = 0):
        array = np.asarray(cells)
        if array.ndim != 2 or array.size == 0:
            raise ConfigError('grid must be a nonempty 2D array')
        if not np.isin(array, (CELL_FREE, CELL_OBSTACLE)).all():
            raise ConfigError('grid values must be 0 (free) or 1 (obstacle)')
        if resolution_m <= 0:
            raise ConfigError('grid resolution must be positive')
        self.cells = array.astype(np.uint8, copy=True)
        self.resolution_m = float(resolution_m)
        self.origin_x, self.origin_y = float(origin_x), float(origin_y)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_config(cls, cfg: dict):
        for key in ('width', 'height', 'resolution_m'):
            if key not in cfg:
                raise ConfigError(f'map config is missing {key}')
        return cls(np.zeros((int(cfg['height']), int(cfg['width'])), np.uint8),
                   cfg['resolution_m'], cfg.get('origin_x', 0.0), cfg.get('origin_y', 0.0))

    @classmethod
    def empty(cls, width: int, height: int, resolution_m: float = 0.1,
              origin_x: float = 0.0, origin_y: float = 0.0):
        return cls(np.zeros((height, width), np.uint8), resolution_m, origin_x, origin_y)

    @classmethod
    def load(cls, path: str | Path, resolution_m: float = 0.1):
        path = Path(path)
        if not path.exists():
            raise ConfigError(f'map file does not exist: {path}')
        try:
            if path.suffix == '.npy':
                cells = np.load(path)
            elif path.suffix in ('.yaml', '.yml'):
                obj = yaml.safe_load(path.read_text(encoding='utf-8'))
                cells = obj['cells']
                resolution_m = obj.get('resolution_m', resolution_m)
            else:
                cells = np.loadtxt(path, dtype=np.uint8)
        except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
            raise ConfigError(f'cannot load map {path}: {exc}') from exc
        return cls(cells, resolution_m)

    # ------------------------------------------------------------------ queries
    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def obstacle_count(self) -> int:
        return int(np.count_nonzero(self.cells))

    def in_bounds(self, cell) -> bool:
        try:
            x, y = int(cell[0]), int(cell[1])
        except (TypeError, ValueError, IndexError):
            return False
        return 0 <= y < self.height and 0 <= x < self.width

    def traversable(self, cell) -> bool:
        return self.in_bounds(cell) and self.cells[int(cell[1]), int(cell[0])] == CELL_FREE

    def is_obstacle(self, cell) -> bool:
        if not self.in_bounds(cell):
            raise PlanningError(f'cell {cell} is outside the map')
        return self.cells[int(cell[1]), int(cell[0])] == CELL_OBSTACLE

    # ------------------------------------------------------------------ mutation
    def set_obstacle(self, cell, occupied: bool = True) -> None:
        if not self.in_bounds(cell):
            raise PlanningError(f'cell {cell} is outside the map')
        self.cells[int(cell[1]), int(cell[0])] = CELL_OBSTACLE if occupied else CELL_FREE

    def clear_obstacle(self, cell) -> None:
        self.set_obstacle(cell, False)

    def set_obstacles(self, cells, occupied: bool = True) -> None:
        for cell in cells:
            self.set_obstacle(cell, occupied)

    def inflate_obstacles(self, radius_cells: int = 1) -> 'GridMap':
        """Return a copy with every obstacle dilated by *radius_cells* (safety margin)."""
        if radius_cells < 0:
            raise ConfigError('inflation radius must be non-negative')
        inflated = self.cells.copy()
        obstacles = np.argwhere(self.cells == CELL_OBSTACLE)
        for y, x in obstacles:
            y0, y1 = max(0, y - radius_cells), min(self.height, y + radius_cells + 1)
            x0, x1 = max(0, x - radius_cells), min(self.width, x + radius_cells + 1)
            inflated[y0:y1, x0:x1] = CELL_OBSTACLE
        return GridMap(inflated, self.resolution_m, self.origin_x, self.origin_y)

    def copy(self) -> 'GridMap':
        return GridMap(self.cells.copy(), self.resolution_m, self.origin_x, self.origin_y)

    # ------------------------------------------------------------------ transforms
    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor((x - self.origin_x) / self.resolution_m)),
                int(np.floor((y - self.origin_y) / self.resolution_m)))

    def grid_to_world(self, cell) -> tuple[float, float]:
        return (self.origin_x + (int(cell[0]) + 0.5) * self.resolution_m,
                self.origin_y + (int(cell[1]) + 0.5) * self.resolution_m)

    def world_bounds(self) -> tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` covered by the grid."""
        return (self.origin_x, self.origin_y,
                self.origin_x + self.width * self.resolution_m,
                self.origin_y + self.height * self.resolution_m)

    def clamp_cell(self, cell) -> tuple[int, int] | None:
        """Nearest traversable cell to *cell*, or ``None`` when the map is full."""
        if self.traversable(cell):
            return (int(cell[0]), int(cell[1]))
        for radius in range(1, max(self.width, self.height)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    candidate = (int(cell[0]) + dx, int(cell[1]) + dy)
                    if self.traversable(candidate):
                        return candidate
        return None

    def as_text(self, free: str = '.', obstacle: str = '#') -> str:
        return '\n'.join(''.join(obstacle if v else free for v in row) for row in self.cells)
