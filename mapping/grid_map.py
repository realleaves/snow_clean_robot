from pathlib import Path
import numpy as np
import yaml


class GridMap:
    def __init__(self, cells: np.ndarray, resolution_m: float = 0.1,
                 origin_x: float = 0, origin_y: float = 0):
        self.cells = np.asarray(cells, dtype=np.uint8)
        if self.cells.ndim != 2 or not self.cells.size or not np.isin(self.cells, (0, 1)).all():
            raise ValueError('grid must be nonempty 2D with values 0 and 1')
        if resolution_m <= 0:
            raise ValueError('grid resolution must be positive')
        self.resolution_m, self.origin_x, self.origin_y = resolution_m, origin_x, origin_y

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(np.zeros((cfg['height'], cfg['width']), np.uint8),
                   cfg['resolution_m'], cfg['origin_x'], cfg['origin_y'])

    @classmethod
    def load(cls, path: str | Path, resolution_m: float = 0.1):
        path = Path(path)
        if path.suffix == '.npy':
            cells = np.load(path)
        elif path.suffix in ('.yaml', '.yml'):
            obj = yaml.safe_load(path.read_text())
            cells = obj['cells']
            resolution_m = obj.get('resolution_m', resolution_m)
        else:
            cells = np.loadtxt(path, dtype=np.uint8)
        return cls(cells, resolution_m)

    def in_bounds(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= y < self.cells.shape[0] and 0 <= x < self.cells.shape[1]

    def traversable(self, cell: tuple[int, int]) -> bool:
        return self.in_bounds(cell) and self.cells[cell[1], cell[0]] == 0

    def set_obstacle(self, cell: tuple[int, int], occupied: bool = True) -> None:
        if not self.in_bounds(cell):
            raise ValueError('cell outside map')
        self.cells[cell[1], cell[0]] = int(occupied)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return (int(np.floor((x - self.origin_x) / self.resolution_m)),
                int(np.floor((y - self.origin_y) / self.resolution_m)))

    def grid_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        return (self.origin_x + (cell[0] + .5) * self.resolution_m,
                self.origin_y + (cell[1] + .5) * self.resolution_m)
